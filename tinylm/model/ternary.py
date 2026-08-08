"""g128 삼진 양자화 (v3에서 검증된 구현). AMP-안전 STE."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TernarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, group, thr_ratio, ste_clip, sparse34=False):
        O, I = w.shape
        G = I // group
        wg = w.reshape(O, G, group)
        aw = wg.abs()
        if sparse34:
            # (P016) 3:4 희소: group 을 4-블록으로 쪼개 각 블록에서 |w|최소 1개를 0강제.
            #   → 정확히 3/4 유지(C(4,3)·2^3=32=2^5 → 1.25bpw 무낭비 패킹). TWN threshold 미적용
            #   (nonzero 개수를 4당 3으로 고정해야 5비트 코드공간과 정합).
            b = aw.reshape(O, G, group // 4, 4)
            keep = torch.ones_like(b)
            keep.scatter_(3, b.argmin(dim=3, keepdim=True), 0.0)   # 최소 |w| 위치만 0
            mask = keep.reshape(O, G, group)
        else:
            mask = (aw >= thr_ratio * aw.mean(dim=2, keepdim=True)).to(w.dtype)
        cnt = mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        alpha = (aw * mask).sum(dim=2, keepdim=True) / cnt
        ctx.save_for_backward(aw, alpha)
        ctx.meta = (O, I, G, group, ste_clip)
        return (torch.sign(wg) * mask * alpha).reshape(O, I)

    @staticmethod
    def backward(ctx, g):
        aw, alpha = ctx.saved_tensors
        O, I, G, group, clip = ctx.meta
        win = 1.0 / (1.0 + (aw / (clip * alpha).clamp_min(1e-8)).pow(4))
        return (g.reshape(O, G, group) * win).reshape(O, I), None, None, None, None


def ternary(w, cfg):
    return _TernarySTE.apply(w, cfg.micro_group, cfg.twn_thr_ratio, cfg.ste_clip,
                             getattr(cfg, "sparse34", False))


from .ternary_kernel import ternary_kernel_linear  # noqa: E402  (커스텀 커널 진입점, 기본 off)


class TLinear(nn.Module):
    """g128 삼진 선형층. depth-delta 없음. 모드 delta는 선택."""

    def __init__(self, cfg, in_f, out_f, out_scale=1.0, mode_delta=False):
        super().__init__()
        assert in_f % cfg.micro_group == 0, f"{in_f} % {cfg.micro_group} != 0"
        self.cfg, self.in_f, self.out_f = cfg, in_f, out_f
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * (out_scale / math.sqrt(in_f)))
        self.use_mode = mode_delta and cfg.n_modes > 1 and cfg.mode_rank > 0
        if self.use_mode:
            r = cfg.mode_rank
            self.mode_a = nn.Parameter(torch.randn(r, in_f) / math.sqrt(in_f))
            self.mode_b = nn.Parameter(torch.zeros(out_f, r))
            self.mode_gain = nn.Parameter(torch.ones(cfg.n_modes, r) + 0.02*torch.randn(cfg.n_modes, r))
        self._wq = None
        self._anneal_t = None            # refresh_quant가 채우는 어닐 스칼라 텐서(커널 경로 재컴파일 방지용)
        self._latent_shape = None        # drop_latent() 후에도 파라미터 수를 알기 위한 기록
        self._i8 = None                  # P034 단계3: 삼진 코드(int8)
        self._alpha = None               # P034 단계3: 그룹 스케일(fp32)
        # ★P034 단계3C(2026-08-06): 언팩 결과 캐시. 같은 forward 안에서 **같은 객체가 g 번**
        #   불리므로(타잉) 한 번만 되돌리고 재사용한다. `_unpack_gen` 이 None 이면 기능 off =
        #   종전 경로와 **비트 동일**. 세대(gen)가 바뀌면 버린다 → 버퍼는 항상 1개만 산다.
        self._i8_cache = None
        self._i8_cache_gen = None
        self._unpack_gen = None          # transformer 가 forward 마다 갱신하는 세대 카운터(정수)

    # ---------- P034 단계2: 추론 시 latent 해제 ----------
    def latent_dropped(self):
        return self._latent_shape is not None

    def weight_numel(self):
        """★`weight.numel()` 대신 이걸 쓴다. latent 를 해제해도 **파라미터 수 회계가 유지**된다.

        해제 후 `weight` 는 빈 텐서가 되므로 `mem_breakdown()` 이 삼진 파라미터를 0 으로 세고
        저장 MB 가 갑자기 0 이 된다 — 그러면 P034 단계2 의 효과를 잴 수가 없다.
        """
        if self._latent_shape is not None:
            n = 1
            for d in self._latent_shape:
                n *= d
            return n
        return self.weight.numel()

    def to_int8(self):
        """★P034 단계3 — dequant 사본을 **fp32 → int8 코드 + fp32 α** 로 바꾼다.

        삼진 값은 {−α, 0, +α} 세 개뿐이므로 **코드는 int8 하나로 충분**하다. 지금은 그걸
        fp32(4바이트)로 들고 있어 파라미터당 32비트를 쓴다 — **이론값 1.375~1.71bpw 의 20배 이상**이다.
        여기서 8비트로 내리면 삼진 항이 **정확히 1/4** 이 된다.

        ★대가: forward 마다 `code * alpha` 로 **되돌려야 한다**(fp32 버퍼 1개). 즉 이건
        **메모리를 연산과 바꾸는 것**이다. 결과 014 §10·016 §9 가 우리는 **연산 바운드**라고
        했으므로 **속도는 나빠질 가능성이 높다** — 그래서 배치가 속도도 함께 잰다.
        기대를 미리 적어 두는 이유는, 느려졌을 때 그것을 실패가 아니라 **예측된 트레이드오프**로
        읽기 위해서다.

        ★단계4 와의 관계: 단계4(5비트/1.375bpw 패킹)는 여기서 int8 을 **더 쪼개는** 것이고,
        `code*alpha` 를 fp32 로 되돌리지 않고 **커널이 직접 읽어야** 비로소 이론값에 닿는다.
        단계3 만으로는 이론값의 7.6배에서 멈춘다(계획 P034 §단계3·4 표).
        """
        if self._wq is None:
            raise RuntimeError("to_int8() 전에 freeze_quant() 가 필요하다.")
        if self._i8 is not None:
            return
        import torch
        w = self._wq.detach()
        O, I = w.shape
        g = self.cfg.micro_group
        wg = w.reshape(O, I // g, g)
        alpha = wg.abs().amax(dim=2, keepdim=True)              # 그룹 스케일
        code = torch.zeros_like(wg, dtype=torch.int8)
        nz = alpha > 0
        # 코드는 {-1,0,+1}. alpha 가 0 인 그룹(전부 0)은 코드도 0 이다.
        ratio = torch.where(nz, wg / alpha.clamp_min(1e-12), torch.zeros_like(wg))
        code = torch.where(ratio > 0.5, 1, torch.where(ratio < -0.5, -1, 0)).to(torch.int8)
        self._i8 = code.reshape(O, I)
        self._alpha = alpha.squeeze(-1).contiguous()            # (O, I//g) fp32
        self._wq = None                                         # ★fp32 사본 해제 = 이 단계의 전부

    def _wq_from_i8(self):
        """int8 코드 + α 를 fp32 로 되돌린다.

        ★P034 단계3C — `_unpack_gen` 이 설정돼 있으면 **같은 세대 안에서 결과를 재사용**한다.
        타잉 모델은 `Layer` 들이 **같은 MLP 객체**를 참조하므로(`transformer.py` 의
        `mid_mlps[j // mlp_group]`), 이 함수가 **forward 호출당**이 아니라 **유니크 모듈당**
        한 번만 돌면 된다. g8 이면 중간 16층에서 언팩이 16회 → **2회**가 된다.

        메모리 무손상 근거: 층 스케줄이 `mid_mlps[j//g]` 라 **같은 MLP 가 연속 g 층**에
        나타난다. 세대가 바뀌면(다음 토큰) 캐시를 버리므로 **동시에 사는 fp32 버퍼는
        타잉 그룹 하나분**이고, 이는 캐시가 없을 때의 임시버퍼와 같은 크기다.

        비트 동일성: 재사용하는 값은 **직전에 계산한 그 텐서**이므로 수치가 바뀔 여지가 없다
        (로짓 동등성 게이트가 `0.000e+00` 이어야 하고, 아니면 구현이 틀린 것이다).
        """
        gen = self._unpack_gen
        if gen is not None and self._i8_cache is not None and self._i8_cache_gen == gen:
            return self._i8_cache
        O, I = self._i8.shape
        g = self.cfg.micro_group
        wq = (self._i8.reshape(O, I // g, g).to(self._alpha.dtype)
              * self._alpha.unsqueeze(-1)).reshape(O, I)
        if gen is not None:
            self._i8_cache, self._i8_cache_gen = wq, gen
        return wq

    def set_unpack_gen(self, gen):
        """언팩 캐시 세대 설정. `None` 이면 기능 off(종전 경로)."""
        self._unpack_gen = gen
        if gen is None:
            self._i8_cache = self._i8_cache_gen = None

    def clear_unpack_cache(self):
        """★버퍼만 즉시 해제한다(기능은 켜진 채로).

        결과 016 §13 의 사고: 1차 구현은 캐시를 **다음 forward 까지 들고 있었다.**
        그래서 dense 는 적중이 0 인데 **20층분 fp32 버퍼(472.5MB)가 전부 상주**했고
        `p6d` 가 13.85 → 8.26 tok/s 로 **40% 느려졌다**(대조군이 잡았다).
        지금은 `transformer.forward()` 가 **그룹을 벗어나는 순간** 이걸 부른다 →
        동시에 사는 버퍼는 **MLP 한 벌**뿐이다.
        """
        self._i8_cache = self._i8_cache_gen = None

    def unpack_cache_bytes(self):
        """캐시가 지금 쥐고 있는 바이트. **계측 도구가 이걸 세야 한다**(결과 016 §13)."""
        c = self._i8_cache
        return 0 if c is None else c.numel() * c.element_size()

    def drop_latent(self):
        """freeze 후 fp32 latent `weight` 의 저장공간을 해제한다(추론 전용, 되돌릴 수 없다).

        ★근거(결과 016): 상주 = 유니크삼진 × 4B × **2벌**(latent + dequant 사본) + 나머지.
          추론에는 backward 가 없으므로 latent 는 **죽은 무게**다. 이 한 줄이 상주의 절반을
          없앤다 — 커스텀 커널 없이 오늘 가능한 **가장 큰 단일 레버**다(P034 §단계2).

        ★`_wq` 를 detach 하는 이유: `refresh_quant()` 는 autograd Function 을 타므로 `_wq` 가
          grad_fn 으로 `weight` 를 **붙잡고 있다**. detach 하지 않으면 저장공간을 비워도
          그래프가 참조를 유지해 실제 해제가 일어나지 않는다.
        """
        if self._wq is None:
            raise RuntimeError("drop_latent() 전에 freeze_quant() 가 필요하다 "
                               "(_wq 가 없으면 forward 가 latent 를 다시 읽는다).")
        if self._latent_shape is None:
            self._latent_shape = tuple(self.weight.shape)
        self._wq = self._wq.detach()
        # 파라미터 객체는 유지하고 저장공간만 비운다(state_dict 키 구조를 깨지 않기 위해).
        self.weight.data = self.weight.data.new_empty(0)
        self.weight.requires_grad_(False)

    def _w(self):
        """(옵션) g128 그룹별 mean-centering 후 latent weight 반환."""
        w = self.weight
        if getattr(self.cfg, "center_weights", False):
            O, I = w.shape; g = self.cfg.micro_group
            wg = w.reshape(O, I // g, g)
            w = (wg - wg.mean(dim=2, keepdim=True)).reshape(O, I)
        return w

    def refresh_quant(self, anneal, arena=None):
        # torch.compile 안전: Python 분기 없이 항상 STE 항을 더한다(anneal은 스칼라 텐서).
        self._anneal_t = anneal          # 커널 경로가 float(cfg.quant_anneal) 대신 이 텐서를 쓰게 함
        w = self._w()
        wq = ternary(w, self.cfg)
        self._wq = wq + (1.0 - anneal) * (w - wq).detach()
        # ★P036 Arenas — `Y = X·Tα + λ_t·X·W` (논문 식 7). **detach 하지 않는다**:
        #   존재 이유가 정확히 `∂L/∂X` 에 latent W 를 넣는 것이다(식 8). 위의 어닐 항은
        #   `.detach()` 라 그 경로가 없다 — 두 항은 **다른 일을 한다**(결과 016 §8.6).
        #   arena 가 None 이면 기존 경로 그대로 = 비트 동일.
        if arena is not None:
            self._wq = self._wq + arena * w

    def clear_quant(self):
        self._wq = None
        self._i8 = self._alpha = None        # int8 저장도 해제(학습 재개 시 필수)
        self._i8_cache = self._i8_cache_gen = None   # P034 단계3C 캐시도 함께

    def forward(self, x, mode_p=None):
        if getattr(self.cfg, "use_ternary_kernel", False):   # 분리된 커스텀 커널 경로(기본 off)
            if self._latent_shape is not None:
                raise RuntimeError("latent 해제 상태에서는 커스텀 커널 경로를 쓸 수 없다 "
                                   "(커널이 latent weight 를 직접 읽는다).")
            y = ternary_kernel_linear(x, self._w(), self.cfg, self._anneal_t)
        else:
            if self._i8 is not None:                 # P034 단계3: int8 저장 → 매번 되돌린다
                wq = self._wq_from_i8()
            else:
                wq = self._wq if self._wq is not None else ternary(self._w(), self.cfg)
            y = F.linear(x, wq)
        if self.use_mode and mode_p is not None:
            h = F.linear(x, self.mode_a) * (mode_p @ self.mode_gain)
            y = y + F.linear(h, self.mode_b)
        return y


def ternary_g(w, group, cfg):
    """명시적 group 으로 삼진화(LoRA용). group 은 w 의 마지막 차원을 나눠야 한다.
    LoRA 보정은 3:4 대상 아님 → sparse34=False 고정(backward 인자 수 정합 위해 명시)."""
    return _TernarySTE.apply(w, group, cfg.twn_thr_ratio, cfg.ste_clip, False)


class LoRA(nn.Module):
    """공유 MLP projection 위에 얹는 '층별' 저랭크 보정(RRT식).
    U 를 0으로 초기화해 시작 시 no-op → 안전. bits=2면 D·U 도 삼진(거의 공짜)."""

    def __init__(self, cfg, in_f, out_f, r):
        super().__init__()
        self.cfg, self.in_f, self.r = cfg, in_f, r
        self.D = nn.Parameter(torch.randn(r, in_f) / math.sqrt(in_f))
        self.U = nn.Parameter(torch.zeros(out_f, r))
        self.bits = cfg.mlp_lora_bits
        # ★P008 — LoRA 출력 스케일 s(t). **버퍼**로 두는 이유는 `set_anneal` 과 같다:
        #   파이썬 float 로 두면 값이 바뀔 때마다 torch.compile 이 재컴파일한다.
        #   기본 1.0 = 종전 동작(고정 LoRA). `--lora-decay` 를 안 주면 끝까지 1.0 이다.
        self.register_buffer("_scale", torch.tensor(1.0), persistent=False)

    def forward(self, x):
        D, U = self.D, self.U
        if self.bits == 2:
            gD = self.cfg.micro_group if self.in_f % self.cfg.micro_group == 0 else self.in_f
            D = ternary_g(D, gD, self.cfg)
            U = ternary_g(U, self.r, self.cfg)      # r개를 한 그룹으로(출력 행별 스케일)
        # ★s(t)=1 이면 곱셈 하나만 늘고 값은 종전과 동일하다(비트 동일 아님 — 부동소수점
        #   곱 1.0 은 값을 안 바꾸지만 연산이 추가된다. 대조 실험은 --lora-rank 0 로 한다).
        return F.linear(F.linear(x, D), U) * self._scale
