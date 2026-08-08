"""TiedMLPTransformer (v5) — 조립.

v4 대비 변경(학습 발산 대응, 메모리·파라미터·FLOPs 중립):
  A. QK-norm (modules.Attention).  B. torch.compile 안전 어닐(_anneal 버퍼 + set_anneal).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..config import TMTConfig, dense_baseline  # noqa: F401  (재export)
from .ternary import TLinear, ternary  # noqa: F401
from .modules import RMSNorm, Attention, MLP, Layer, build_rope, apply_rope  # noqa: F401
from .ternary import LoRA  # noqa: F401


class TiedMLPTransformer(nn.Module):
    def __init__(self, cfg: TMTConfig):
        super().__init__()
        self.cfg = cfg
        E = cfg.emb_rank if cfg.emb_rank else cfg.dim

        self.emb = nn.Embedding(cfg.vocab_size, E)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.emb_up = nn.Linear(E, cfg.dim, bias=False) if cfg.emb_rank else None
        if cfg.emb_rank:
            nn.init.normal_(self.emb_up.weight, std=0.02)

        self.pre_mlps = nn.ModuleList([MLP(cfg) for _ in range(cfg.n_prelude)])
        self.mid_mlps = nn.ModuleList([MLP(cfg) for _ in range(cfg.n_mlp_groups)])
        self.coda_mlps = nn.ModuleList([MLP(cfg) for _ in range(cfg.n_coda)])

        layers, self.owner = [], []
        for i in range(cfg.n_layers):
            owns = (i % cfg.cla_group == 0)
            self.owner.append(i - (i % cfg.cla_group))
            if i < cfg.n_prelude:
                mlp = self.pre_mlps[i]
            elif i < cfg.n_prelude + cfg.n_middle:
                j = i - cfg.n_prelude
                mlp = self.mid_mlps[j // cfg.mlp_group if cfg.tie_mlp else j]
            else:
                mlp = self.coda_mlps[i - cfg.n_prelude - cfg.n_middle]
            is_mid_tied = (cfg.n_prelude <= i < cfg.n_prelude + cfg.n_middle) and cfg.tie_mlp
            layers.append(Layer(cfg, owns, mlp, mlp_lora=is_mid_tied, mlp_film=is_mid_tied))
        self.layers = nn.ModuleList(layers)

        self.norm_f = RMSNorm(cfg.dim, cfg.norm_eps)
        self.norm_f_scale = nn.Parameter(torch.ones(cfg.dim))

        if cfg.n_modes > 1:
            self.router = nn.Linear(cfg.dim, cfg.n_modes, bias=False)
            nn.init.normal_(self.router.weight, std=0.02)
            self.router_bias = nn.Parameter(torch.zeros(cfg.n_layers, cfg.n_modes))

        cos, sin = build_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # v5: 삼진 어닐 계수를 버퍼로 → torch.compile 재컴파일 방지.
        self.register_buffer("_anneal", torch.tensor(float(cfg.quant_anneal)), persistent=False)
        self._tlinear_cache = list(self._tlinears())   # ②: 매 forward 모듈 트리 순회 제거(plain list)
        self._quant_frozen = False                     # freeze_quant() 참조(추론 전용 최적화)
        self._int8_store = False                       # P034 단계3
        self._unpack_cache = False                     # P034 단계3C (기본 off = 종전 경로)
        self._unpack_gen = 0
        self._cacheable_mlps = []
        self._armed_tlinears = []
        self._arena = None                             # P036 Arenas λ_t (None = 항 없음)
        self._arena_v = 0.0

    # ---------- quantization ----------
    def _tlinears(self):
        for m in self.modules():
            if isinstance(m, TLinear):
                yield m

    def set_anneal(self, v: float):
        self._anneal.fill_(float(v))
        self.cfg.quant_anneal = float(v)

    def set_lora_scale(self, v: float):
        """★P008 — 층별 LoRA 의 출력 스케일 s(t) 를 설정한다.

        **s=1 이면 고정 LoRA(종전), s=0 이면 LoRA 가 완전히 사라진다.**
        학습 중 1 → 0 으로 어닐하면 **초반에는 층별 특화(≈untied), 종료 시 순수 타잉**이 된다.
        그리고 s=0 이므로 **배포 메모리 대가가 0** 이다 — 이것이 고정 LoRA(1.82×→1.71×)와
        결정적으로 다른 점이다.

        ★삼진 어닐(`set_anneal`)과 **같은 원리·같은 구현 패턴**이다:
        제약을 처음부터 걸지 않고 학습이 진행되며 점진적으로 조인다.

        적용 대상은 **모든 `LoRA` 모듈 하나의 집합**으로만 정한다 — 결과 016 §14 의 교훈
        ("적용 대상 집합을 두 곳에서 따로 정하면 한쪽만 고쳐도 통과한다").
        """
        for m in self.modules():
            if isinstance(m, LoRA):
                m._scale.fill_(float(v))

    def set_arena(self, v: float):
        """P036 Arenas 의 λ_t 를 설정한다. **0 이면 항이 완전히 사라진다**(배포 상태)."""
        self._arena_v = float(v)
        if self._arena is None:
            self._arena = torch.tensor(float(v), device=self.emb.weight.device)
        else:
            self._arena.fill_(float(v))

    def refresh_quant(self):
        # ★arena 는 λ_t ^> 0 일 때만 전달한다 → 기본 학습·추론 경로는 종전과 **비트 동일**.
        ar = self._arena if (self._arena is not None and self._arena_v > 0.0) else None
        for m in self._tlinear_cache:
            m.refresh_quant(self._anneal, ar)

    def clear_quant(self):
        self._quant_frozen = False
        for m in self._tlinear_cache:
            m.clear_quant()

    def latent_dropped(self):
        """P034 단계2 가 적용됐는지. `mem_breakdown()` 의 상주 계산이 이 값을 본다."""
        return any(m.latent_dropped() for m in self._tlinear_cache)

    def drop_latent(self):
        """★P034 단계2 — 추론 시 fp32 latent 가중치를 해제한다(되돌릴 수 없다).

        `freeze_quant()` 이후에만 유효하다. backward 가 없는 추론에서 latent 는 죽은 무게이고,
        결과 016 의 실측식(상주 = 유니크삼진 × 4B × **2벌** + 나머지)에서 그 2벌 중 한 벌이다.
        커스텀 커널 없이 **상주를 약 절반**으로 줄이는 오늘 가능한 최대 레버다.

        ⚠️ 학습 재개 불가. `clear_quant()` 로도 되돌아오지 않는다 — 다시 쓰려면 재로드한다.
        """
        if not self._quant_frozen:
            raise RuntimeError("drop_latent() 전에 freeze_quant() 가 필요하다.")
        for m in self._tlinear_cache:
            m.drop_latent()

    def to_int8(self):
        """★P034 단계3 — 삼진 dequant 사본을 int8 코드 + fp32 α 로 저장한다.

        `drop_latent()` 와 함께 쓰는 것이 정상 순서다(latent 해제 → int8 저장).
        되돌릴 수 없다. 추론 전용.
        """
        if not self._quant_frozen:
            raise RuntimeError("to_int8() 전에 freeze_quant() 가 필요하다.")
        for m in self._tlinear_cache:
            m.to_int8()
        self._int8_store = True

    def int8_stored(self):
        return getattr(self, "_int8_store", False)

    # ---------- P034 단계5 : 임베딩 (★설계 미완 — 구현하지 않는다) ----------
    #
    # ★★2026-08-06 발견 — **임베딩은 지금까지 한 번도 양자화된 적이 없다.**
    #   `cfg.quantize_embedding=True` 는 `mem_breakdown()` 의 **저장(packed) 계산에만** 쓰이고
    #   (`e_bits = b_e if cfg.quantize_embedding else 16`), `self.emb` 는 평범한
    #   `nn.Embedding`(fp32) 이다. 즉 `packed_mb` 27.1MB 는 삼진뿐 아니라 **임베딩 쪽에서도
    #   아직 없는 포맷을 가정**하고 있다. 상주 33.0MB 중 32.8MB 가 이것이다(mC 상주의 38%).
    #
    # ★그런데 단순 int8 화가 **성립하지 않는다** — 임베딩이 **출력 헤드와 묶여 있기 때문**이다:
    #       logits = F.linear(F.linear(x, emb_up.weight.t()), self.emb.weight)
    #   조회(lookup)는 행 몇 개만 되돌리면 되지만, **헤드는 (32768, 256) 전체를 GEMM 에 넣는다.**
    #   int8 로 저장하고 헤드에서 fp32 로 되돌리면 **forward 마다 32.8MB 를 쓰고 읽는다** —
    #   결과 016 §13 에서 실측한 대가(0.1035 ms/tok per MB)로 환산하면 **+3.4ms/토큰(약 +5%)** 이고,
    #   무엇보다 **§13 이 방금 저지른 것과 같은 종류의 트레이드**다. 설계를 마치기 전에는 넣지 않는다.
    #
    # 선택지(계획 P034 §단계5 에 기록):
    #   (a) 헤드를 임베딩에서 분리(untie) — 파라미터 +8.4M. **메모리 목표와 정면 충돌**
    #   (b) 헤드를 int8 GEMM 으로(`torch._int_mm`) — 활성값도 int8 여야 한다. 연구 과제
    #   (c) 어휘 축소 — P039(어휘 활용도 진단)가 **미학습 토큰을 찾으면** 32,768 을 줄일 수 있다.
    #       ★같은 33MB 를 **양자화 없이** 줄이는 유일한 경로이고, **선결 실험이 이미 설계돼 있다**
    #   (d) 임베딩 랭크 축소(E=256 → 128) — 재학습 필요. 품질 영향 미지
    #
    # → **(c) 가 가장 값싸고 위험이 낮다. P039 를 먼저 돌린다.**

    def enable_unpack_cache(self, on=True):
        """★P034 단계3C — int8 언팩 결과를 **유니크 모듈당 1회**로 줄인다(타잉 전용 이득).

        근거: `docs/20260806_레이어타이잉-메모리와속도-원인분석.md` §5.
        `forward()` 는 층마다 `TLinear.forward()` 를 부르고, 타이된 층들은 **같은 객체**를
        공유한다. 그런데 `_wq_from_i8()` 은 호출마다 fp32 로 되돌리므로 g8 이면 **같은 언팩을
        16번** 한다. 세대 카운터로 한 번만 돌게 만든다.

        ★★2차 구현(2026-08-06, 결과 016 §13 의 사고 대응). 1차는 두 가지가 틀렸다:

          ① **공유되지 않는 모듈까지 캐시했다.** dense 는 모든 TLinear 가 층당 1회라 적중이
             0 인데 버퍼만 20층분(472.5MB) 남았다 → `p6d` 가 **40% 느려졌다.**
             지금은 **층이 2회 이상 참조하는 MLP 의 TLinear 에만** 캐시를 켠다.
             **dense 는 켤 대상이 하나도 없어 종전 경로와 완전히 동일하다.**
          ② **버퍼를 다음 forward 까지 들고 있었다.** 지금은 `forward()` 가 **그룹을 벗어나는
             순간** 해제한다(층 스케줄이 `mid_mlps[j//g]` 라 같은 MLP 가 연속으로 온다)
             → 동시에 사는 버퍼는 **MLP 한 벌(약 18.9MB)** 이 상한이다.

        ⚠️ **`to_int8()` 을 쓴 추론 경로 전용**이다. 학습에는 의미가 없다(latent 가 매 스텝 바뀐다).
        ⚠️ 기본 off. 켜도 **결과는 비트 동일**해야 한다 — 다르면 구현이 틀린 것이고,
           `scripts/mem_runtime.py` 의 로짓 동등성 게이트가 `0.000e+00` 로 그것을 확인한다.
           **그리고 그 도구는 캐시 바이트도 함께 센다** — §13 은 안 세서 못 잡았다.
        """
        from collections import Counter
        self._unpack_cache = bool(on)
        for m in self._tlinear_cache:                 # 항상 전부 끄고 시작(재호출 안전)
            m.set_unpack_gen(None)
        self._cacheable_mlps = []
        self._armed_tlinears = []                     # ★3차 수정(§14): 무장된 것만 따로 든다
        if not on:
            return
        # ★공유 판정 = "몇 개의 층이 이 MLP 객체를 참조하는가". 1이면 캐시할 이유가 없다.
        cnt = Counter(id(l.mlp[0]) for l in self.layers)
        seen = set()
        for l in self.layers:
            k = id(l.mlp[0])
            if cnt[k] > 1 and k not in seen:
                seen.add(k)
                self._cacheable_mlps.append(l.mlp[0])
        for mlp in self._cacheable_mlps:
            for sub in mlp.modules():
                if isinstance(sub, TLinear):
                    sub.set_unpack_gen(0)
                    self._armed_tlinears.append(sub)

    @staticmethod
    def _clear_mlp_unpack(mlp):
        for sub in mlp.modules():
            if isinstance(sub, TLinear):
                sub.clear_unpack_cache()

    def unpack_cache_mb(self):
        """언팩 캐시가 지금 쥐고 있는 MB. **상주 회계에 반드시 포함**한다(결과 016 §13)."""
        return sum(m.unpack_cache_bytes() for m in self._tlinear_cache) / 1024 ** 2

    def unpack_cached(self):
        return getattr(self, "_unpack_cache", False)

    def freeze_quant(self):
        """★추론용: 삼진 가중치를 **한 번만** 계산하고 이후 forward 에서 재계산하지 않는다.

        `forward()` 는 매 호출마다 `refresh_quant()` 를 부른다(학습 중에는 latent weight 가
        스텝마다 바뀌므로 **반드시 그래야 한다**). 그런데 추론에서는 가중치가 고정이라
        그 재계산이 전부 낭비다. 결과 014 에서 이 비용이 **CPU 추론 시간의 약 79%** 를
        차지했고, sparse34 는 `argmin`+`scatter_` 경로라 더 비쌌다(mA 가 dense 보다 느려 보인 원인).

        되돌리려면 `clear_quant()` 를 부른다(학습 재개 시 필수). 그래서 `clear_quant` 가
        플래그를 함께 내린다 — 안 그러면 얼어붙은 가중치로 학습하게 된다.
        """
        self.refresh_quant()
        self._quant_frozen = True

    # ---------- P031 단계0 : 층 방문 스케줄 ----------
    def visit_schedule(self):
        """forward 가 층을 방문할 순서. **`infer_repeat == 1.0` 이면 `range(n_layers)` 와 동일**하다.

        ★왜 이렇게 정의하나(계획 P031): 공유되는 것은 **MLP 가중치뿐**이고 어텐션·norm 은 층마다
        따로 있다. 그래서 "추론 시 g 를 늘린다"는 성립하지 않는다 — 성립하는 유일한 정의는
        **middle 블록 전체를 R 회 통과**시키는 것이다. prelude·coda 는 건드리지 않는다.

        `total = round(n_middle * R)` 이 middle 통과 횟수다.
          R ^< 1 : middle 에서 `total` 개만 고른다(축소 — 메모리 그대로, 지연만 감소)
          R ^> 1 : 전체를 돈 뒤 `total - n_middle` 개를 더 돈다(확장)
        `repeat_where` 가 **어디를** 더/덜 돌지 정한다. 이 선택이 결과를 바꾸므로
        한 배치만 보고 일반화하지 않는다.
        """
        cfg = self.cfg
        n, p, m = cfg.n_layers, cfg.n_prelude, cfg.n_middle
        R = float(getattr(cfg, "infer_repeat", 1.0) or 1.0)
        if R == 1.0:
            return list(range(n))                      # ★기본 경로는 종전과 완전히 같다
        where = getattr(cfg, "repeat_where", "front")
        mid = list(range(p, p + m))
        total = int(round(m * R))
        total = max(1, total)
        if total <= m:                                  # 축소
            k = total
            if where == "front":
                sel = mid[:k]
            elif where == "back":
                sel = mid[-k:]
            else:                                       # even — 균등 간격으로 남긴다
                sel = [mid[round(i * (m - 1) / max(k - 1, 1))] for i in range(k)] if k > 1 else [mid[m // 2]]
            body = sel
        else:                                           # 확장
            extra = total - m
            if where == "front":
                body = mid + mid[:extra] if extra <= m else mid * (total // m) + mid[:total % m]
            elif where == "back":
                body = mid + mid[-extra:] if extra <= m else mid * (total // m) + mid[:total % m]
            else:                                       # even — 균등 간격 층을 제자리에서 한 번 더
                dup = {mid[round(i * (m - 1) / max(extra - 1, 1))] for i in range(extra)} \
                    if extra > 1 else {mid[m // 2]}
                body = [j for jj in mid for j in ((jj, jj) if jj in dup else (jj,))]
        return list(range(p)) + body + list(range(p + m, n))

    # ---------- forward ----------
    def forward(self, tokens, mode_override=None, return_aux=False,
                past_kv=None, use_cache=False):
        """`past_kv` 는 **owner 층 인덱스 → (k, v)** 딕셔너리다.

        ★CLA 주의: KV 를 공유하는 층들은 **캐시도 공유**한다. 그래서 캐시 키는 층 인덱스가
        아니라 `self.owner[i]`(= 그 그룹의 소유 층)이다. 층마다 캐시를 두면 같은 KV 를
        cla_group 배로 중복 저장하게 된다.
        """
        cfg = self.cfg
        B, T = tokens.shape
        # 캐시가 있으면 이번 forward 의 토큰은 past 뒤에 붙는다 → 절대위치가 past_len 만큼 밀린다.
        past_len = 0
        if past_kv:
            past_len = next(iter(past_kv.values()))[0].shape[2]   # (B, n_kv_heads, T, head_dim)
        assert past_len + T <= cfg.max_seq_len, \
            f"past {past_len} + 입력 {T} > max_seq_len {cfg.max_seq_len}"
        x = self.emb(tokens)
        if self.emb_up is not None:
            x = self.emb_up(x)
        # ★RoPE 는 절대위치다. 캐시 사용 시 [:T] 가 아니라 [past_len : past_len+T] 를 써야 한다.
        cos, sin = self.rope_cos[past_len:past_len + T], self.rope_sin[past_len:past_len + T]
        if not self._quant_frozen:
            self.refresh_quant()

        # ★P034 단계3C: 언팩 캐시 세대 갱신. 기본 off 면 이 블록이 통째로 건너뛰어진다.
        #   ★★3차 수정(결과 016 §14) — 종전에는 `self._tlinear_cache`(전부)를 돌며 세대를
        #   심어서 `enable_unpack_cache` 가 꺼 둔 **어텐션 모듈까지 되살렸다.**
        #   어텐션은 층당 1회 호출이라 적중이 0인데 버퍼만 남는다(dense 112.5MB / 타잉 101.3MB).
        #   지금은 **무장된 목록만** 돈다.
        if getattr(self, "_unpack_cache", False):
            self._unpack_gen += 1
            for _m in getattr(self, "_armed_tlinears", ()):
                _m._unpack_gen = self._unpack_gen

        # ★P031: 기본(infer_repeat=1.0)이면 schedule == range(n_layers) 라 종전과 동일하다.
        schedule = self.visit_schedule()
        repeating = len(schedule) != cfg.n_layers
        if repeating and self.training:
            raise RuntimeError("infer_repeat 는 **추론 전용**이다. 학습 경로에서 켜지 않는다 "
                               "(층 반복은 학습된 깊이 분포 밖이고, grad checkpoint 와도 섞인다).")
        seen = {}                                       # owner -> 그 owner 를 몇 번째 통과 중인가

        kv_bank, mode_hist = {}, []
        for step_idx, i in enumerate(schedule):
            layer = self.layers[i]
            mode_p = None
            if cfg.n_modes > 1:
                if mode_override is not None:
                    mp = mode_override[i] if mode_override.dim() == 2 else mode_override
                    mode_p = mp.to(x.dtype).view(1, 1, -1).expand(B, T, -1)
                else:
                    mode_p = F.softmax(self.router(x) + self.router_bias[i], dim=-1)
                mode_hist.append(mode_p)

            # ★캐시 키 — R=1.0 이면 종전처럼 **정수 owner 인덱스**다(비트 동일성 보존).
            #   반복 중이면 같은 owner 를 여러 번 지나므로 (owner, 통과번호) 로 분리한다.
            #   그렇게 안 하면 두 번째 통과가 첫 통과의 KV 를 덮어써서 캐시가 조용히 틀려진다.
            own = self.owner[i]
            if i == own:
                pas = seen.get(own, 0)
                seen[own] = pas + 1
            else:
                pas = seen.get(own, 1) - 1
            key = own if not repeating else (own, pas)

            if i == own:
                reuse = repeating and cfg.repeat_kv_reuse and pas > 0
                if reuse:
                    kv_bank[key] = kv_bank[(own, 0)]     # 첫 통과 KV 재사용(대조 조건)
                else:
                    k_new, v_new = layer.attn.compute_kv(x, cos, sin)
                    if past_kv and key in past_kv:
                        k_old, v_old = past_kv[key]
                        k_new = torch.cat([k_old, k_new], dim=2)
                        v_new = torch.cat([v_old, v_new], dim=2)
                    kv_bank[key] = (k_new, v_new)
            elif key not in kv_bank:
                # ★축소(R^<1)에서 CLA 그룹 경계가 잘리면 owner 가 스케줄에 없을 수 있다.
                #   그때는 이 층이 스스로 KV 를 계산한다(조용한 KeyError 대신 명시적 폴백).
                k_new, v_new = layer.attn.compute_kv(x, cos, sin)
                if past_kv and key in past_kv:
                    k_old, v_old = past_kv[key]
                    k_new = torch.cat([k_old, k_new], dim=2)
                    v_new = torch.cat([v_old, v_new], dim=2)
                kv_bank[key] = (k_new, v_new)
            kv = kv_bank[key]

            if cfg.grad_checkpoint and self.training:
                x = checkpoint(lambda inp, L=layer, k=kv, mp=mode_p:
                               L(inp, k, cos, sin, mp), x, use_reentrant=False)
            else:
                x = layer(x, kv, cos, sin, mode_p)

            # ★P034 단계3C — 이 MLP 를 쓰는 연속 구간이 끝나면 **즉시 버퍼를 버린다.**
            #   1차 구현은 이걸 안 해서 dense 가 20층분 fp32 를 들고 40% 느려졌다(§13).
            if getattr(self, "_unpack_cache", False):
                _nxt = schedule[step_idx + 1] if step_idx + 1 < len(schedule) else None
                if _nxt is None or self.layers[_nxt].mlp[0] is not layer.mlp[0]:
                    self._clear_mlp_unpack(layer.mlp[0])

        x = self.norm_f(x) * self.norm_f_scale
        if self.emb_up is not None:
            logits = F.linear(F.linear(x, self.emb_up.weight.t()), self.emb.weight)
        else:
            logits = F.linear(x, self.emb.weight)

        if use_cache:
            # kv_bank 는 owner 층만 키로 갖는다(§CLA 주의 참조) → 그대로 다음 스텝의 past 가 된다.
            return (logits, kv_bank) if not return_aux else (logits, kv_bank, None)
        if not return_aux:
            return logits
        aux = {"router_loss": self._balance(mode_hist),
               "mode_usage": (torch.stack([p.mean(dim=(0, 1)) for p in mode_hist]).detach()
                              if mode_hist else None)}
        return logits, aux

    def _balance(self, hist):
        if not hist:
            return torch.zeros((), device=self.emb.weight.device)
        loss = 0.0
        for p in hist:
            f = p.mean(dim=(0, 1))
            loss = loss + self.cfg.n_modes * (f * f).sum()
        return loss / len(hist)

    # ---------- optimizer ----------
    def param_groups(self, lr, weight_decay=0.1):
        import math
        tied = {id(p) for m in self.mid_mlps for p in m.parameters()} if self.cfg.tie_mlp else set()
        gt, dense, nodecay = [], [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() < 2 or any(k in n for k in ("scale", "shift", "gates", "gain", "bias")):
                nodecay.append(p)
            elif id(p) in tied:
                gt.append(p)
            else:
                dense.append(p)
        g = self.cfg.mlp_group if self.cfg.tie_mlp else 1
        return [{"params": gt, "lr": lr / math.sqrt(g), "weight_decay": weight_decay},
                {"params": dense, "lr": lr, "weight_decay": weight_decay},
                {"params": nodecay, "lr": lr, "weight_decay": 0.0}]

    # ---------- accounting ----------
    # ★bpw 회계 규약 (2026-07-31 통일 — 결과 016 §7.4·§8.4, P034 §5)
    #   종전 문제: 삼진 1.95 는 "코드 + 그룹스케일 + 컨테이너" 인데 sparse34 1.25 는
    #   **순수 코드공간뿐**이었다. 서로 다른 규약의 두 값으로 비율을 만들어 감축비가
    #   2.66× 로 과대됐다(동일 회계로는 2.11×).
    #
    #   정본 = **안 B(코드 + 그룹 스케일)**. 이유:
    #     ① log2(3) + 16/128 = 1.7100 이 문서의 "g128 삼진 1.71" 과 정확히 일치하고
    #        **설계 시점(가중치가 난수일 때)에 계산 가능한 고정 상수**다.
    #        안 C 로 1.95 를 유지하려면 학습 후에만 아는 실측 엔트로피(1.554)가 필요한데
    #        report() 는 학습 **시작 전**에 호출된다 → 계산 자체가 불가능.
    #     ② 코드와 스케일은 **우리가 반드시 저장해야** 하는 것이다(둘 중 하나만으로는 복원 불가).
    #        컨테이너는 **포맷 선택**의 문제이고 우리는 아직 포맷을 안 정했다.
    #     ③ 논문(Sherry) 기준선 1.67 과 같은 층위 → 외부 비교가 성립한다.
    #   병기 = **안 C(B + 컨테이너)**. 실제 배포 파일 크기 추정용이고 구 문서(30.9MB 계열)와 연속.
    CODE_TERNARY = 1.5849625007211562   # log2(3) — 삼진 심볼 하나의 정보량
    CODE_S34 = 1.25                      # C(4,3)*2^3 = 32 = 2^5, 4가중치당 5비트(무낭비)
    SCALE_BITS = 16.0                    # 그룹 스케일 alpha 를 fp16 로 저장
    CONTAINER_BPW = 0.274                # (안 C 전용) 1.95 - (측정엔트로피 1.554 + 0.125) 역산.
                                         #   GGUF 가정이며 **우리 포맷이 아니다**. 참고값.

    def _bpw(self, container=False):
        """(삼진 bpw, sparse34 bpw, 임베딩 bpw). 규약을 한 곳에서 만든다."""
        scale = self.SCALE_BITS / self.cfg.micro_group
        ovh = self.CONTAINER_BPW if container else 0.0
        b_t = self.CODE_TERNARY + scale + ovh
        b_s = self.CODE_S34 + scale + ovh
        return b_t, b_s, b_t          # 임베딩은 3:4 대상이 아니므로 삼진 규약을 그대로 쓴다

    def mem_breakdown(self, bpw=None, container=False):
        """저장(packed) 메모리의 **정확한** 분해(MB). `report()`·`compare` 의 단일 소스.

        ★ 3:4 준정형(P016)은 **삼진 가중치에만** 적용된다. 임베딩·norm·gate 는 대상이 아니다.
        따라서 "전체 파라미터 × 단일 bpw" 로 근사하면 sparse34 런에서만 과소가 된다.
        이 함수가 그 실수를 원천 차단한다.

        `bpw` 인자는 **구 호출부 호환용**이며 무시된다(규약은 위 상수가 정한다).
        `container=True` 면 안 C(컨테이너 포함) 로 계산한다.
        """
        cfg = self.cfg
        MB = lambda n, b: n * b / 8 / 1024 ** 2
        b_t, b_s, b_e = self._bpw(container)
        bpw_t = b_s if getattr(cfg, "sparse34", False) else b_t
        # ★`weight.numel()` 이 아니라 `weight_numel()` — P034 단계2(latent 해제) 후에도
        #   파라미터 수 회계가 유지된다(해제하면 weight 는 빈 텐서가 된다).
        tern = sum(m.weight_numel() for m in self._tlinears())
        mode = sum(p.numel() for m in self._tlinears() if m.use_mode
                   for p in (m.mode_a, m.mode_b, m.mode_gain))
        emb = self.emb.weight.numel() + (self.emb_up.weight.numel() if self.emb_up is not None else 0)
        lora = sum(p.numel() for m in self.modules() if isinstance(m, LoRA) for p in m.parameters())
        # ★`parameters()` 합에서 빼는 방식은 P034 단계2 후에 **깨진다** — latent 를 해제하면
        #   `parameters()` 합만 줄고 `tern`(=weight_numel)은 그대로라 `other` 가 **음수**가 된다.
        #   그래서 해제분을 되돌려 더한 뒤 뺀다. 해제 전에는 값이 종전과 완전히 같다.
        dropped = sum(m.weight_numel() - m.weight.numel() for m in self._tlinears())
        total_live = sum(p.numel() for p in self.parameters()) + dropped
        other = total_live - tern - mode - emb - lora
        e_bits = b_e if cfg.quantize_embedding else 16
        l_bits = cfg.mlp_lora_bits
        parts = {"ternary": MB(tern, bpw_t), "mode": MB(mode, 16), "emb": MB(emb, e_bits),
                 "other": MB(other, 16), "lora": MB(lora, l_bits)}
        # ★상주(runtime) — 결과 016 실측식: 유니크 삼진 x 4B x 2벌(latent + dequant 사본) + 나머지 fp32
        #   P034 단계2 로 latent 를 해제하면 **1벌**이 된다(= 이 값이 절반 근처로 떨어진다).
        copies = 1 if self.latent_dropped() else 2
        # ★P034 단계3: 삼진 사본이 int8(1바이트) + 그룹 α(fp32, 파라미터당 4/micro_group 바이트)
        if self.int8_stored():
            tern_bytes = tern * (1 + 4 / cfg.micro_group)
            runtime_mb = (tern_bytes + (mode + lora) * 4 + (emb + other) * 4) / 1024 ** 2
        else:
            runtime_mb = ((tern + mode + lora) * 4 * copies + (emb + other) * 4) / 1024 ** 2
        return {"total_mb": sum(parts.values()),          # = packed_mb (구 호출부 호환 별칭)
                "packed_mb": sum(parts.values()),
                "runtime_mb": runtime_mb,
                "runtime_copies": copies,        # 2 = latent + dequant / 1 = P034 단계2 적용
                "int8_stored": self.int8_stored(),   # P034 단계3
                "latent_dropped": copies == 1,
                "parts_mb": parts,
                "params": {"ternary": tern, "mode": mode, "emb": emb, "other": other, "lora": lora,
                           "total": tern + mode + emb + other + lora},
                "bpw_ternary": bpw_t, "bpw_emb": e_bits,
                "bpw_convention": ("C: code + group scale + container(0.274, GGUF 가정)"
                                   if container else
                                   "B: code(log2 3 / 1.25) + group scale(16/micro_group)"),
                "sparse34": bool(getattr(cfg, "sparse34", False))}

    def mem_report_all(self):
        """정본(B) + 병기(C) + 상주 를 한 번에. trainer 가 json 에 이걸 펼쳐 넣는다."""
        b = self.mem_breakdown(container=False)
        c = self.mem_breakdown(container=True)
        return {"packed_mb": b["packed_mb"],
                "packed_mb_container": c["packed_mb"],
                "runtime_mb": b["runtime_mb"],
                "bpw_convention": b["bpw_convention"],
                "bpw_ternary": b["bpw_ternary"],
                "mem_parts_mb": b["parts_mb"],
                "mem_params": b["params"]}

    def report(self, bpw=None, l3_mb=32.0):
        cfg = self.cfg
        MB = lambda n, b: n * b / 8 / 1024 ** 2
        bd = self.mem_breakdown()
        bd_c = self.mem_breakdown(container=True)
        tern, mode = bd["params"]["ternary"], bd["params"]["mode"]
        emb, other, lora = bd["params"]["emb"], bd["params"]["other"], bd["params"]["lora"]
        total, mem = bd["params"]["total"], bd["total_mb"]
        bpw_t, e_bits, l_bits = bd["bpw_ternary"], bd["bpw_emb"], cfg.mlp_lora_bits

        per_l = (cfg.dim*cfg.dim*2 + cfg.kv_dim*cfg.dim*2) + 3*cfg.dim*cfg.ffn_dim
        flops = 2 * cfg.n_layers * per_l / 1e9
        n_kv = sum(1 for i in range(cfg.n_layers) if i == self.owner[i])
        kv_kb = n_kv * 2 * cfg.kv_dim * 2 / 1024
        # ★2026-07-31 수정 — `bpw`(함수 인자)를 그대로 쓰고 있었다. 회계 통일 커밋(7b123fc)이
        #   기본값을 `1.95` → `None` 으로 바꾸면서 **이 줄만 따라오지 않아** 모든 학습이
        #   `report()` 에서 죽었다(TypeError: int * NoneType). 정본은 규약이 정한 `bpw_t` 다.
        #   `bpw` 인자는 구 호출부 호환용으로만 남아 있고 무시되는 것이 맞다.
        mlp_mb = MB(3*cfg.dim*cfg.ffn_dim, bpw_t)

        L = []; A = L.append
        A("=" * 72)
        A(f"d={cfg.dim} ffn={cfg.ffn_dim} layers={cfg.n_prelude}+{cfg.n_middle}+{cfg.n_coda}"
          f"  MLP g={cfg.mlp_group if cfg.tie_mlp else 1}  CLA={cfg.cla_group}"
          f"  vocab={cfg.vocab_size} E={cfg.emb_rank or cfg.dim}")
        A("=" * 72)
        _tlbl = '삼진 가중치(3:4 1.25bpw)' if getattr(cfg, "sparse34", False) else '삼진 가중치'
        A(f"  {_tlbl:<24}{tern/1e6:>9.1f}M{MB(tern,bpw_t):>10.1f} MB")
        if mode: A(f"  {'모드 delta (fp16)':<24}{mode/1e6:>9.1f}M{MB(mode,16):>10.1f} MB")
        if lora: A(f"  {'LoRA(r='+str(cfg.mlp_lora_rank)+', '+str(l_bits)+'bit)':<24}{lora/1e6:>9.1f}M{MB(lora,l_bits):>10.1f} MB")
        A(f"  {'임베딩(factorized)':<24}{emb/1e6:>9.1f}M{MB(emb,e_bits):>10.1f} MB")
        A(f"  {'기타(norm/gate/router)':<24}{other/1e6:>9.1f}M{MB(other,16):>10.1f} MB")
        A(f"  {'-'*52}")
        A(f"  {'합계':<24}{total/1e6:>9.1f}M{mem:>10.1f} MB")
        A("")
        A("")
        A(f"  {'저장(packed, 안 B)':<24}{'':>9} {mem:>10.1f} MB   {bd['bpw_convention']}")
        A(f"  {'저장(참고, 안 C 컨테이너)':<24}{'':>9} {bd_c['packed_mb']:>10.1f} MB")
        A(f"  {'★상주(runtime, 현 구현)':<24}{'':>9} {bd['runtime_mb']:>10.1f} MB"
          f"   = 유니크삼진 x 4B x 2벌 + 나머지 fp32 (결과 016)")
        A(f"  L3({l3_mb:.0f}MB) 상주       : "
          f"{'가능' if bd['runtime_mb'] < l3_mb*0.7 else '불가'}"
          f"   ★**상주 기준**으로 판정한다 — 저장 기준은 {mem:.1f}MB 라 오해를 부른다(결과 016)")
        if cfg.tie_mlp:
            A(f"  타이드 MLP 1그룹      : {mlp_mb:.1f} MB  ({cfg.mlp_group}회 연속 재사용)")
        else:
            A(f"  MLP 블록(층별 독립)    : {mlp_mb:.1f} MB  (dense: 재사용 없음)")
        A(f"  토큰당 FLOPs          : {flops:.3f} GFLOP")
        A(f"  KV 캐시               : {kv_kb:.1f} KB/token  ({n_kv}/{cfg.n_layers} 층만 소유)")
        A(f"    1024 ctx {kv_kb*1024/1024:.1f} MB   2048 ctx {kv_kb*2048/1024:.1f} MB")
        A("=" * 72)
        return "\n".join(L)
