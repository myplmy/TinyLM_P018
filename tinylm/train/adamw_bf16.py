"""P022B 단계2 — **옵티마이저 상태를 저정밀로 드는 AdamW.**

## 왜 있나

DeepSeek-V3 기술보고서(arXiv:2412.19437) §3.3.3 *Low-Precision Optimizer States*:

> *"We adopt the BF16 data format instead of FP32 to track the first and second moments in the
>  AdamW optimizer, **without incurring observable performance degradation**. However, the master
>  weights (stored by the optimizer) and gradients (used for batch size accumulation) are still
>  retained in FP32 to ensure numerical stability."*

우리 학습 파라미터는 약 **63.5M**(삼진 54.85M + 임베딩 8.59M + 기타)이고
AdamW 는 모멘트를 **2벌 × 4B** 로 든다 = **약 508MB**. bf16 이면 **약 254MB 를 돌려받는다.**
이 저장소의 1순위 목표가 *"연산이 아니라 메모리"* 이므로 FP8 GEMM(속도)보다 **이쪽이 결이 맞는다.**

## 무엇을 저정밀로 하고 무엇을 안 하는가 — 논문 그대로

| | 정밀도 | 이유 |
|---|---|---|
| `exp_avg`, `exp_avg_sq` (상태) | **`state_dtype`**(bf16) | 이 클래스가 바꾸는 유일한 것 |
| master weight = `p` | **fp32** | 논문이 명시적으로 유지. 우리 `TLinear.weight` latent 가 그것이다 |
| gradient = `p.grad` | **fp32** | 〃 |
| **산술 자체** | **항상 fp32** | 상태를 fp32 로 올려 계산하고 **저장할 때만** 내린다 |

★**"상태를 bf16 으로 든다" 는 "bf16 으로 계산한다" 가 아니다.** 매 스텝 fp32 로 올려서
갱신하고 다시 내린다. 임시 텐서는 **파라미터 하나 크기**(최대 2048×768×4×2 ≈ 12.6MB)라
절감분을 잠식하지 않는다.

## ⚠️ 알려진 대가와 한계 — 먼저 적는다

1. **파이썬 루프다.** `torch.optim.AdamW(fused=True)` 의 단일 커널을 잃는다.
   결과 033 이 *"타잉이 줄이는 건 AdamW 스텝(전체의 0.13%)뿐"* 이라 했으므로 옵티마이저는
   스텝 시간의 **약 0.13%** 다. 이게 5배 느려져도 **전체 +0.5% 내외**다. **그래도 실측한다.**
2. **결정론적 반올림(round-to-nearest)이다.** 확률적 반올림(stochastic rounding)이 아니다.
   EMA 에 매 스텝 반올림 편향이 누적될 수 있고, 그건 **긴 런에서만 보인다.**
   → **250스텝으로 판정하지 않는다**(P022B 단계2 설계).
3. **`--resume` 주의.** `torch.optim.Optimizer.load_state_dict` 는 상태를 **파라미터 dtype 으로
   캐스팅**한다 → 그냥 두면 bf16 상태가 **조용히 fp32 로 되돌아간다.**
   아래 `load_state_dict` 오버라이드가 다시 내린다. **그래도 resume 은 권장하지 않는다.**
4. **이 클래스는 `fused` 도 `foreach` 도 아니다.** `capturable`·CUDA Graph 와 함께 쓰지 않는다.

## 자기검증 설계 — 구현 위험과 dtype 위험을 분리한다

`state_dtype=torch.float32` 로 두면 **이 클래스는 참조 AdamW 와 같은 수식**을 돈다.
그 모드(`--opt-dtype fp32c`)와 `torch.optim.AdamW`(`--opt-dtype fp32`)의 손실이 **거의 같아야**
비로소 `bf16` 모드의 차이를 **dtype 탓으로 귀속**할 수 있다.

⚠️ **비트 동일은 기대하지 않는다** — 융합 커널과 연산 순서가 다르다. 인쇄 4자리 일치가 기준이다.
**기본값은 `fp32`(= 종전 `torch.optim.AdamW(fused=True)`)라 켜지 않으면 비트 동일이다.**
"""
from __future__ import annotations

import math

import torch


class AdamWLowPrec(torch.optim.Optimizer):
    """AdamW with configurable optimizer-state dtype (decoupled weight decay).

    `torch.optim.AdamW` 와 **같은 수식**이다:
        p  <- p * (1 - lr*wd)
        m  <- b1*m + (1-b1)*g
        v  <- b2*v + (1-b2)*g^2
        p  <- p - (lr/bc1) * m / (sqrt(v)/sqrt(bc2) + eps)
    다른 것은 `m`·`v` 를 **`state_dtype` 으로 저장**한다는 것뿐이고, **산술은 항상 fp32** 다.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
                 state_dtype=torch.float32):
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"betas 범위 오류: {betas}")
        if lr < 0.0 or eps < 0.0:
            raise ValueError(f"lr/eps 범위 오류: {lr} {eps}")
        self.state_dtype = state_dtype
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    def state_bytes(self):
        """지금 옵티마이저 상태가 실제로 쓰고 있는 바이트. **계측 도구가 이걸 세야 한다.**

        계산값(63.5M × 2 × 2B)을 문서에 적고 끝내면 결과 026 계열 사고가 난다
        (*"계산에 기여하지 않는다" ≠ "메모리에 없다"*). 실제 텐서를 센다.
        """
        n = 0
        for st in self.state.values():
            for k in ("exp_avg", "exp_avg_sq"):
                t = st.get(k)
                if t is not None:
                    n += t.numel() * t.element_size()
        return n

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                if g.is_sparse:
                    raise RuntimeError("AdamWLowPrec 는 sparse gradient 를 지원하지 않는다.")
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p, dtype=self.state_dtype)
                    st["exp_avg_sq"] = torch.zeros_like(p, dtype=self.state_dtype)
                st["step"] += 1
                t = st["step"]
                m, v = st["exp_avg"], st["exp_avg_sq"]

                if m.dtype == torch.float32:
                    # 상태가 이미 fp32 — 제자리 갱신. 복사도 캐스팅도 없다(참조 구현과 동일 경로).
                    m.mul_(b1).add_(g, alpha=1.0 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                    m32, v32 = m, v
                else:
                    # ★저정밀 상태 — fp32 로 **올려서** 갱신하고 **내려서** 저장한다.
                    #   `.float()` 는 dtype 이 다르므로 반드시 복사본을 만든다(제자리 오염 없음).
                    m32 = m.float().mul_(b1).add_(g, alpha=1.0 - b1)
                    v32 = v.float().mul_(b2).addcmul_(g, g, value=1.0 - b2)
                    m.copy_(m32)
                    v.copy_(v32)

                if wd != 0.0:                                   # decoupled weight decay
                    p.mul_(1.0 - lr * wd)
                bc1 = 1.0 - b1 ** t
                bc2 = 1.0 - b2 ** t
                denom = v32.sqrt().div_(math.sqrt(bc2)).add_(eps)
                p.addcdiv_(m32, denom, value=-lr / bc1)
        return loss

    def load_state_dict(self, state_dict):
        """⚠️ 부모 구현이 상태를 **파라미터 dtype 으로 캐스팅**한다 → bf16 이 조용히 fp32 가 된다.

        그대로 두면 `--resume` 한 런이 **의도한 조건이 아니게** 되고, 로그 어디에도 안 남는다
        (함정 계열: *"조용한 무동작이 가장 나쁜 실패"* — 결과 031). 여기서 되돌린다.
        """
        super().load_state_dict(state_dict)
        for st in self.state.values():
            for k in ("exp_avg", "exp_avg_sq"):
                if k in st and st[k].dtype != self.state_dtype:
                    st[k] = st[k].to(self.state_dtype)
