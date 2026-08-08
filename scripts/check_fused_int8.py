#!/usr/bin/env python3
"""P014C 단계1 — **PyTorch 에 per-row 스케일 융합 int8 matmul 이 있는가.** GPU 0 · 1분.

## 왜 이 한 줄이 P014 의 방향을 정하는가

결과 028 이 **per-row α 의 품질 대가가 +0.0050~0.0063 bpb = 분해능(0.008) 미만**임을 재서
게이트 G0 을 통과시켰다. 그래서 남은 질문은 하나다:

> **per-row 스케일을 받는 융합 연산이 실제로 존재하는가.**

- **있으면** `TLinear._wq_from_i8()` 의 **fp32 복원이 사라진다** — 그게 우리 CPU 병목의
  정체이고(결과 014 §11.4: 층당 2.1~2.2ms × 20층 ≈ 43ms/토큰) **커널을 짤 필요가 없다.**
- **없으면** 결과 028 의 표가 **"우리가 커널을 짜야 하는" 정량 근거**가 된다(P014 안 C).

⚠️ **이 진단은 "있는가" 만 답한다.** 있다고 해서 빨라진다는 뜻이 아니다 —
**속도는 단계2 가 잰다**(단독 실행 + 직전 5분 유휴 + 같은 배치에 기준선, 결과 016 §12.5).
여기서 속도를 재려는 유혹을 막기 위해 **이 스크립트는 타이밍을 찍지 않는다.**

## 무엇을 보는가

우리 `TLinear` 의 shape 은 `(O, I)` 이고 per-row 는 **행(=출력 채널)당 α 하나**다.
`torch._weight_int8pack_mm(x, w_int8, scales)` 가 정확히 그 규약이라 1순위 후보다.
그 밖에 `torchao`·`torch.compile` 융합·oneDNN 경로도 이름만 확인한다.

사용:
    python scripts/check_fused_int8.py
"""
from __future__ import annotations

import sys

W = 92


def head(t):
    print()
    print("=" * W)
    print(f"  {t}")
    print("=" * W)


def main():
    try:
        import torch
    except ImportError:
        print("[stop] torch 가 없다. 이 진단은 torch 설치 환경에서 돌린다.")
        return 3

    head("P014C 단계1 — per-row 융합 int8 matmul 존재 확인 (속도는 안 잰다)")
    print(f"  torch {torch.__version__}   cuda={torch.cuda.is_available()}")
    print("  ★결과 028: per-row Δbpb +0.0050~+0.0063 < 0.008 → 규약은 감당된다.")
    print("    남은 질문은 **연산이 있는가** 하나다.")

    # ── 1순위: torch._weight_int8pack_mm (per-row scale 규약) ──────────────
    head("1순위  torch._weight_int8pack_mm  (per-row scale — 우리 규약과 일치)")
    cands = {
        "torch._weight_int8pack_mm": getattr(torch, "_weight_int8pack_mm", None),
        "torch.ops.aten._weight_int8pack_mm": getattr(torch.ops.aten, "_weight_int8pack_mm", None),
    }
    found = None
    for name, fn in cands.items():
        ok = fn is not None
        print(f"  {name:42s} {'있음' if ok else '없음'}")
        if ok and found is None:
            found = (name, fn)

    if found:
        name, fn = found
        print(f"\n  ★{name} 가 있다. **실제로 호출해 규약을 확인한다** —")
        print(f"    이름만 보고 인용하면 함정 10(파서에 있는데 전달 안 됨)의 반복이다.")
        # (M,K) x (N,K)int8 x (N,)scale -> (M,N).  우리 TLinear 는 (O,I) = (N,K).
        for dev in (["cpu"] + (["cuda"] if torch.cuda.is_available() else [])):
            for (M, K, N) in [(8, 768, 2048), (8, 2048, 768), (1, 768, 768)]:
                x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
                w = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=dev)
                s = torch.rand(N, dtype=torch.bfloat16, device=dev) * 0.05 + 0.01
                try:
                    y = fn(x, w, s)
                    ref = (x.float() @ w.float().t()) * s.float()
                    rel = float((y.float() - ref).norm() / ref.norm().clamp_min(1e-9))
                    print(f"    {dev:4s} M={M:2d} K={K:4d} N={N:4d} → out {tuple(y.shape)}"
                          f"  dtype {y.dtype}  상대오차 vs 참조 {rel:.5f}")
                except Exception as e:                      # noqa: BLE001
                    print(f"    {dev:4s} M={M:2d} K={K:4d} N={N:4d} → 🚫 {type(e).__name__}: {e}")
        print()
        print("  ★상대오차는 bf16 누산 때문에 0 이 아니다 — **자릿수만** 본다(1e-2 급이면 정상).")
        print("    0.5 이상이면 스케일 규약이 우리 가정과 다르다는 뜻이니 그걸 보고할 것.")

    # ── 대안 경로 ─────────────────────────────────────────────────────────
    head("대안 경로 (1순위가 없거나 우리 shape 을 거부할 때)")
    print("  torchao                : ", end="")
    try:
        import torchao                                       # noqa: F401
        print(f"있음 (버전 {getattr(torchao, '__version__', '?')})")
    except ImportError:
        print("없음  (pip install torchao — 단 의존성이 늘어난다)")

    for n in ("_int_mm", "_scaled_mm"):
        print(f"  torch.{n:21s}: {'있음' if getattr(torch, n, None) else '없음'}")
    try:                                     # 버전에 따라 속성 구성이 다르다 — 죽지 않게
        ok = bool(torch.backends.mkldnn.is_available())
        print(f"  torch.backends.mkldnn  : {'사용가능' if ok else '아니오'}   (oneDNN — CPU 융합 경로)")
    except Exception as e:                                       # noqa: BLE001
        print(f"  torch.backends.mkldnn  : 확인 실패 ({type(e).__name__})")

    # ── 판정 ──────────────────────────────────────────────────────────────
    head("★어떻게 읽는가 — 게이트 G1")
    if found:
        print("  ✅ **연산이 있다** → 단계2(속도 게이트)로 간다.")
        print("     구현: `TLinear` 에 per-row 경로를 **기본 off 플래그**로 추가하고")
        print("     ①로짓 오차 ②CPU tok/s 를 잰다. 기준선은 언팩 캐시 **19.07 tok/s**")
        print("     (결과 016 §15) — **그걸 못 넘으면 이 경로도 접는다.**")
        print("  ⚠️ per-row 로 가려면 **α 를 per-row 로 재추정한 모델이 필요**하다.")
        print("     결과 028 은 재학습 없이 쟀고 그건 **비관적 상한**이다. 채택 시에는")
        print("     per-row 규약으로 재학습해야 STE 가 그 규약에 적응한다(약 3.5h).")
    else:
        print("  🚫 **1순위 연산이 없다.** 대안 순서: torchao → torch.compile 융합 → 자체 커널.")
        print("     자체 커널로 가면 결과 028 의 표가 **왜 외부 규약을 못 쓰는지의 정량 근거**다")
        print("     (per-tensor = bitnet.cpp 규약이 **0.261 nats(21.7σ)** 를 잃는다).")
    print()
    print("  ★한계 — 이 진단은 **존재만** 본다. 속도는 단계2, 품질은 단계3 의 몫이다.")
    print("    그래서 여기서는 **일부러 타이밍을 찍지 않는다**(측정 위생: 결과 016 §12.5 는")
    print("    단독 실행 + 직전 5분 유휴 + 같은 배치에 기준선을 요구한다).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
