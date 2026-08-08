#!/usr/bin/env python3
"""P022 0단계 마이크로벤치 — FP8(E4M3) `torch._scaled_mm` vs bf16 `F.linear`.

목적: 우리 실제 GEMM 형상(M=micro_bs*seq, K/N=768·2048)에서 Ada FP8 텐서코어가 bf16 cuBLAS 대비
      **순수 GEMM**으로 유의미(>1.2~1.3×)한지 게이트. 아니면 소형모델에선 FP8 무의미 결론.

주의(정직):
  - 이 벤치는 **순수 matmul 시간만** 잰다. 실제 학습은 매 스텝 activation을 FP8로 캐스팅/스케일하는
    elementwise 오버헤드가 추가된다 → 여기서 빨라도 end-to-end 이득은 더 작다(그 상한 판단용).
  - torch._scaled_mm 은 FP8-capable GPU(Ada sm_89+, CUDA 12.x, 지원 빌드)에서만 동작. 아니면 스킵.

실행(사용자 환경):  python scripts/bench_fp8_gemm.py
"""
from __future__ import annotations
import torch, torch.nn.functional as F, time

def _sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

def bench(fn, iters=50, warmup=15):
    for _ in range(warmup): fn()
    _sync(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    _sync(); return (time.perf_counter() - t0) / iters

def to_e4m3(x):
    # per-tensor absmax 스케일 → e4m3 표현범위(max ~448)로. scale 은 fp32 스칼라.
    amax = x.abs().amax().clamp_min(1e-8)
    scale = (448.0 / amax)
    xq = (x * scale).to(torch.float8_e4m3fn)
    inv = (1.0 / scale).to(torch.float32).view(1)   # dequant scale
    return xq, inv

def main():
    if not torch.cuda.is_available():
        print("CUDA 없음 — 종료."); return
    dev = "cuda"; cap = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()}  sm_{cap[0]}{cap[1]}  torch {torch.__version__}")
    has_fp8 = hasattr(torch, "float8_e4m3fn") and hasattr(torch, "_scaled_mm")
    if not has_fp8:
        print("이 torch 빌드에 FP8/_scaled_mm 없음 — 종료(업그레이드 필요)."); return

    # (M, K, N): 우리 실제 형상. M = micro_bs * seq.
    shapes = [
        ("MLP gate/up  (m100)", 8192, 768, 2048),
        ("MLP down     (m100)", 8192, 2048, 768),
        ("Attn q/o     (m100)", 8192, 768, 768),
        ("MLP gate/up  (mb12)", 12288, 768, 2048),
        ("large ref    (2048^)", 8192, 2048, 2048),
    ]
    print(f"\n{'shape':<22}{'M':>6}{'K':>6}{'N':>6}"
          f"{'bf16 ms':>10}{'fp8 ms':>10}{'speedup':>9}{'bf16 TF':>9}{'fp8 TF':>9}")
    print("-" * 91)
    for name, M, K, N in shapes:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)      # F.linear(x, w): [M,N]
        # bf16 기준
        t_bf16 = bench(lambda: F.linear(x, w))
        # FP8: a[M,K] row-major, b 는 [K,N] column-major 필요 → w[N,K].t()
        xq, sx = to_e4m3(x); wq, sw = to_e4m3(w)
        b = wq.t()                                                   # [K,N] column-major
        try:
            _ = torch._scaled_mm(xq, b, scale_a=sx.to(dev), scale_b=sw.to(dev),
                                 out_dtype=torch.bfloat16)
            t_fp8 = bench(lambda: torch._scaled_mm(xq, b, scale_a=sx.to(dev),
                          scale_b=sw.to(dev), out_dtype=torch.bfloat16))
        except Exception as e:
            print(f"{name:<22}{M:>6}{K:>6}{N:>6}  _scaled_mm 실패: {str(e)[:40]}")
            continue
        flop = 2 * M * N * K
        tf_bf16 = flop / t_bf16 / 1e12; tf_fp8 = flop / t_fp8 / 1e12
        print(f"{name:<22}{M:>6}{K:>6}{N:>6}{t_bf16*1e3:>10.3f}{t_fp8*1e3:>10.3f}"
              f"{t_bf16/t_fp8:>8.2f}x{tf_bf16:>9.1f}{tf_fp8:>9.1f}")
    print("-" * 91)
    print("판정: 우리 형상들에서 speedup 이 대체로 >1.2~1.3x 여야 P022 진행 가치. "
          "1.0x 근방/이하면 소형모델에선 FP8도 무의미 → 음성결과로 기록하고 종료.")
    print("주의: 여기 speedup 은 순수 GEMM 상한. 실제 학습엔 캐스팅/스케일 오버헤드가 더해져 이득은 더 작다.")

if __name__ == "__main__":
    main()
