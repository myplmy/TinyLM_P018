#!/usr/bin/env python3
"""P014C 단계0 — **α 그룹 크기를 거칠게 하면 품질이 얼마나 나빠지는가.** 학습 0.

## 왜 이 진단이 P014 의 방향을 정하는가

우리 삼진 가중치는 `w[o,i] = sign[o,i] × α[o, g(i)]` 이고 **α 가 `(O, I//128)`** 개다
(`micro_group=128`). 그런데 외부 저비트 커널들의 스케일 규약이 훨씬 거칠다:

| 규약 | α 개수(mC 기준) | 누가 쓰나 |
|---|---|---|
| **g128**(우리) | 약 **428,000** | `tinylm` |
| **per-row**(행당 1개) | **O 개**(층당 768~2048) | PyTorch 융합 int8 matmul 계열 |
| **per-tensor**(텐서당 1개) | **1** | bitnet.cpp I2_S · TL (실사 확인, 결과문서 §13.2) |

**거친 규약을 쓸 수 있으면 커널을 짜지 않아도 된다** — 이미 있는 융합 연산에 얹으면
`_wq_from_i8()` 의 역양자화가 **사라진다**(그게 우리 CPU 병목의 정체다, 결과 014 §11.4).

**못 쓰면 g128 을 지원하는 커널을 우리가 만들어야 한다.** 그 갈림길을 **학습 없이** 판정한다.

## 어떻게 재는가 — 재학습하지 않는다

이미 배포되는 값 `_wq`(= sign × α_g)를 **그대로 두고 α 만 거칠게 다시 묶는다.**
부호는 보존하고 크기만 바꾸므로 **변환기가 실제로 할 일과 같다.**

L2 최적 재추정: `sign·A − sign·α_g = sign(A − α_g)` 이므로
**`A = mean(|w|)` (비영 위치에 대해)** 가 프로베니우스 오차를 최소화한다.

그리고 **바뀐 가중치로 SQuAD 공통 원문 bpb 를 실제로 잰다** — 가중치 오차(상대 노름)는
품질과 단조가 아니므로 **오차만 보고 판정하지 않는다**(결과 019 의 "구조가 정한 값을 지표로
읽지 말 것" 과 같은 계열의 주의다).

사용:
    python scripts/diag_alpha_group.py --models mC_g8_k4 --preset m100
    python scripts/diag_alpha_group.py --models mC_wsd --preset m100R1c --max-docs 2000
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common_bpb import load_squad_contexts, SQUAD    # noqa: E402  (원문 로딩 단일 소스)

# 재추정할 granularity. None = per-row, 0 = per-tensor, 정수 = 그 그룹 크기
GRANS = [("g128(원본)", 128), ("g256", 256), ("g512", 512),
         ("per-row", None), ("per-tensor", 0)]


def regroup_alpha(wq, gran):
    """`_wq`(sign×α_g)의 **부호를 보존**하고 α 를 `gran` 단위로 L2 최적 재추정한다.

    gran: 정수 = 그 그룹 크기 / None = per-row / 0 = per-tensor
    반환: (새 텐서, fell_back). `fell_back=True` 면 `I % gran != 0` 이라
          **그 텐서만 per-row 로 대체**했다는 뜻이다.

    ★2026-08-07 수정 — 1차 실행(결과 028)에서 **g512 행이 전부 `—` 로 비었다.**
      `I % 512 != 0` 인 텐서 하나(어텐션 `q/o_proj`, I=768)를 만나면 `None` 을 돌려
      호출부가 **granularity 전체를 버렸다.** 판정은 per-row 로 이미 갈렸으므로 결론에는
      영향이 없었지만, "재려던 양을 못 잰" 것은 사실이다(함정 1의 계열).
      → 나눠지지 않는 텐서만 per-row 로 내려가고, **몇 개가 내려갔는지 표에 적는다.**

    ★`.detach()` 이유 — `freeze_quant()` 가 만든 `_wq` 는 `ternary()` autograd Function 을
      타고 나온 값이라 `requires_grad=True` 다. 그대로 `float(...)` 하면 1차 실행처럼
      UserWarning 이 뜬다(경고 자체는 무해했지만 **그래프를 붙들어 메모리를 쓴다**).
    """
    import torch
    wq = wq.detach()
    O, I = wq.shape
    sign = torch.sign(wq)
    aw = wq.abs()
    nz = (aw > 0)

    def _per_row():
        cnt = nz.sum(dim=1, keepdim=True).clamp_min(1)
        return sign * (aw.sum(dim=1, keepdim=True) / cnt)

    if gran == 0:                                   # per-tensor
        cnt = nz.sum().clamp_min(1)
        return sign * (aw.sum() / cnt), False
    if gran is None:                                # per-row
        return _per_row(), False
    if I % gran != 0:                               # ★버리지 않고 per-row 로 내린다
        return _per_row(), True
    G = I // gran
    awg = aw.reshape(O, G, gran)
    nzg = (awg > 0)
    cnt = nzg.sum(dim=2, keepdim=True).clamp_min(1)
    A = awg.sum(dim=2, keepdim=True) / cnt
    return (sign.reshape(O, G, gran) * A).reshape(O, I), False


def main():
    ap = argparse.ArgumentParser(description="P014C 단계0 — α 그룹 크기별 품질 대가")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data", default="ko-en")
    ap.add_argument("--tokens", default="300M")
    ap.add_argument("--preset", default="m100")
    ap.add_argument("--arch", nargs="+", default=None)
    ap.add_argument("--max-docs", type=int, default=4000)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from tinylm import paths
    from tinylm.infer.generate import load_model
    from tinylm.data import tokenizer_path
    from tinylm.model.ternary import TLinear
    from tokenizers import Tokenizer

    if not SQUAD.exists():
        raise SystemExit(f"[STOP] SQuAD 원문이 없다: {SQUAD}")
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    docs = load_squad_contexts(limit=a.max_docs)
    text = "\n\n".join(docs)
    n_bytes = len(text.encode("utf-8"))

    W = 96
    print("=" * W)
    print("  P014C 단계0 — α 그룹 크기를 거칠게 하면 품질이 얼마나 나빠지는가")
    print("=" * W)
    print(f"  device={dev}  SQuAD context {len(docs):,}개  {n_bytes/1e6:.2f} MB  seq={a.seq}")
    print("  ★재학습하지 않는다. 배포되는 `_wq` 의 **부호를 보존**하고 α 만 다시 묶는다.")
    print("    A = mean(|w|)(비영 위치) 가 프로베니우스 오차 최소 = 변환기가 실제로 할 일.")
    print("  ★판정은 **bpb** 로 한다. 가중치 상대오차는 품질과 단조가 아니다(참고값).")
    print()

    archs = a.arch or ["dense" if t.startswith(("p6d", "dense", "p12d")) else "tied"
                       for t in a.models]

    for tag, arch in zip(a.models, archs):
        ck = paths.resolve_ckpt(a.preset, a.data, a.tokens, tag)
        if not ck.exists():
            print(f"  [건너뜀] 체크포인트 없음: {ck.name}")
            continue
        tok = Tokenizer.from_file(str(tokenizer_path(a.data)))
        ids = np.array(tok.encode(text).ids, dtype=np.int64)
        bpt = n_bytes / max(len(ids), 1)

        print(f"  ── {tag} ({arch})   토큰 {len(ids):,}  bytes/token {bpt:.4f}")
        print(f"     {'granularity':<14}{'α 개수':>12}{'가중치 상대오차':>16}{'손실':>10}{'bpb':>10}{'Δbpb':>10}")
        print("     " + "-" * (W - 8))

        base_bpb = None
        for name, gran in GRANS:
            model, cfg, _ = load_model(arch=arch, ckpt_path=str(ck), device=dev)
            model.set_anneal(1.0); model.eval(); model.freeze_quant()

            tls = [m for m in model.modules() if isinstance(m, TLinear)]
            n_alpha, num, den, n_fb = 0, 0.0, 0.0, 0
            with torch.no_grad():                    # ★그래프를 만들지 않는다
                for m in tls:
                    wq = m._wq.detach()
                    new, fell_back = regroup_alpha(wq, gran)
                    num += float((new - wq).pow(2).sum())
                    den += float(wq.pow(2).sum())
                    O, I = wq.shape
                    if fell_back:                    # 이 텐서만 per-row 로 내려갔다
                        n_fb += 1
                        n_alpha += O
                    else:
                        n_alpha += (1 if gran == 0 else (O if gran is None else O * (I // gran)))
                    m._wq = new
            rel = math.sqrt(num / max(den, 1e-12))
            if n_fb:                                 # 표에 정직하게 적는다
                name = f"{name}*{n_fb}"

            n_crop = (len(ids) - 1) // a.seq
            tot, cnt = 0.0, 0
            with torch.no_grad():
                for i in range(0, n_crop, a.micro_bs):
                    ch = [ids[j*a.seq:(j+1)*a.seq + 1]
                          for j in range(i, min(i + a.micro_bs, n_crop))]
                    t = torch.from_numpy(np.stack(ch)).to(dev)
                    x, y = t[:, :-1], t[:, 1:]
                    with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
                        logits = model(x)
                    l = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                        y.reshape(-1))
                    tot += float(l) * y.numel(); cnt += y.numel()
            loss = tot / cnt
            bpb = loss / math.log(2) / bpt
            if base_bpb is None:
                base_bpb = bpb
            d = bpb - base_bpb
            print(f"     {name:<14}{n_alpha:>12,}{rel:>16.5f}{loss:>10.4f}{bpb:>10.4f}"
                  f"{d:>+10.4f}")
            del model
            if dev == "cuda":
                torch.cuda.empty_cache()
        print()

    print("=" * W)
    print("  ★어떻게 읽는가")
    print("=" * W)
    print("  per-row 의 Δbpb 가 **0.008 미만**")
    print("    → ★커널을 짜지 않아도 된다. PyTorch 의 융합 int8 matmul(per-row scale)에 얹으면")
    print("      역양자화가 사라진다. P014C 단계1 로 간다. **가장 값싼 결말.**")
    print("  per-row 는 넘고 g256/g512 는 안 넘는다")
    print("    → 중간 granularity 로 타협 가능. 단 그 규약을 받는 융합 연산이 있어야 한다")
    print("      (없으면 결국 커널을 짜야 하고, 그러면 g128 을 유지하는 게 낫다)")
    print("  per-row 도 per-tensor 도 크게 넘는다")
    print("    → ★g128 을 지원하는 커널을 우리가 만들어야 한다(P014 안 C). 그리고 이 표가")
    print("      **왜 외부 구현을 그대로 못 쓰는지에 대한 정량 근거**가 된다.")
    print()
    print("  ★per-tensor 열은 bitnet.cpp I2_S/TL 규약이다 — 그 값이 곧 '지금 BitNet 에 넣으면")
    print("    품질이 이만큼 나빠진다' 이다(실사 §13.2).")
    print()
    print("  ★표 읽기 — `g512*8` 처럼 `*N` 이 붙으면 그 granularity 에서 **N개 텐서가")
    print("    나눠지지 않아 per-row 로 내려갔다**는 뜻이다(어텐션 q/o_proj 는 I=768).")
    print("    그 행은 순수한 g512 가 아니라 **혼합**이므로 서열 판단에 쓰지 않는다.")
    print("  ★한계")
    print("   · 영문 전용(SQuAD). 한국어 쪽 대가는 이 진단으로 안 나온다")
    print("   · 재학습하지 않았다. **재학습하면 거친 α 에 적응해 대가가 줄어들 수 있다**")
    print("     — 즉 이 표는 **비관적 상한**이다")
    print("   · 가중치 상대오차는 참고값이다. 품질과 단조가 아니다")
    print("   · 모델당 시드 1개")


if __name__ == "__main__":
    main()
