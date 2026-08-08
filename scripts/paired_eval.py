#!/usr/bin/env python3
"""P032 — **결정적 full-val + paired per-crop 비교**. 두 모델의 차이를 노이즈 아래로 밀어넣는다.

★왜 필요한가 (계획 P032 §1)
   결과 012 는 `mA_g4s34_k4` 3.7003 vs `p6d` 3.7045 = **-0.0042** 를 보고했다. 그런데
   σ = 0.012 이고 실무 분해능은 0.024 다. 이 차이를 **평균 비교**로 확정하려면 각 그룹
   49런 = 약 196시간이 필요하다 — 비현실적이다. 계획서가 그 경로를 폐기한 이유다.

★이 스크립트가 하는 두 가지 (둘 다 **학습을 다시 하지 않는다**)
   1. **결정적 full-val** — `evaluate()` 는 val 에서 50배치를 **랜덤 오프셋으로 복원추출**한다
      (1.5M 중 약 27%). 그래서 같은 체크포인트도 부를 때마다 값이 흔들린다. 여기서는
      `0, seq, 2*seq, ...` 로 **겹치지 않게 전부** 순회한다 → 같은 모델은 **항상 같은 값**.
      이것만으로 σ 의 eval 성분이 **0** 이 된다.
   2. **paired per-crop 비교** — 두 모델을 **같은 크롭 집합**에 통과시키고 크롭별 차이
      `d_i = loss_A(crop_i) - loss_B(crop_i)` 를 본다. "이 크롭이 원래 어려운가"는 두 모델에
      공통이므로 차이에서 **상쇄된다**. 독립 평균 비교보다 훨씬 작은 차이를 검출한다.

★★반드시 결과문서에 적을 한계 (계획 P032 §2.2)
   paired 비교는 **eval 노이즈만** 없앤다. **학습 재현 노이즈(시드)는 그대로 남는다.**
   따라서 이 스크립트가 확정할 수 있는 명제는
       "이 **두 체크포인트**는 다르다 / 같다"
   이고, 확정할 수 없는 명제는
       "이 **아키텍처**가 더 낫다"
   이다. 후자는 여전히 여러 시드가 필요하다. **이 구분을 흐리면 결과 015 의 실수
   (best 로 읽어 부호가 뒤집힌 사건)를 다른 형태로 반복하는 것이다.**

사용법
  python scripts/paired_eval.py --models mA_g4s34_k4 mC_g8_k4 p6d
  python scripts/paired_eval.py --models mA_g4s34_k4 p6d --seq 1024 --micro-bs 8
"""
from __future__ import annotations

import argparse
import itertools
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_MODELS = ["mA_g4s34_k4", "mC_g8_k4", "p6d"]


def _arch_of(tag):
    return "dense" if tag.startswith(("p6d", "dense", "p12d")) else "tied"


def banner(s, ch="="):
    print("\n" + ch * 92)
    print(f"  {s}")
    print(ch * 92)


def full_val_losses(model, cfg, data_dir, seq, micro_bs, device):
    """val 전체를 **겹치지 않게 순차 순회**하며 크롭별 손실 벡터를 돌려준다.

    반환: (크롭별 손실 리스트, 사용 토큰 수). 같은 체크포인트면 **몇 번 불러도 같다.**
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    d = np.memmap(Path(data_dir) / "val.bin", dtype=np.uint16, mode="r")
    n_crop = (len(d) - 1) // seq
    starts = [i * seq for i in range(n_crop)]

    was = cfg.quant_anneal
    model.set_anneal(1.0)                 # 배포 상태(완전 삼진)
    model.eval()
    model.freeze_quant()
    out = []
    with torch.no_grad():
        for i in range(0, n_crop, micro_bs):
            chunk = starts[i:i + micro_bs]
            arr = np.stack([d[s:s + seq + 1] for s in chunk]).astype(np.int64)
            t = torch.from_numpy(arr).to(device)
            x, y = t[:, :-1], t[:, 1:]
            with torch.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
                logits = model(x)
            # ★크롭별로 따로 잰다 — paired 비교는 크롭 단위 대응이 있어야 성립한다.
            per = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y.reshape(-1), reduction="none").reshape(y.shape).mean(dim=1)
            out.extend(float(v) for v in per)
    model.clear_quant()
    model.train()
    model.set_anneal(was)
    return out, n_crop * seq


def paired_stats(a, b):
    """a - b 의 paired 통계. t 는 정규근사로만 읽는다(자유도가 크므로 무방)."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    sd = statistics.stdev(d) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    t = m / se if se else float("nan")
    win = sum(1 for v in d if v < 0)
    return m, sd, se, t, n, win


def main():
    ap = argparse.ArgumentParser(description="P032 결정적 full-val + paired 비교")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--data", default="ko-en")
    ap.add_argument("--tokens", default="300M")
    ap.add_argument("--preset", default="m100")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    import torch
    from tinylm import paths
    from tinylm.data import prepare
    from tinylm.infer.generate import load_model

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base = f"{a.preset}_{a.data}_{a.tokens}"
    n_tok = int(float(a.tokens.rstrip("MmBb")) * (1e9 if a.tokens[-1] in "Bb" else 1e6))

    banner("P032 — 결정적 full-val + paired per-crop 비교 (학습 없음)", "#")
    print(f"  device={dev}  seq={a.seq}  micro_bs={a.micro_bs}  data={a.data} {a.tokens}")
    print("  ★val 을 겹치지 않게 전부 순회한다 → eval 샘플링 노이즈 성분 = 0.")
    print("  ★크롭별 손실을 저장해 두 모델을 **같은 크롭 위에서** 뺀다.")
    print("  ⚠️ 이것은 **체크포인트 간** 비교다. 아키텍처 우열은 여기서 나오지 않는다.")

    meta = prepare(a.data, n_tok)
    per = {}
    for tag in a.models:
        ck = paths.resolve_ckpt(a.preset, a.data, a.tokens, tag)
        if not ck.exists():
            print(f"\n  [건너뜀] 체크포인트 없음: {ck.name}")
            continue
        model, cfg, _ = load_model(arch=_arch_of(tag), ckpt_path=str(ck), device=dev)
        losses, used = full_val_losses(model, cfg, meta["dir"], a.seq, a.micro_bs, dev)
        per[tag] = losses
        mean = sum(losses) / len(losses)
        print(f"\n  {tag:<16} full-val {mean:.4f}   크롭 {len(losses)}개 / {used:,} 토큰")
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    if len(per) < 2:
        print("\n  비교할 모델이 2개 미만이다.")
        return 2

    # 크롭 수가 다르면 대응이 깨진다. 같은 data/seq 면 같아야 하므로 방어적으로 자른다.
    n = min(len(v) for v in per.values())
    for k in per:
        per[k] = per[k][:n]

    banner("결정적 full-val (같은 체크포인트면 재실행해도 동일한 값)")
    print(f"  {'모델':<16}{'full-val':>10}{'크롭 표준편차':>14}")
    print("  " + "-" * 44)
    for tag, v in sorted(per.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        print(f"  {tag:<16}{sum(v)/len(v):>10.4f}{statistics.stdev(v):>14.4f}")

    banner("★paired per-crop 비교 — 이것이 판정의 근거다")
    print(f"  {'A vs B':<30}{'mean(A-B)':>12}{'SE':>9}{'t':>8}{'A 승률':>9}  판정")
    print("  " + "-" * 88)
    RES = 0.024                                    # 실무 분해능(2σ, CLAUDE.md)
    for x, y in itertools.combinations(per, 2):
        m, sd, se, t, nn, win = paired_stats(per[x], per[y])
        # |t| ^> 2 면 "이 두 체크포인트는 다르다". 크기 판정은 분해능과 따로 본다.
        if abs(t) < 2:
            verdict = "구분 불가(체크포인트 수준에서도)"
        elif abs(m) < RES:
            verdict = "차이 유의하나 분해능(0.024) 미만 — 실무상 동급"
        else:
            verdict = "★유의하고 분해능 초과"
        print(f"  {x + ' vs ' + y:<30}{m:>+12.4f}{se:>9.4f}{t:>8.2f}{win/nn:>8.1%}  {verdict}")

    print("""
  읽는 법
    · SE 가 결과 012 의 σ=0.012 보다 **한 자릿수 작다**면 paired 설계가 의도대로 작동한 것이다.
    · |t| ^> 2 는 "이 두 **체크포인트**가 다르다"만 말한다. 시드가 하나이므로
      **아키텍처 우열의 근거가 아니다.** 그 주장에는 여러 시드가 필요하다(P032 §2.2).
    · 'A 승률' 은 크롭 중 A 가 이긴 비율이다. 평균이 작아도 승률이 한쪽으로 크게 쏠리면
      일관된 차이이고, 50% 근처면 큰 크롭 몇 개가 평균을 끌고 있는 것이다 — 둘은 다르다.

  한계
    · full-val 은 **결정적**이지만 **여전히 이 val 셋**이다. 데이터 문제(P037)는 그대로 남는다.
    · 크롭 경계에서 컨텍스트가 끊긴다(긴 문서에 약간 불리). 두 모델에 공통이라 차이에서는 상쇄된다.
    · bf16 autocast(cuda)면 크롭별 손실에 ULP 노이즈가 섞인다. 정밀 비교는 --device cpu 로
      한 번 더 확인한다(느리지만 fp32).""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
