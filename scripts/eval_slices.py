#!/usr/bin/env python3
"""P028 단계 0.5 — **인과** 확인: train 앞부분 vs 뒷부분 vs val 손실 비교.

★핵심 착상
  단계 0(`diag_cache.py`)은 **통계**를 줬을 뿐 인과가 아니었다. 그리고 크기가 안 맞는다 —
  관측 괴리는 3.14 nats(ppl 23배)인데 유니그램 JS·한글비율·중복률을 다 합쳐도 그만큼이 안 나온다.

  결정적 테스트: **둘 다 학습한 구간**인 train 의 앞/뒤를 비교한다.
  그러면 "본 적 없음(generalization)" 변수가 제거되고 **"그 구간이 원래 어려운가"만 남는다.**

      앞 ≈ 뒤 ≪ val   →  순수하게 '본 적 없음' 때문 → 중복·암기(H1)가 주범
      앞 ≪ 뒤 ≈ val   →  꼬리 구간이 원래 어렵다 → 분포이동(H2) 확정
      앞 ≈ 뒤 ≈ val   →  캐시로 설명 안 됨 → 학습설정·토크나이저 등 다른 원인

  ko-en 을 같은 방식으로 재면 **정상 코퍼스의 기준선**이 나온다(ko-en 은 val−train ≈ 0 이었다).

★한계
  - 앞/뒤 둘 다 **학습한** 구간이라 "얼마나 외웠나"는 분리하지 못한다. 분리하는 건 "구간 난이도"다.
  - 손실은 크롭 경계·문서 경계에 따라 흔들린다 → 구간당 여러 크롭을 평균한다.
  - **인과를 100% 확정하진 못한다.** 재생성 후 재학습만이 최종 확인이다.

사용법
  python scripts/eval_slices.py --data ko-edu-en --tokens 300M --tag dense
  python scripts/eval_slices.py --data ko-en --tokens 300M --tag dense --arch dense   # 대조군
  python scripts/eval_slices.py --data ko-edu-en --tokens 300M --tag dense --docstats  # val 문서통계도
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _tok(s):
    s = str(s).strip().upper()
    if s.endswith("M"):
        return int(float(s[:-1]) * 10 ** 6)
    if s.endswith("K"):
        return int(float(s[:-1]) * 10 ** 3)
    return int(s)


def slice_loss(model, arr, lo, hi, seq, n_crops, device, rng):
    """[lo,hi) 구간에서 n_crops 개 크롭을 뽑아 teacher-forcing 손실 평균."""
    import torch
    import torch.nn.functional as F
    tot, cnt = 0.0, 0
    hi = min(hi, len(arr) - seq - 1)
    if hi <= lo:
        return float("nan")
    with torch.no_grad():
        for _ in range(n_crops):
            i = int(rng.integers(lo, hi))
            chunk = np.asarray(arr[i:i + seq + 1]).astype(np.int64)
            x = torch.tensor(chunk[:-1], device=device).unsqueeze(0)
            y = torch.tensor(chunk[1:], device=device).unsqueeze(0)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            tot += float(loss); cnt += 1
    return tot / max(cnt, 1)


def main():
    ap = argparse.ArgumentParser(description="P028 단계0.5 구간별 손실")
    ap.add_argument("--data", required=True)
    ap.add_argument("--doc-filter", action="store_true",

                    help="(P037 단계2) 필터된 캐시({data}_{N}_filtered)를 읽는다. 결과 023: 이 플래그가 없어 검증이 옛 캐시를 쟀다")
    ap.add_argument("--tokens", default="300M")
    ap.add_argument("--tag", default="dense")
    ap.add_argument("--arch", default="dense", choices=["dense", "tied"])
    ap.add_argument("--preset", default="m100")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--crops", type=int, default=200, help="구간당 크롭 수(많을수록 안정)")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--docstats", action="store_true", help="val 문서 길이 통계도 출력")
    a = ap.parse_args()

    from tinylm import paths
    from tinylm.infer.generate import load_model

    # ★P037 단계2: 필터 캐시는 별도 디렉터리다(결과 023 — 이걸 안 봐서 옛 캐시를 쟀다)
    _suf = "_filtered" if a.doc_filter else ""
    cache = paths.DATA_CACHE / f"{a.data}_{_tok(a.tokens)}{_suf}"
    if not (cache / "train.bin").exists():
        print(f"[!] 캐시 없음: {cache}")
        return 1
    base = f"{a.preset}_{a.data}_{a.tokens}"
    ck = paths.RUNS / "ckpt" / f"{base}_{a.tag}.pt"
    if not ck.exists():
        print(f"[!] 체크포인트 없음: {ck}")
        return 1

    tr = np.memmap(cache / "train.bin", dtype=np.uint16, mode="r")
    va = np.memmap(cache / "val.bin", dtype=np.uint16, mode="r")
    model, cfg, device = load_model(arch=a.arch, ckpt_path=str(ck))
    rng = np.random.default_rng(a.seed)
    n = len(tr)
    W = min(1_500_000, n // 10)          # val 크기와 비슷한 폭

    print("=" * 78)
    print(f"  P028 단계0.5 — {a.data} / {a.tag} ({a.arch})")
    print(f"  train {n/1e6:.1f}M 토큰, val {len(va)/1e6:.2f}M, 구간폭 {W/1e6:.2f}M, 크롭 {a.crops}개/구간")
    print("=" * 78)

    res = {}
    for name, lo, hi in [("train 앞 (구간 1/10, 학습함)", 0, W),
                         ("train 중간 (구간 5/10, 학습함)", n // 2, n // 2 + W),
                         ("train 뒤 (구간 10/10, 학습함, val 직전)", max(0, n - W), n)]:
        L = slice_loss(model, tr, lo, hi, a.seq, a.crops, device, rng)
        res[name] = L
        print(f"  {name:38} loss {L:.4f}   ppl {math.exp(L):8.1f}")
    Lv = slice_loss(model, va, 0, len(va), a.seq, a.crops, device, rng)
    res["val (학습 안 함)"] = Lv
    print(f"  {'val (학습 안 함)':38} loss {Lv:.4f}   ppl {math.exp(Lv):8.1f}")

    head = res["train 앞 (구간 1/10, 학습함)"]
    tail = res["train 뒤 (구간 10/10, 학습함, val 직전)"]
    print("\n  " + "-" * 74)
    print(f"  뒤 − 앞 = {tail-head:+.4f}   (둘 다 학습한 구간 → '구간 난이도' 차이)")
    print(f"  val − 뒤 = {Lv-tail:+.4f}   ('본 적 없음'의 순수 비용)")
    print(f"  val − 앞 = {Lv-head:+.4f}   (합계)")
    print("\n  판정")
    if abs(tail - head) < 0.15 and Lv - tail > 0.5:
        print("    → 앞 ≈ 뒤 ≪ val : 구간 난이도는 균일한데 val 만 나쁘다.")
        print("      **H1(중복·암기)** 쪽. val 이 '본 적 없다'는 이유만으로 이만큼 나쁘다는 뜻.")
    elif tail - head > 0.5 and abs(Lv - tail) < 0.5:
        print("    → 앞 ≪ 뒤 ≈ val : **꼬리 구간이 원래 어렵다 → H2(분포이동) 확정.**")
        print("      val 은 그 꼬리의 연장일 뿐이고, 말미 분할이 직접 원인이다.")
    elif tail - head > 0.5 and Lv - tail > 0.5:
        print("    → 앞 ≪ 뒤 ≪ val : **H1 과 H2 가 함께** 작용한다(가장 흔한 실제 상황).")
        print(f"      분포이동 기여 {tail-head:+.3f} / 미관측 기여 {Lv-tail:+.3f}")
    else:
        print("    → 뚜렷한 패턴 없음. 캐시 통계로 설명되지 않는다 —")
        print("      학습설정·토크나이저·풀 비율 등 다른 원인을 봐야 한다.")

    if a.docstats:
        print("\n  " + "-" * 74)
        print("  val 문서 길이 통계 (단계0 도구의 사각지대 — 거대 문서 1개가 val 을 지배할 수 있다)")
        # ★2026-07-31 수정: eos 를 **추정하지 않고 토크나이저에 조회**한다.
        #   종전: sample=train 앞 5M 에서 id 0~7 최빈값을 eos 로 추측 → 문서 경계를 절반만 찾아
        #   "val 의 23.5% 가 단일 문서(352,399토큰)" 라는 **틀린 경보**를 냈다(결과 011 §0.6-1).
        #   train 앞 5M 과 val 은 분포가 다르다 — 이 도구가 발견한 현상이 이 도구를 망가뜨렸다.
        from tinylm.data import tokenizer_path as _tp
        from tokenizers import Tokenizer as _Tk
        eos = _Tk.from_file(str(_tp(a.data))).token_to_id("<eos>")
        if eos is None:
            eos = 2
        print(f"    (eos id={eos}, 토크나이저 조회)")
        pos = np.flatnonzero(np.asarray(va) == eos)
        if len(pos) > 1:
            lens = np.diff(np.concatenate(([-1], pos)))
            print(f"    문서 {len(lens)}개  평균 {lens.mean():.0f}  중위 {np.median(lens):.0f}  "
                  f"최대 {lens.max():,}  최대/전체 {lens.max()/len(va)*100:.1f}%")
            if lens.max() / len(va) > 0.2:
                print("    [!] 단일 문서가 val 의 20% 초과 — 그 문서 하나가 val 손실을 지배할 수 있다.")
        else:
            print("    문서 경계를 찾지 못함(eos 추정 실패)")

    print("\n  " + "-" * 74)
    print("  ★한계: 앞/뒤 둘 다 학습한 구간이라 '얼마나 외웠나'는 분리하지 못한다.")
    print("        인과의 최종 확인은 캐시 재생성 후 재학습뿐이다.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
