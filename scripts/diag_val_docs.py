#!/usr/bin/env python3
"""P028 단계 0.6 — **문서 단위** val 손실 분해. "거대 문서 하나가 val 을 지배하는가"

★왜 이게 다음 단계인가 (단계 0.5 결과)
  ko-edu-en 의 val−뒤(미관측 순수비용)가 **+2.198** 이었다. 대조군 ko-en 은 **+0.153**.
  **14배**다. 같은 코퍼스에서 뽑은 미관측 텍스트가 이만큼 비쌀 이유가 없다.
  그런데 같은 단계에서 **val 의 23.5% 가 단일 문서**(352,399 토큰)라는 게 드러났다.
  → **"본 적 없어서 비싼 것"과 "그 한 문서가 이상해서 비싼 것"이 구분되지 않았다.**

  이 스크립트는 그걸 분리한다. `prepare()` 를 고치기 **전에** 해야 한다 —
  원인에 따라 **고치는 방법이 완전히 다르기 때문**이다:
      한 문서가 지배  → **필터링 문제**(길이 상한·문서 비중 상한). 좁은 수정
      전 문서가 고르게 나쁨 → **분할 문제**(무작위 분할). 전 데이터셋 영향, 큰 수정

사용법
  python scripts/diag_val_docs.py --data ko-edu-en --tokens 300M --arch dense
  python scripts/diag_val_docs.py --data ko-en     --tokens 300M --arch dense   # 대조군
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _charstats(s: str) -> str:
    """문서 성격을 한 줄로. 왜 어려운지의 단서(표·코드·나열·비한국어)를 본다."""
    if not s:
        return "(빈 문자열)"
    n = len(s)
    ko = sum(1 for c in s if "\uac00" <= c <= "\ud7a3")
    ascii_ = sum(1 for c in s if ord(c) < 128)
    digit = sum(1 for c in s if c.isdigit())
    nl = s.count("\n")
    pipe = s.count("|")
    # 반복도: 서로 다른 줄의 비율(작을수록 표·나열처럼 반복적)
    lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
    uniq = len(set(lines)) / max(len(lines), 1)
    return (f"한글 {ko/n:5.1%}  ASCII {ascii_/n:5.1%}  숫자 {digit/n:5.1%}  "
            f"줄바꿈 {nl/n:5.1%}  '|' {pipe/n:5.1%}  줄 고유율 {uniq:5.1%}")


def _inspect(arr, docs, rows_by_contrib, tok, eos, a):
    """P037 단계1 — 0.6 의 경계 판정이 믿을 만한지 A~D 로 검사한다."""
    import numpy as np
    print("\n  " + "#" * 74)
    print("  (P037 단계1) 경계 판정 신뢰도 검사 — 0.5 가 도구 결함이었으므로 0.6 도 검증한다")
    print("  " + "#" * 74)

    # --- A. 원문에 <eos> 리터럴이 섞일 수 있는가 ---
    print("\n  [A] `<eos>` 리터럴이 문서를 과분할할 수 있는가")
    lit = tok.encode("<eos>").ids
    print(f"      tok.encode('<eos>').ids = {lit}")
    if lit == [eos]:
        print(f"      ⚠️ 원문에 '<eos>' 라는 **문자열**이 있으면 경계 토큰과 **구분되지 않는다**.")
        print(f"         → 그 문서는 과분할된다. 아래 [C] 에서 실제로 그런지 본다.")
    else:
        print(f"      ✅ '<eos>' 문자열은 eos({eos}) 로 인코딩되지 않는다 → 과분할 위험 없음.")

    # --- B. 경계 개수·길이 분포 sanity ---
    lens = np.array([e - s for s, e in docs])
    print(f"\n  [B] 경계 판정 sanity")
    print(f"      eos 출현 {int((arr == eos).sum()):,}회 → 문서 {len(docs):,}개 "
          f"(차이는 길이<2 문서 제외 + 마지막 꼬리)")
    print(f"      문서 길이: 중위 {int(np.median(lens)):,}  평균 {lens.mean():,.0f}  "
          f"최대 {lens.max():,}  최소 {lens.min():,}")
    tiny = int((lens < 16).sum())
    print(f"      16토큰 미만 문서 {tiny:,}개 ({tiny/len(lens):.1%})"
          + ("  ⚠️ 과분할 신호 — [A] 와 함께 볼 것" if tiny / len(lens) > 0.05 else "  ✅ 정상 범위"))

    # --- C·D. 상위 문서를 실제로 디코딩해서 본다 ---
    k = a.inspect
    print(f"\n  [C][D] 기여 상위 {k}개 문서 — 앞/뒤 {a.inspect_chars}자 + 문자통계")
    print(f"        ★볼 것: ① 중간에 새 문서의 서두가 또 나오는가(=경계 누락)")
    print(f"                ② 표·나열·코드·비한국어인가(=길이가 아니라 내용이 원인)")
    for i, (s, e, n, tot) in enumerate(rows_by_contrib[:k], 1):
        ids = arr[s:e].tolist()
        txt = tok.decode(ids)
        print("\n  " + "-" * 74)
        print(f"  [{i}] 토큰 {n:,}  문서손실 {tot/n:.4f}  (val 의 {n/len(arr):.1%})")
        print(f"      {_charstats(txt)}")
        head = txt[:a.inspect_chars].replace("\n", " / ")
        tail = txt[-a.inspect_chars:].replace("\n", " / ")
        print(f"      앞: {head}")
        print(f"      뒤: {tail}")

    print("\n  " + "=" * 74)
    print("  판정 가이드")
    print("""    A 가 ⚠️ 이고 B 에 16토큰 미만이 많다  → **과분할**. prepare() 에서 특수토큰 문자열
                                              이스케이프 후 0.6 재실행
    C 에서 문서 중간에 서두가 또 나온다     → **경계 누락**. 원인이 길이가 아니다
    D 가 표·나열·코드·비한국어를 가리킨다   → 길이 상한이 아니라 **내용 필터**가 필요
    전부 정상                               → 0.6 확정. P037 단계2(재생성)로""")


def main():
    ap = argparse.ArgumentParser(description="P028 단계0.6 문서단위 val 손실 분해")
    ap.add_argument("--data", default="ko-edu-en")
    ap.add_argument("--doc-filter", action="store_true",

                    help="(P037 단계2) 필터된 캐시({data}_{N}_filtered)를 읽는다. 결과 023: 이 플래그가 없어 검증이 옛 캐시를 쟀다")
    ap.add_argument("--tokens", default="300M",
                    help="**데이터 풀** 크기. 필터 캐시는 600M 인데 학습 예산은 300M 일 수 있다")
    ap.add_argument("--ckpt-tokens", default=None,
                    help="**학습 토큰 예산**(체크포인트 이름에 쓰이는 값). 미지정이면 --tokens 를 쓴다. "
                         "★결과 023 §7: 이 둘이 한 플래그라서 600M 풀을 지정하니 "
                         "존재하지 않는 m100_..._600M_dense.pt 를 찾다 죽었다")
    ap.add_argument("--arch", default="dense", choices=["dense", "tied"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--preset", default="m100")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--max-docs", type=int, default=0, help="0=전부")
    ap.add_argument("--top", type=int, default=15, help="기여 상위 몇 개를 인쇄할지")
    ap.add_argument("--inspect", type=int, default=0,
                    help="(P037 단계1) 상위 N개 문서를 **디코딩해서** 앞/뒤 텍스트와 문자통계를 "
                         "출력하고, eos 경계 판정의 신뢰도를 검사한다. 0=끔")
    ap.add_argument("--inspect-chars", type=int, default=300, help="앞/뒤 몇 자를 볼지")
    a = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from tinylm import paths
    from tinylm.data import prepare, tokenizer_path
    from tinylm.infer.generate import load_model
    from tokenizers import Tokenizer

    n_tok = int(float(a.tokens.rstrip("MmBb")) * (1e9 if a.tokens[-1] in "Bb" else 1e6))
    meta = prepare(a.data, n_tok, doc_filter=a.doc_filter)
    val = np.memmap(Path(meta["dir"]) / "val.bin", dtype=np.uint16, mode="r")
    tok = Tokenizer.from_file(str(tokenizer_path(a.data)))
    eos = tok.token_to_id("<eos>")
    if eos is None:
        eos = 2

    # ★풀 크기(--tokens)와 학습 예산(--ckpt-tokens)은 다른 값이다(결과 023 §7).
    ck_tok = a.ckpt_tokens or a.tokens
    ck = paths.resolve_ckpt(a.preset, a.data, ck_tok, a.tag or a.arch)
    if ck_tok != a.tokens:
        print(f"[ckpt] 풀 {a.tokens} / 학습예산 {ck_tok} 로 분리 지정 -> {ck.name}")
    if not ck.exists():
        print(f"[!] 체크포인트 없음: {ck}")
        return 1
    model, cfg, dev = load_model(arch=a.arch, ckpt_path=str(ck))

    # --- 문서 경계 분해 (<eos> 기준) ---
    arr = np.asarray(val)
    cuts = np.flatnonzero(arr == eos)
    starts = np.concatenate([[0], cuts + 1])
    ends = np.concatenate([cuts + 1, [len(arr)]])
    docs = [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= 2]
    if a.max_docs:
        docs = docs[:a.max_docs]

    print("=" * 88)
    print(f"  P028 단계0.6 — 문서단위 val 손실 : {a.data} / {a.tag or a.arch}")
    print(f"  val {len(arr):,} 토큰, 문서 {len(docs):,}개, eos id={eos}")
    print("=" * 88)

    # --- 문서별 손실(토큰수 가중) ---
    rows = []
    with torch.no_grad():
        for (s, e) in docs:
            ids = arr[s:e]
            tot, ntok = 0.0, 0
            for off in range(0, len(ids) - 1, a.seq):
                chunk = ids[off:off + a.seq + 1]
                if len(chunk) < 2:
                    break
                x = torch.tensor(chunk[:-1].astype(np.int64), device=dev)[None]
                y = torch.tensor(chunk[1:].astype(np.int64), device=dev)[None]
                with torch.autocast(dev if isinstance(dev, str) else dev.type,
                                    dtype=torch.bfloat16, enabled=(str(dev) == "cuda")):
                    lg = model(x)
                l = F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(),
                                    y.reshape(-1), reduction="sum").item()
                tot += l
                ntok += y.numel()
            if ntok:
                rows.append((s, e, ntok, tot))

    N = sum(r[2] for r in rows)
    L = sum(r[3] for r in rows)
    overall = L / N
    print(f"\n  전체 val 손실(토큰 가중) = {overall:.4f}   (총 {N:,} 토큰)")

    # --- 기여도 = 그 문서가 총 손실합에서 차지하는 비율 ---
    rows_by_contrib = sorted(rows, key=lambda r: -r[3])
    print(f"\n  기여 상위 {a.top}개 문서")
    print(f"  {'#':>3} {'토큰수':>9} {'토큰비중':>8} {'문서손실':>9} {'손실기여':>8}")
    print("  " + "-" * 46)
    for i, (s, e, n, t) in enumerate(rows_by_contrib[:a.top], 1):
        print(f"  {i:>3} {n:>9,} {n/N:>7.1%} {t/n:>9.4f} {t/L:>7.1%}")

    # --- ★핵심: 상위 문서를 빼면 val 이 어떻게 되나 ---
    print("\n  " + "=" * 60)
    print("  ★상위 기여 문서를 제외하면 val 손실이 어떻게 되는가")
    print("  " + "=" * 60)
    print(f"  {'제외':>10} {'남은토큰':>11} {'val손실':>9} {'변화':>9}")
    print("  " + "-" * 44)
    print(f"  {'없음':>10} {N:>11,} {overall:>9.4f} {'—':>9}")
    for k in (1, 2, 5, 10):
        if k >= len(rows_by_contrib):
            break
        rest = rows_by_contrib[k:]
        n2 = sum(r[2] for r in rest)
        l2 = sum(r[3] for r in rest) / n2
        print(f"  {'상위 '+str(k)+'개':>10} {n2:>11,} {l2:>9.4f} {l2-overall:>+9.4f}")

    # --- 분포 ---
    per = sorted(r[3] / r[2] for r in rows)
    def q(p):
        return per[min(len(per) - 1, int(len(per) * p))]
    print(f"\n  문서별 손실 분포(비가중): 중위 {q(.5):.3f}  "
          f"25% {q(.25):.3f}  75% {q(.75):.3f}  95% {q(.95):.3f}  최대 {per[-1]:.3f}")

    # --- 판정 ---
    top1 = rows_by_contrib[0]
    rest1 = rows_by_contrib[1:]
    l_wo1 = sum(r[3] for r in rest1) / sum(r[2] for r in rest1) if rest1 else overall
    drop = overall - l_wo1
    print("\n  " + "=" * 60)
    print("  판정")
    print("  " + "=" * 60)
    if drop > 0.5:
        print(f"    → ★**단일 문서가 지배한다**(제외 시 {drop:.3f} nats 하락).")
        print("       이건 말뭉치 품질이 아니라 **필터링 문제**다.")
        print("       고칠 것: 문서 길이 상한 + val 내 단일문서 비중 상한.")
        print("       **무작위 분할(전 데이터셋 영향)까지 갈 필요가 없을 수 있다.**")
    elif drop > 0.15:
        print(f"    → 단일 문서 영향이 **있으나 지배적이지 않다**({drop:.3f}).")
        print("       필터링과 분할을 **둘 다** 고쳐야 한다.")
    else:
        print(f"    → 단일 문서로 설명되지 않는다({drop:.3f}).")
        print("       **전 문서가 고르게 어렵다** = 분포 자체의 문제.")
        print("       → 무작위 분할(P028 부록 A, 전 데이터셋 영향)이 필요하다.")
    # =====================================================================
    # (P037 단계1) --inspect : 경계 판정 자체를 검사한다
    #   단계 0.5 의 문서통계가 **도구 결함**으로 틀렸다(eos 를 빈도로 추정).
    #   그러니 이 도구(0.6)도 검증 없이 믿으면 안 된다. 아래 A~D 가 그 검증이다.
    # =====================================================================
    if a.inspect:
        _inspect(arr, docs, rows_by_contrib, tok, eos, a)

    print("""
  ★한계
    · 문서 경계를 <eos> 로 잡는다. prepare() 가 문서마다 eos 를 붙이므로 타당하지만,
      원문에 eos 문자열이 있었다면 과분할된다.
    · 이 모델은 **이 val 로 평가돼 온 모델**이다. 절대 난이도가 아니라
      "이 모델이 이 문서를 얼마나 못 맞히나"를 잰다.
    · 문서 손실은 청크 경계에서 컨텍스트가 끊긴다(긴 문서에 약간 불리).""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
