#!/usr/bin/env python3
"""P028 단계 0 — 토큰 캐시 건전성 진단 (GPU 불필요, 학습 없음).

왜: 결과 009 에서 `ko-edu-en` dense 가 train 3.03 / val 6.17 = **3.14 nats 괴리**로 끝났다.
    같은 코드의 `ko-en` 은 그런 괴리가 없다. 300M 토큰 · 132M 삼진 파라미터에서 정상 과적합이 아니다.
    유력 가설 2개를 **학습 없이** 가른다:
      (H1) train 쪽 문서 중복(near-duplicate) → 반복 노출로 train 손실만 내려간다
      (H2) val 분할의 분포 이동 → `prepare()` 가 스트림 **말미 0.5%** 를 val 로 쓰는데,
           각 소스 내부는 샤드 순서대로 소비되므로 샤드가 정렬돼 있으면 말미가 편향된다

검사 항목
  1. 문서 수·길이 분포 (구간별)
  2. 문서 중복률 — 앞 N토큰 해시 기준 (H1)
  3. train <-> val 문서 중복 (역방향 누설)
  4. 구간별 유니그램 분포의 JS divergence — val 이 유별난지 (H2)
  5. 구간별 한글 토큰 비율 — 소스 혼합이 구간마다 다른지 (H2)

★한계: 이 스크립트는 **캐시의 통계**만 본다. "정상"이 나와도 데이터가 학습에 좋다는 뜻은 아니고,
  "이상"이 나와도 그것이 3.14 nats 괴리의 원인이라고 증명하는 것은 아니다. 가설의 우선순위를
  좁히는 도구다. 인과는 재학습으로만 확인된다.

사용법
  python scripts/diag_cache.py                          # 캐시 목록
  python scripts/diag_cache.py --data ko-en --tokens 300M
  python scripts/diag_cache.py --data ko-edu-en --tokens 300M
  python scripts/diag_cache.py --all                    # 존재하는 실데이터 캐시 전부(비교용)
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data_cache"
HEAD = 256          # 문서 앞 몇 토큰으로 중복 판정할지
CHUNKS = 10         # train 을 몇 구간으로 나눠 분포를 볼지


def _tok(s):
    s = str(s).strip().upper()
    if s.endswith("M"):
        return int(float(s[:-1]) * 10 ** 6)
    if s.endswith("K"):
        return int(float(s[:-1]) * 10 ** 3)
    return int(s)


def list_caches():
    out = []
    for d in sorted(CACHE.glob("*_*")):
        if (d / "meta.json").exists() and (d / "train.bin").exists():
            try:
                m = json.loads((d / "meta.json").read_text())
            except Exception:
                continue
            if m.get("data") == "synthetic":
                continue
            m["dir"] = d
            out.append(m)
    return out


def load(d: Path):
    tr = np.memmap(d / "train.bin", dtype=np.uint16, mode="r")
    va = np.memmap(d / "val.bin", dtype=np.uint16, mode="r")
    return tr, va


def find_eos(arr, cap=None):
    """eos 토큰 id 추정: prepare() 는 `tok.token_to_id('<eos>') or 2` 를 쓴다.
    가장 빈번한 상위 후보 중 문서 경계로 그럴듯한 것을 고르되, 실패하면 2 를 쓴다."""
    sample = np.asarray(arr[: cap or min(len(arr), 5_000_000)])
    cnt = np.bincount(sample, minlength=8)
    # 0..7 중 가장 많이 나오는 특수토큰 후보
    cand = int(np.argmax(cnt[:8]))
    return cand if cnt[cand] > 0 else 2


def docs_from(arr, eos, limit_docs=None):
    """eos 위치로 문서 경계를 나눠 (start, end) 리스트를 준다."""
    a = np.asarray(arr)
    pos = np.flatnonzero(a == eos)
    if len(pos) == 0:
        return []
    starts = np.concatenate(([0], pos[:-1] + 1))
    ends = pos
    if limit_docs:
        starts, ends = starts[:limit_docs], ends[:limit_docs]
    return list(zip(starts.tolist(), ends.tolist()))


def hash_docs(arr, spans, head=HEAD):
    a = np.asarray(arr)
    out = []
    for s, e in spans:
        seg = a[s:min(e, s + head)]
        out.append(hashlib.blake2b(seg.tobytes(), digest_size=8).digest())
    return out


def js_div(p, q, eps=1e-12):
    p = p / max(p.sum(), eps)
    q = q / max(q.sum(), eps)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / np.maximum(b[mask], eps))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def hangul_token_mask(data_name, vocab=32768):
    """토크나이저 vocab 을 **한 번** 디코드해 '한글 바이트를 포함한 토큰' 마스크를 만든다.
    문서를 전부 디코드하지 않고 bincount 만으로 구간별 한글 비율을 낼 수 있다."""
    tp = CACHE / f"tok-{data_name}-{vocab}.json"
    if not tp.exists():
        return None
    try:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(str(tp))
    except Exception as e:
        print(f"  [i] 토크나이저 로드 실패({e}) → 한글 비율 생략")
        return None
    mask = np.zeros(vocab, dtype=bool)
    for i in range(vocab):
        try:
            s = tk.decode([i])
        except Exception:
            continue
        if any("가" <= c <= "힣" or "ㄱ" <= c <= "ㆎ" for c in s):
            mask[i] = True
    return mask


def diagnose(meta):
    d = Path(meta["dir"])
    name = meta["data"]
    print("=" * 78)
    print(f"  캐시: {d.name}   데이터={name}  총 {meta.get('tokens',0)/1e6:.1f}M 토큰"
          f"  (train {meta.get('train',0)/1e6:.1f}M / val {meta.get('val',0)/1e3:.0f}K)")
    print("=" * 78)
    tr, va = load(d)
    eos = find_eos(tr)
    print(f"  eos 토큰 id 추정 = {eos}")

    # ---- 1. 문서 수·길이
    tspans = docs_from(tr, eos)
    vspans = docs_from(va, eos)
    if not tspans:
        print("  [!] 문서 경계를 찾지 못했습니다(eos 추정 실패?). 이후 검사 생략.")
        return
    tlen = np.array([e - s for s, e in tspans])
    print(f"\n[1] 문서 수  train {len(tspans):,}  val {len(vspans):,}")
    print(f"    길이(토큰) 평균 {tlen.mean():.0f}  중위 {np.median(tlen):.0f}  "
          f"p90 {np.percentile(tlen,90):.0f}  최대 {tlen.max():,}")

    # ---- 2. 중복률
    th = hash_docs(tr, tspans)
    uniq = len(set(th))
    dup_rate = 1 - uniq / len(th)
    from collections import Counter
    c = Counter(th)
    top = c.most_common(3)
    print(f"\n[2] 문서 중복률(앞 {HEAD}토큰 해시) = {dup_rate*100:.2f}%"
          f"   ({len(th):,}개 중 유일 {uniq:,}개)")
    print(f"    최다 반복 문서 등장횟수: {[n for _, n in top]}")
    if dup_rate > 0.05:
        print(f"    [!] 중복 5% 초과 → H1(train 중복) 지지. 반복 노출로 train 손실만 내려갈 수 있다.")
    else:
        print(f"    [OK] 중복 낮음 → H1 약화.")

    # ---- 3. train <-> val 누설
    vh = hash_docs(va, vspans)
    tset = set(th)
    leak = sum(1 for h in vh if h in tset)
    print(f"\n[3] val 문서 중 train 에도 있는 것: {leak}/{len(vh)}"
          f"  ({leak/max(len(vh),1)*100:.1f}%)")
    if leak:
        print("    [!] 누설. val 이 과소평가된다(반대 방향 오염).")
    else:
        print("    [OK] 누설 없음.")

    # ---- 4. 구간별 유니그램 분포 vs val
    V = int(meta.get("vocab", 32768))
    n = len(tr) // CHUNKS
    chunk_hist = []
    for i in range(CHUNKS):
        seg = np.asarray(tr[i * n:(i + 1) * n])
        chunk_hist.append(np.bincount(seg, minlength=V).astype(np.float64))
    vhist = np.bincount(np.asarray(va), minlength=V).astype(np.float64)
    ref = np.sum(chunk_hist, axis=0)          # train 전체
    print(f"\n[4] 유니그램 JS divergence (train 전체 기준, 낮을수록 유사)")
    for i, h in enumerate(chunk_hist):
        print(f"    train 구간 {i+1:2}/10 : {js_div(h, ref):.5f}")
    jv = js_div(vhist, ref)
    jmax = max(js_div(h, ref) for h in chunk_hist)
    print(f"    ** val           : {jv:.5f}   (train 구간 최대 {jmax:.5f})")
    if jv > jmax * 1.5:
        print("    [!] val 이 어떤 train 구간보다도 유별나다 → H2(val 분포 이동) 지지.")
    elif jv > jmax:
        print("    [~] val 이 train 구간 최대보다 크다. 경계선.")
    else:
        print("    [OK] val 이 train 구간 변동폭 안에 있다 → H2 약화.")

    # ---- 5. 구간별 한글 토큰 비율
    hm = hangul_token_mask(name, V)
    if hm is not None:
        print(f"\n[5] 한글 토큰 비율 (소스 혼합의 구간별 균일성)")
        for i, h in enumerate(chunk_hist):
            r = h[hm].sum() / max(h.sum(), 1)
            print(f"    train 구간 {i+1:2}/10 : {r*100:5.1f}%")
        rv = vhist[hm].sum() / max(vhist.sum(), 1)
        rates = [h[hm].sum() / max(h.sum(), 1) for h in chunk_hist]
        print(f"    ** val           : {rv*100:5.1f}%   "
              f"(train 범위 {min(rates)*100:.1f}~{max(rates)*100:.1f}%)")
        if not (min(rates) <= rv <= max(rates)):
            print("    [!] val 의 한글 비율이 train 범위를 벗어난다 → 소스 혼합 편향.")
        else:
            print("    [OK] val 이 train 범위 안.")

    print("\n" + "-" * 78)
    print("  판정 요약: [!] 가 붙은 항목이 원인 가설을 지지한다. 어느 것도 안 붙으면")
    print("  3.14 nats 괴리는 캐시 통계로 설명되지 않으므로 다른 원인(학습 설정·풀 크기 등)을 본다.")
    print("  ★어느 쪽이든 인과는 재학습으로만 확인된다.")
    print("-" * 78 + "\n")


def main():
    ap = argparse.ArgumentParser(description="P028 단계0 캐시 진단(GPU 불필요)")
    ap.add_argument("--data")
    ap.add_argument("--doc-filter", action="store_true",

                    help="(P037 단계2) 필터된 캐시({data}_{N}_filtered)를 읽는다. 결과 023: 이 플래그가 없어 검증이 옛 캐시를 쟀다")
    ap.add_argument("--tokens")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    caches = list_caches()
    if not caches:
        print(f"[!] {CACHE} 에 실데이터 캐시가 없습니다.")
        return 1
    if a.all:
        targets = caches
    elif a.data:
        # ★P037 단계2: 필터 캐시 별도 디렉터리(결과 023)
        _suf = "_filtered" if a.doc_filter else ""
        want = f"{a.data}_{_tok(a.tokens)}{_suf}" if a.tokens else None
        targets = [m for m in caches
                   if m["data"] == a.data and (want is None or Path(m["dir"]).name == want)]
        if not targets:
            print(f"[!] 해당 캐시 없음. 존재하는 캐시:")
            for m in caches:
                print(f"    {Path(m['dir']).name}")
            return 1
    else:
        print("존재하는 캐시 (하나를 --data/--tokens 로 지정, 또는 --all):")
        for m in caches:
            print(f"  {Path(m['dir']).name:32} {m.get('tokens',0)/1e6:>7.1f}M 토큰")
        return 0

    for m in targets:
        diagnose(m)
    if len(targets) > 1:
        print("★ 비교 요령: ko-en 이 정상 기준선이다. ko-edu-en 에서만 [!] 가 붙는 항목이 원인 후보다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
