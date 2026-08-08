"""데이터 캐시의 **언어 구성**을 잰다 — 특히 val 셋. (2026-08-06 신설, 결과 006 §5.3)

## 왜 필요한가

`prepare()` 의 val 은 **스트림 말미 0.5%** 다(`arr[-n_val:]`). 그리고 소스가 고갈되면
`_stream()` 이 조용히 다른 소스로 채운다(L4). 둘이 만나면 **풀 크기가 val 셋의 언어 구성을
바꾼다** — 실측:

    ko-en 300M 풀  : val 한글 13.5%
    ko-en 600M 풀  : val 한글  5.5%
    ko-en 1200M 풀 : val 한글  0.0%   ← 한국어가 한 글자도 없다

결과 006 §5 의 `p12d` vs `p6d` 비교(+0.0614)가 무효인 이유가 이것이다.
**두 모델을 서로 다른 시험지로 채점했다.**

## 왜 별도 도구인가

`meta.json` 의 `mix_token_frac` 은 **신규 빌드에서만** 기록된다(2026-08-06 이후).
구 캐시(300M·600M)에는 없다. 이 도구는 **재생성 없이** `.bin` 을 직접 디코드해서 잰다.

## 어떻게 재는가

토크나이저 json 의 vocab 을 읽어 **ByteLevel BPE 를 직접 역매핑**한다(`tokenizers` 패키지의
`decode` 를 써도 되지만, 이 경로는 의존성이 적고 토큰별 바이트 길이까지 같은 표에서 얻는다).
그 다음 **한글 코드포인트 비율**을 센다.

⚠️ 한글 비율은 "한국어 토큰 비율" 이 아니다 — 한국어 문서에도 라틴 문자·숫자·구두점이 섞인다.
**절대값이 아니라 캐시 간 비교**로 읽는다. 0.0% vs 5.3% 같은 차이는 그 한계를 넘는다.

사용:
    python scripts/diag_val_lang.py --data ko-en --pools 100M 300M 600M 1200M
    python scripts/diag_val_lang.py --data ko-edu-en --pools 100M 300M
    python scripts/diag_val_lang.py --data ko-en --pools 600M --sample 500000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinylm import paths                                    # noqa: E402
from tinylm.config import VOCAB                             # noqa: E402

# ★이 도구는 **torch 없이도 돌아야 한다** — 모델을 안 쓰는 순수 데이터 진단이고,
#   그래야 학습이 GPU 를 점유한 중에도 다른 셸에서 돌릴 수 있다.
#   `tinylm.data.__init__` 이 `Loader` 를 통해 torch 를 끌어오므로 정본을 먼저 시도하고
#   실패하면 같은 규약으로 폴백한다(규약 단일 소스 = `prepare.tokenizer_path`).
try:
    from tinylm.data.prepare import tokenizer_path          # noqa: E402
except Exception:                                            # torch 부재 등
    def tokenizer_path(name, vocab_size=VOCAB):
        return paths.DATA_CACHE / f"tok-{name}-{vocab_size}.json"


def _bytes_to_unicode():
    """GPT-2/ByteLevel 의 바이트 ↔ 가시문자 매핑(원본과 동일한 상수 집합)."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _load_vocab(data):
    p = tokenizer_path(data)
    if not p.exists():
        raise SystemExit(f"[STOP] 토크나이저가 없다: {p}")
    vocab = json.loads(p.read_text(encoding="utf-8"))["model"]["vocab"]
    u2b = {v: k for k, v in _bytes_to_unicode().items()}
    n = max(vocab.values()) + 1
    id2s = [None] * n
    for s, i in vocab.items():
        id2s[i] = s
    blen = np.zeros(n, dtype=np.int32)
    for i, s in enumerate(id2s):
        if s:
            blen[i] = sum(1 for ch in s if ch in u2b)
    return id2s, u2b, blen


def _decode(ids, id2s, u2b):
    out = bytearray()
    for i in ids:
        s = id2s[i] if i < len(id2s) else None
        if not s:
            continue
        for ch in s:
            b = u2b.get(ch)
            if b is not None:
                out.append(b)
    return out.decode("utf-8", errors="replace")


def _hangul_frac(txt):
    if not txt:
        return 0.0
    h = sum(1 for c in txt
            if "가" <= c <= "힣"        # 완성형 음절
            or "ᄀ" <= c <= "ᇿ"        # 자모
            or "㄰" <= c <= "㆏")       # 호환 자모
    return h / len(txt)


def _tok(s):
    s = str(s).strip().upper()
    if s.endswith("M"):
        return int(float(s[:-1]) * 1e6)
    if s.endswith("B"):
        return int(float(s[:-1]) * 1e9)
    return int(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ko-en")
    ap.add_argument("--pools", nargs="+", required=True,
                    help="풀 크기 목록(예: 100M 300M 600M 1200M). data_cache/{data}_{N} 를 찾는다")
    ap.add_argument("--suffix", default="", help="'_filtered' 등 캐시 접미사")
    ap.add_argument("--sample", type=int, default=200_000,
                    help="구간당 디코드할 토큰 수(기본 20만). 늘리면 정확하지만 느리다")
    a = ap.parse_args()

    id2s, u2b, blen = _load_vocab(a.data)
    W = 96
    print("=" * W)
    print(f"  캐시 언어 구성 진단 — data={a.data}  표본 {a.sample:,} 토큰/구간")
    print("=" * W)
    print("  ★val 은 스트림 **말미 0.5%** 다. 소스가 고갈되면(L4) 말미가 남은 소스로 채워지므로")
    print("    **풀 크기가 val 의 언어 구성을 바꾼다**(결과 006 §5.3). 그러면 풀이 다른 런의")
    print("    val_loss·bpb 비교는 **무효**다 — 시험지가 다르기 때문이다.")
    print("  ⚠️ '한글 비율' 은 문자 기준이지 '한국어 토큰 비율' 이 아니다. **캐시 간 비교로만** 읽는다.")
    print()
    hdr = (f"  {'캐시':<26}{'val토큰':>10}{'train앞':>9}{'train중':>9}{'train90%':>10}"
           f"{'★VAL':>9}{'val B/tok':>11}{'meta B/tok':>12}{'오차':>8}")
    print(hdr)
    print("  " + "-" * (W - 4))

    rows = []
    for pool in a.pools:
        n = _tok(pool)
        d = paths.DATA_CACHE / f"{a.data}_{n}{a.suffix}"
        if not (d / "val.bin").exists():
            print(f"  {d.name:<26}  캐시 없음 — 건너뜀")
            continue
        v = np.memmap(d / "val.bin", dtype=np.uint16, mode="r")
        t = np.memmap(d / "train.bin", dtype=np.uint16, mode="r")
        S = min(a.sample, len(v))

        def seg(arr, start_frac):
            i0 = int(len(arr) * start_frac)
            return _decode(arr[i0:i0 + S].tolist(), id2s, u2b)

        val_h = _hangul_frac(_decode(v[:S].tolist(), id2s, u2b))
        tr0 = _hangul_frac(seg(t, 0.0))
        tr5 = _hangul_frac(seg(t, 0.5))
        tr9 = _hangul_frac(seg(t, 0.9))

        vb = float(blen[np.asarray(v)].sum()) / max(len(v), 1)
        meta = {}
        mp = d / "meta.json"
        if mp.exists():
            try:
                meta = json.loads(mp.read_text())
            except Exception:
                pass
        mb = meta.get("bytes_per_token")
        err = (100 * (vb - mb) / mb) if mb else float("nan")
        rows.append((d.name, len(v), tr0, tr5, tr9, val_h, vb, mb, err, meta))
        print(f"  {d.name:<26}{len(v)/1e6:9.1f}M{tr0:9.1%}{tr5:9.1%}{tr9:10.1%}"
              f"{val_h:9.1%}{vb:11.4f}"
              + (f"{mb:12.4f}{err:+7.1f}%" if mb else f"{'—':>12}{'—':>8}"))

    if not rows:
        raise SystemExit("[STOP] 읽은 캐시가 없다.")

    print()
    print("=" * W)
    print("  ★판정")
    print("=" * W)
    hs = [r[5] for r in rows]
    spread = max(hs) - min(hs)
    print(f"  val 한글 비율 범위 : {min(hs):.1%} ~ {max(hs):.1%}   (폭 {spread*100:.1f}%p)")
    if spread > 0.02:
        print("  → 🚫 **풀이 다른 런끼리 val_loss·bpb 를 비교하면 안 된다.** 시험지가 다르다.")
        print("     공통 val 을 고정하기 전에는 P007 계열 풀 비교를 재개하지 않는다(P037 단계5).")
    else:
        print("  → ✅ val 구성은 풀 크기에 걸쳐 안정적이다.")

    errs = [abs(r[8]) for r in rows if r[7]]
    if errs:
        print(f"  bpb 오차(스트림 B/tok 을 val 에 적용) 최대 {max(errs):.1f}% "
              f"= bpb 로 약 {max(errs)/100*1.25:.4f}  (분해능 0.008)")
        if max(errs) / 100 * 1.25 > 0.008:
            print("  → ⚠️ **분해능을 넘는다.** `bytes_per_token_val`(2026-08-06 신설)로 보정할 것.")

    print()
    print("  ★L4 고갈 기록(있는 캐시만 — 2026-08-06 이후 빌드)")
    any_ex = False
    for name, *_rest in rows:
        meta = _rest[-1]
        ex = meta.get("mix_exhausted")
        mf = meta.get("mix_token_frac")
        if mf:
            srcs = meta.get("mix_sources", [])
            frac = "  ".join(f"{s.split('/')[-1][:22]} {f:.1%}" for s, f in zip(srcs, mf))
            print(f"    {name:<26} 토큰비율  {frac}")
            any_ex = True
        if ex:
            print(f"    {name:<26} ★고갈  {ex}")
    if not any_ex:
        print("    (없음 — 전부 계측 이전 캐시다. 재생성 없이는 믹스를 알 수 없고,")
        print("     위의 한글 비율 열이 그 대용이다.)")


if __name__ == "__main__":
    main()
