"""P039 — 어휘 활용도·미학습 토큰 진단. (2026-08-06 신설)

근거: **"Fishing for Magikarp: Automatically Detecting Under-trained Tokens in LLMs"**
(Land & Bartolo, 2024) — arXiv:2405.05417.
토크나이저가 만든 토큰 중 상당수가 **학습 데이터에 거의 안 나온다**. 그 토큰들의 임베딩은
초기값 근처에 머물고, 프롬프트에 들어오면 **예측 불가능한 행동**을 만든다.

## 왜 이 저장소에서 중요한가

1. **메모리** — 결과 016 §10.5: int8 이후 `mC` 상주 86.9MB 중 **33.0MB(38%)가 fp32 임베딩**이다.
   그리고 그 임베딩은 **한 번도 양자화된 적이 없다**(`quantize_embedding` 은 회계 플래그일 뿐).
   그런데 int8 화는 **출력 헤드와 묶여 있어** 간단하지 않다(`transformer.py` P034 단계5 주석).
   **어휘를 줄이는 것이 같은 33MB 를 양자화 없이 줄이는 유일한 경로**다.
   32,768 → 24,000 이면 임베딩이 **26.7% 줄어든다.**

2. **데이터 품질** — 결과 018: 토크나이저 학습 표본 20만 문서에 **SEO 스팸이 있었다**
   (자격증 덤프 판매 문구, 단어 중간 키워드 삽입). **제품코드가 어휘를 차지했을 수 있다.**
   그리고 결과 023 §8 의 필터는 **그 스팸을 학습에서 뺐다** → 필터 후 데이터로 학습하면
   **미학습 토큰이 새로 생긴다.**

3. **토크나이저 효율** — 결과 011 §1.3: `ko-en` 토크나이저가 영문을 `ko-edu-en` 보다
   **8.1% 더 잘 압축**한다(4.366 vs 4.038 B/tok). 어휘 배분이 다르다는 뜻이고 여기서 보인다.

## 무엇을 재는가

| 지표 | 의미 |
|---|---|
| **빈도 0 토큰** | 학습 스트림에 **한 번도** 안 나온 토큰. 임베딩이 초기값 그대로다 |
| 하위 빈도 분포 | 어휘의 몇 %가 실질적으로 안 쓰이는가 |
| **임베딩 노름**(체크포인트 필요) | 미학습 토큰은 초기 std 0.02 근처에 머문다 → **빈도와 독립적인 두 번째 증거** |
| 언어별 배분 | 한글 포함 토큰 / 라틴 전용 / 숫자·기호 각각 몇 개인가 |

★ **빈도와 노름을 함께 본다.** 하나만 보면 "자주 안 나오지만 중요한 토큰"(고유명사)과
"쓰레기 토큰"을 구분할 수 없다.

사용:
    python scripts/diag_vocab.py --data ko-en --tokens 600M
    python scripts/diag_vocab.py --data ko-en --tokens 600M --ckpt runs/ckpt/m100_ko-en_300M_p6d.pt
    python scripts/diag_vocab.py --data ko-edu-en --tokens 300M --top-unused 40
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

try:
    from tinylm.data.prepare import tokenizer_path          # noqa: E402
except Exception:                                            # torch 부재 시(이 도구는 torch 불필요)
    def tokenizer_path(name, vocab_size=VOCAB):
        return paths.DATA_CACHE / f"tok-{name}-{vocab_size}.json"


def _bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _tok(s):
    s = str(s).strip().upper()
    if s.endswith("M"):
        return int(float(s[:-1]) * 1e6)
    if s.endswith("B"):
        return int(float(s[:-1]) * 1e9)
    return int(s)


def _classify(text):
    """토큰 문자열의 대략적 종류. 절대 분류가 아니라 배분을 보기 위한 것이다."""
    if not text:
        return "empty"
    has_ko = any("가" <= c <= "힣" or "ᄀ" <= c <= "ᇿ" for c in text)
    has_lat = any(("a" <= c <= "z") or ("A" <= c <= "Z") for c in text)
    has_dig = any(c.isdigit() for c in text)
    if has_ko:
        return "korean"
    if has_lat:
        return "latin"
    if has_dig:
        return "digit"
    if any(ord(c) > 0x2E80 for c in text):
        return "cjk-other"
    return "symbol"


def main():
    ap = argparse.ArgumentParser(description="P039 어휘 활용도·미학습 토큰 진단")
    ap.add_argument("--data", default="ko-en")
    ap.add_argument("--tokens", default="600M", help="검사할 캐시 크기(디렉터리 이름)")
    ap.add_argument("--suffix", default="", help="'_filtered' 등")
    ap.add_argument("--ckpt", default=None,
                    help="체크포인트 경로. 주면 임베딩 노름도 함께 본다(torch 필요)")
    ap.add_argument("--top-unused", type=int, default=25, help="표본으로 보여줄 미사용 토큰 수")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="스캔할 최대 토큰 수(0=전체). 큰 캐시에서 빠르게 보고 싶을 때")
    a = ap.parse_args()

    # ── 토크나이저 vocab ────────────────────────────────────────────────
    tp = tokenizer_path(a.data)
    if not tp.exists():
        raise SystemExit(f"[STOP] 토크나이저가 없다: {tp}")
    vocab = json.loads(tp.read_text(encoding="utf-8"))["model"]["vocab"]
    V = max(vocab.values()) + 1
    id2s = [""] * V
    for s, i in vocab.items():
        id2s[i] = s
    u2b = {v: k for k, v in _bytes_to_unicode().items()}

    def decode1(s):
        return bytes(u2b[c] for c in s if c in u2b).decode("utf-8", errors="replace")

    # ── 캐시 스캔 ──────────────────────────────────────────────────────
    n = _tok(a.tokens)
    d = paths.DATA_CACHE / f"{a.data}_{n}{a.suffix}"
    if not (d / "train.bin").exists():
        raise SystemExit(f"[STOP] 캐시가 없다: {d}")
    arr = np.memmap(d / "train.bin", dtype=np.uint16, mode="r")
    scanned = len(arr) if a.max_tokens <= 0 else min(len(arr), a.max_tokens)
    freq = np.bincount(np.asarray(arr[:scanned]), minlength=V).astype(np.int64)

    W = 92
    print("=" * W)
    print(f"  P039 — 어휘 활용도 진단   data={a.data}  캐시={d.name}")
    print(f"  어휘 {V:,}   스캔 토큰 {scanned/1e6:.1f}M" + ("" if a.max_tokens <= 0 else "  (부분)"))
    print("=" * W)
    print("  근거: Fishing for Magikarp, arXiv:2405.05417 (미학습 토큰 자동 탐지)")
    print("  ★빈도와 임베딩 노름을 **함께** 본다 — 하나만 보면 '희귀하지만 중요한 토큰'과")
    print("    '쓰레기 토큰'을 구분할 수 없다.")
    print()

    unused = int((freq == 0).sum())
    print(f"  ★빈도 0 토큰      {unused:,} / {V:,}  ({unused/V:.2%})")
    for thr in (1, 10, 100, 1000):
        c = int((freq < thr).sum())
        print(f"   빈도 < {thr:<5,}      {c:,} ({c/V:.2%})")

    # ── 언어별 배분 ────────────────────────────────────────────────────
    print()
    print("  종류별 배분 (토큰 문자열 기준 — 대략적 분류다)")
    print(f"  {'종류':<12}{'개수':>9}{'비중':>9}{'빈도0':>9}{'빈도0 비율':>12}{'총 사용량':>14}")
    print("  " + "-" * (W - 4))
    kinds = {}
    for i in range(V):
        kinds.setdefault(_classify(decode1(id2s[i])), []).append(i)
    for k in sorted(kinds, key=lambda x: -len(kinds[x])):
        idx = np.array(kinds[k])
        z = int((freq[idx] == 0).sum())
        print(f"  {k:<12}{len(idx):>9,}{len(idx)/V:>9.1%}{z:>9,}{z/max(len(idx),1):>12.1%}"
              f"{int(freq[idx].sum()):>14,}")

    # ── 미사용 토큰 표본 ───────────────────────────────────────────────
    if unused:
        print()
        print(f"  미사용 토큰 표본 (최대 {a.top_unused}개)")
        shown = 0
        for i in np.nonzero(freq == 0)[0]:
            t = decode1(id2s[i]).replace("\n", "\\n")
            print(f"    id {int(i):>6}  {t!r}")
            shown += 1
            if shown >= a.top_unused:
                break

    # ── 임베딩 노름 (선택) ─────────────────────────────────────────────
    if a.ckpt:
        print()
        print("=" * W)
        print("  ★임베딩 노름 — 빈도와 독립적인 두 번째 증거")
        print("=" * W)
        try:
            import torch
            st = torch.load(a.ckpt, map_location="cpu")
            sd = st.get("model", st)
            key = next((k for k in sd if k.endswith("emb.weight")), None)
            if key is None:
                print("  [skip] 체크포인트에서 emb.weight 를 못 찾았다")
            else:
                w = sd[key].float()
                nrm = w.norm(dim=1).numpy()
                used = freq[:len(nrm)] > 0
                zero = ~used
                print(f"  임베딩 shape {tuple(w.shape)}   초기 std=0.02 기준 기대 노름 "
                      f"= 0.02*sqrt({w.shape[1]}) = {0.02*np.sqrt(w.shape[1]):.4f}")
                print(f"  {'집합':<16}{'개수':>9}{'노름 중위':>12}{'노름 평균':>12}{'최소':>10}{'최대':>10}")
                print("  " + "-" * (W - 4))
                for name, m in (("사용됨(freq>0)", used), ("미사용(freq=0)", zero)):
                    if m.sum() == 0:
                        continue
                    v = nrm[m]
                    print(f"  {name:<16}{int(m.sum()):>9,}{np.median(v):>12.4f}"
                          f"{v.mean():>12.4f}{v.min():>10.4f}{v.max():>10.4f}")
                if zero.sum() and used.sum():
                    r = np.median(nrm[zero]) / max(np.median(nrm[used]), 1e-9)
                    print()
                    print(f"  ★미사용/사용 노름 비 = {r:.3f}")
                    if r < 0.8:
                        print("  → 미사용 토큰이 **초기값 근처에 머물러 있다**. 빈도 증거와 일치한다.")
                    else:
                        print("  → 노름 차이가 작다. 빈도 0 이어도 gradient 가 흘렀다는 뜻일 수 있다")
                        print("    (출력 헤드가 임베딩과 묶여 있어 **모든 토큰이 softmax 분모에** 들어간다).")
                        print("    ★이 저장소는 헤드가 tie 돼 있으므로 이쪽이 정상일 수 있다 — 단정 금지.")
        except Exception as e:
            print(f"  [skip] 임베딩 분석 실패: {type(e).__name__}: {e}")

    # ── 판정 안내 ──────────────────────────────────────────────────────
    print()
    print("=" * W)
    print("  ★어떻게 읽는가")
    print("=" * W)
    print("  빈도 0 이 1% 미만")
    print("    → 어휘가 데이터에 잘 맞는다. 어휘 축소로는 메모리를 못 번다.")
    print("  빈도 0 이 5% 이상, 또는 빈도<100 이 20% 이상")
    print("    → **어휘 축소가 실효 레버다.** 32,768 -> 24,000 이면 임베딩 26.7% 감소")
    print("      = 상주 33.0MB 중 약 8.7MB. int8 화(헤드 tie 때문에 어렵다)의 대안이다.")
    print("    ⚠️ 단 어휘를 바꾸면 **토크나이저 재학습 + 전 모델 재학습**이다. 값이 크다.")
    print("  특정 종류(숫자·기호)에 빈도 0 이 몰려 있다")
    print("    → 토크나이저 학습 표본 문제. 결과 018 의 SEO 스팸이 후보다.")
    print()
    print("  ★한계")
    print("   · 빈도는 **이 캐시** 기준이다. 필터 캐시(`--suffix _filtered`)로 다시 재면 값이 다르다")
    print("     — 결과 023 §8 의 필터가 **미학습 토큰을 새로 만들었을 수 있다.** 둘 다 재서 비교할 것.")
    print("   · 노름 지표는 **헤드가 임베딩과 tie 된 구조**에서 약해진다(모든 토큰이 분모에 들어간다).")
    print("   · 이 도구는 어휘를 바꾸지 않는다. **측정만** 한다.")


if __name__ == "__main__":
    main()
