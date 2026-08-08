#!/usr/bin/env python3
"""P047 단계0 — **표준 코퍼스 `ko-en` 에 스팸이 있는가.** 학습 0 · GPU 0.

## 왜 이걸 재는가

결과 011 §2 는 우리 이력에서 **가장 큰 단일 품질 이득**을 기록했다:
`ko-edu-en` 에 스팸 필터를 걸고 재학습하니 **공통 원문 bpb −0.1058(= −0.296 nats, 24.7σ)**.
`ko-edu-en` 격차의 **63%가 스팸**이었다.

그런데 **우리 모든 기준선은 `ko-en` 으로 학습돼 있다.** 그리고 `ko-en` 의 스팸률은
**한 번도 측정한 적이 없다.** 결과 018 이 `ko-en` 을 "건강한 대조군" 으로 쓴 것은
**표본 몇 개의 줄바꿈·고유율**을 본 것이고 **전량 스캔이 아니다.**

> **★그래서 이 진단의 결과는 두 갈래로 크게 갈린다.**
> · 스팸률 ≈ 0 → `ko-en` 은 깨끗하다. **기준선이 안전하다는 것을 처음 확인**한다
> · 스팸률 > 0 → **전 기준선이 오염된 데이터 위에 있다.** REVIEW1·REVIEW2 의 모든
>   서열이 그 위에서 매겨졌고, 재학습 없이는 고칠 수 없다
>
> **어느 쪽이든 REVIEW2 보다 먼저 알아야 한다.** 그게 이 진단이 최우선인 이유다.

## 어떻게 재는가 — 필터와 **같은 함수를 호출**한다

`prepare.spam_signature()` 를 **import 해서 쓴다**(베끼지 않는다 — 함정 14).
그래서 이 진단이 세는 것은 정확히 **`--doc-filter` 가 버릴 문서**다.

토큰화하지 않는다. 서명은 **문자 수·줄바꿈·줄 고유율**만 보므로 토크나이저가 불필요하고,
그래서 `prepare` 의 12분 토큰화 없이 스트리밍 속도로 돈다.

## ★서명이 `ko-en` 에 안 맞을 가능성 — 그래서 근접 사례도 본다

서명은 `ko-edu-en` 의 스팸(자격증 덤프 + 키워드 삽입)을 보고 만들었다.
`ko-en` 의 오염이 **다른 형태**라면 서명이 0 을 돌려주고 우리는 "깨끗하다" 로 오독한다.
→ **`min_chars` 이상인 문서 전부에 대해 `nl_frac`·`uniq` 분포를 찍는다.**
서명에 안 걸려도 **경계에 몰려 있으면** 그게 신호다.

사용:
    python scripts/diag_spam_rate.py --data ko-en --max-bytes 1.5G
    python scripts/diag_spam_rate.py --data ko-edu-en --max-bytes 400M   # 양성 대조군
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tinylm.data.prepare import DATASETS, _stream, spam_signature   # noqa: E402  ★호출한다

W = 96


def parse_size(s):
    s = str(s).strip().upper()
    mul = {"K": 10**3, "M": 10**6, "G": 10**9}
    return int(float(s[:-1]) * mul[s[-1]]) if s and s[-1] in mul else int(float(s))


def profile(text):
    """서명이 보는 세 값. `spam_signature` 와 **같은 식**이어야 한다."""
    lines = text.split("\n")
    nl_frac = (len(lines) - 1) / max(len(text), 1)
    uniq = len(set(lines)) / max(len(lines), 1)
    return len(text), nl_frac, uniq


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def wilson_upper(k, n, z=1.96):
    """k/n 의 95% 상한. **k=0 일 때가 이 진단의 핵심** —
    '스팸을 못 찾았다' 를 '스팸률은 최대 이만큼' 으로 바꿔 준다."""
    if n <= 0:
        return 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return min(1.0, (c + r) / d)


def main():
    ap = argparse.ArgumentParser(description="P047 — 코퍼스 스팸률 (학습 0, GPU 0)")
    ap.add_argument("--data", default="ko-en", choices=list(DATASETS))
    ap.add_argument("--max-bytes", default="1.5G",
                    help="이만큼의 원문 바이트를 훑고 멈춘다. 300M 토큰 풀 ~= 1.31G bytes")
    ap.add_argument("--max-docs", type=int, default=0, help="0=제한 없음")
    ap.add_argument("--min-chars", type=int, default=50_000,
                    help="서명의 기본 임계(prepare 기본값과 같게 유지할 것)")
    ap.add_argument("--report-every", default="200M", help="이 바이트마다 중간 보고(중단해도 값이 남는다)")
    ap.add_argument("--show-large", type=int, default=15, help="가장 큰 문서 몇 개를 표로 보일지")
    a = ap.parse_args()

    max_bytes = parse_size(a.max_bytes)
    every = parse_size(a.report_every)
    srcs = DATASETS[a.data]
    n_src = len(srcs)

    print("=" * W)
    print("  P047 단계0 — 코퍼스 스팸률. **필터가 실제로 버릴 문서를 센다**")
    print("=" * W)
    print(f"  data={a.data}  소스 {n_src}개  예산 {max_bytes/1e9:.2f} GB  min_chars={a.min_chars:,}")
    for i, (hf, cf, sp, key, r) in enumerate(srcs):
        print(f"    [{i}] {hf}  config={cf}  ratio={r}")
    print(f"  ★`prepare.spam_signature()` 를 **호출**한다 — 필터와 동일 판정 보장")
    print(f"  ★토큰화하지 않는다(서명은 문자만 본다) → 스트리밍 속도")
    print()

    tot_docs = tot_chars = tot_bytes = 0
    spam_docs = spam_chars = 0
    big_docs = big_chars = 0                     # min_chars 이상 (걸렸든 안 걸렸든)
    big_nl = big_uq = 0                          # 대형 중 조건 **하나만** 맞는 것(누적 카운터로 센다.
    #                                              near 리스트는 잘릴 수 있어 거기서 세면 과소가 된다)
    per_src = [dict(docs=0, chars=0, spam=0, spam_chars=0, big=0) for _ in range(n_src)]
    near = []                                    # min_chars 이상 전부의 (chars, nl, uniq, spam, src)
    exhausted = {}
    t0 = time.time()
    next_report = every

    def _on_exhausted(i):
        exhausted[i] = tot_bytes
        print(f"  [mix] ★소스 고갈 — {srcs[i][0]} ({tot_bytes/1e9:.2f} GB 지점). "
              f"이 뒤로는 남은 소스만 나온다(L4)")

    def partial(final=False):
        el = (time.time() - t0) / 60
        hdr = "최종" if final else "중간"
        print()
        print(f"  ── {hdr} ({tot_bytes/1e9:.3f} GB / {tot_docs:,} 문서 / {el:.1f}분)")
        print(f"     스팸 문서   {spam_docs:,} / {tot_docs:,} = {pct(spam_docs, tot_docs):.5f}%")
        print(f"     스팸 문자   {spam_chars:,} / {tot_chars:,} = {pct(spam_chars, tot_chars):.5f}%")
        up = wilson_upper(spam_docs, tot_docs)
        print(f"     문서율 95% 상한 {up*100:.5f}%"
              + ("   ← ★0건이면 이 값이 결론이다" if spam_docs == 0 else ""))
        print(f"     대형 문서(>={a.min_chars:,}자)  {big_docs:,} = {pct(big_docs, tot_docs):.4f}%"
              f"  / 문자 {pct(big_chars, tot_chars):.3f}%")
        sys.stdout.flush()

    try:
        for text, src_i in _stream(a.data, exhausted_cb=_on_exhausted):
            n, nl, uq = profile(text)
            is_spam = spam_signature(text, a.min_chars)
            tot_docs += 1
            tot_chars += n
            tot_bytes += len(text.encode("utf-8"))
            s = per_src[src_i]
            s["docs"] += 1; s["chars"] += n
            if is_spam:
                spam_docs += 1; spam_chars += n
                s["spam"] += 1; s["spam_chars"] += n
            if n >= a.min_chars:
                big_docs += 1; big_chars += n
                s["big"] += 1
                if nl < 1e-4:
                    big_nl += 1
                if uq >= 0.999:
                    big_uq += 1
                near.append((n, nl, uq, is_spam, src_i))
                if len(near) > 20_000:                    # 상수 메모리 유지(큰 것만 남긴다)
                    near.sort(reverse=True); near = near[:5_000]

            if tot_bytes >= next_report:
                partial(); next_report += every
            if tot_bytes >= max_bytes:
                break
            if a.max_docs and tot_docs >= a.max_docs:
                break
    except KeyboardInterrupt:
        print("\n  ⚠️ 중단됨 — 아래 값은 여기까지의 부분 집계다(비율은 이미 수렴했을 수 있다)")

    partial(final=True)

    print()
    print("=" * W)
    print("  소스별")
    print("=" * W)
    print(f"     {'소스':<40}{'문서':>12}{'문자%':>9}{'대형':>8}{'스팸':>8}{'스팸문자%':>11}")
    for i, (hf, cf, sp, key, r) in enumerate(srcs):
        s = per_src[i]
        print(f"     {hf[:38]:<40}{s['docs']:>12,}{pct(s['chars'], tot_chars):>8.1f}%"
              f"{s['big']:>8,}{s['spam']:>8,}{pct(s['spam_chars'], max(s['chars'],1)):>10.5f}%")
    if exhausted:
        print(f"     ⚠️ 고갈된 소스 {sorted(exhausted)} — 그 지점 이후 믹스가 바뀌었다(L4)")

    print()
    print("=" * W)
    print(f"  ★대형 문서 프로파일 (>={a.min_chars:,}자) — **서명에 안 걸린 것도 본다**")
    print("=" * W)
    if not near:
        print(f"     대형 문서가 **0개**다. 서명은 대형에만 발화하므로 스팸률 0 은 자동이다.")
        print(f"     ⚠️ 이 경우 '깨끗하다' 가 아니라 **'이 서명으로는 볼 수 없다'** 가 맞다.")
        print(f"        --min-chars 를 10000 정도로 낮춰 한 번 더 돌려 볼 것.")
    else:
        near.sort(reverse=True)
        print(f"     {'문자':>12}{'줄바꿈율':>11}{'줄고유율':>11}{'스팸판정':>10}{'소스':>6}")
        for n, nl, uq, sp, si in near[:a.show_large]:
            print(f"     {n:>12,}{nl:>11.5f}{uq:>11.5f}{('YES' if sp else '.'):>10}{si:>6}")
        print()
        print(f"     대형 **{big_docs:,}개** 중 — 줄바꿈율 < 1e-4 인 것 **{big_nl:,}개**"
              f" / 줄고유율 >= 0.999 인 것 **{big_uq:,}개** / **둘 다(=스팸) {spam_docs:,}개**")
        print(f"     (위 표는 큰 순서 상위만. 세 카운터는 **전량 누적**이라 잘리지 않는다)")
        print(f"     ★두 조건 중 **하나만** 맞는 문서가 많으면 서명이 `ko-en` 오염 형태와")
        print(f"       어긋난다는 뜻이다 — 그때는 서명을 손보는 게 아니라 **결과 018 처럼")
        print(f"       실제 손실 기여를 먼저 재야 한다**(`diag_val_docs.py`).")

    print()
    print("=" * W)
    print("  ★어떻게 읽는가 — 게이트")
    print("=" * W)
    print(f"  스팸 문서 0건 AND 대형 문서가 존재")
    print(f"    → ✅ **`ko-en` 은 이 서명으로 깨끗하다.** 전 기준선이 안전하다는 첫 확인.")
    print(f"      P047 종결. `--doc-filter` 는 `ko-edu-en` 전용 도구로 남는다.")
    print(f"  스팸 문자 비중이 0.01% 이상")
    print(f"    → ⚠️ **필터 재학습(P047 단계1)이 필요하다.** 그리고 그것은 기준선 재작성이다.")
    print(f"      ★비중이 작아도 무시하지 말 것 — 결과 023 §8 에서 스팸은 **토큰의 소수인데")
    print(f"        val 손실의 53%** 였다. 게이트는 '토큰 비중' 이 아니라 '존재 여부' 다.")
    print(f"  스팸 0 인데 대형 문서도 0")
    print(f"    → ⚠️ **판정 불가.** 서명이 발화할 대상이 없었다. --min-chars 를 낮춘다.")
    print()
    print("  ★한계")
    print(f"   · 서명은 `ko-edu-en` 의 스팸을 보고 만든 것이다(결과 018). **다른 형태의 오염은")
    print(f"     이 진단이 못 본다** — 위 대형 문서 프로파일이 그 사각지대를 좁히려는 장치다")
    print(f"   · `_stream` 의 rng 는 seed 0 고정이라 **`prepare` 와 같은 순서**를 훑는다.")
    print(f"     즉 여기서 센 것은 실제 캐시에 들어간 문서 집합과 (예산 안에서) 같다")
    print(f"   · 예산을 다 못 채우고 멈추면 비율은 유효하나 **고갈 지점 이후 믹스는 다르다**")
    print(f"   · 문자 수 != 토큰 수. 토큰 비중은 bytes_per_token ~4.37 로 어림한다")


if __name__ == "__main__":
    main()
