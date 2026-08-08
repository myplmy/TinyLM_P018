#!/usr/bin/env python3
"""로그의 `ms/step` 이 **스필(시스템 메모리 대체) 징후**를 보이는지 검사한다. 학습 0 · GPU 0.

## 왜 필요한가 — 스필은 조용하다

2026-08-08 에 두 개의 `--no-ckpt` 런이 정반대로 나왔다(결과 037 section 5):

    mC_tinf_nockpt   reserved 15.37 GiB   -20.5% (건강)
    mC_ct250_nockpt  reserved 14.95 GiB   +26.4% (스필)

**VRAM 이 더 적은 쪽이 스필했다.** 그리고 스필한 런은 **종료코드 0, skip 0, 손실 정상**이었다 —
로그 어디에도 스필했다는 말이 없다. 순간 ms/step 을 손으로 복원해야만 보였다.
그걸 매번 손으로 하지 않기 위한 도구다.

## 어떻게 재는가

트레이너가 찍는 `ms/step` 은 **누적 평균**이다. 두 인쇄 지점 사이의 **구간 순간값**은

    inst = ((n1+1)*c1 - (n0+1)*c0) / (n1 - n0)

로 복원된다(c = 누적평균, n = 스텝 번호). warmup(기본 step 100 미만)을 버리고
**p90/p10 - 1** 을 변동폭으로 본다.

★**max/min 이 아니라 p90/p10 을 쓰는 이유**: 우리 기본 `--eval-every 100` 이라 몇 구간에
eval 이 끼고, 2289스텝 런은 eval 이 22회다. max/min 으로 보면 **정상 런도 28~48% 가 나와**
알려진 스필 런(67%)과 안 갈린다. p90/p10 은 갈린다(아래 교정 표).

## ★교정 — 알려진 정상/스필 런으로 임계를 정했다 (2026-08-08)

    태그                p90/p10   실제
    mC_tinf_on            6.5%    정상 (결과 034 section 9)
    mC_tinf_nockpt        7.4%    정상 (결과 034 section 10.3, -20.5%)
    mC_tinf_off/off2    8.4~9.0%  정상
    mC_g256a             12.5%    정상 (2289스텝, eval 22회)
    mC_ct250             14.0%    정상 (ckpt ON)
    mC_ct                15.4%    정상 (2289스텝)
    mC_ct250_nockpt      48.5%    ★스필 (결과 037 section 4.1)

→ 임계 **0.25 = 의심**, **0.15 = 경계**. 정상 최대(15.4%)와 스필(48.5%) 사이가 넓다.
⚠️ 표본이 8개다. **더 모이면 임계를 다시 정한다.**

## 한계 - 반드시 읽을 것

★**이건 징후 탐지이지 증명이 아니다.** 변동을 만드는 다른 원인이 최소 셋 있다:
  - **eval** 스파이크(p90/p10 로 상당 부분 완화되지만 전부는 아니다)
  - **열 스로틀링** - 소비자 카드에서 흔하다
  - **다른 프로세스**가 GPU 를 건드림
★**반대로 변동이 작다고 스필이 없다는 증명도 아니다** - 균일하게 느려질 수 있다.
**"의심" 까지만 말하고 판정은 사람이 한다.** 확정하려면 같은 조건을 유휴 상태에서 재실행한다.

사용:
    python scripts/check_spill.py test_result/037_log_*.txt
    python scripts/check_spill.py --warn 0.25 --from-step 100 test_result/034_log_20260807_P042_stage1.txt
종료코드 = 의심 런 개수(0 이면 통과)
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

# `  step   190/250  ce 4.19 ... 3767 ms/step`
STEP_RE = re.compile(r"step\s+(\d+)/(\d+).*?(\d+)\s*ms/step")
TAG_RE = re.compile(r"--tag\s+(\S+)")


def blocks(text):
    """로그를 런 단위로 자른다. `--tag X` 부터 다음 `[runlog] 종료코드` 까지.

    ⚠️ `--tag` 는 한 런에 **두 번** 나온다(`[runlog] cmd=` 줄과 `[cmd]` 줄). 같은 지점을
    가리키므로 (태그, 첫 스텝, 표본수, 마지막 누적평균) 으로 **중복을 제거**한다.
    """
    out, seen = [], set()
    for m in TAG_RE.finditer(text):
        end = text.find("[runlog] 종료코드", m.end())
        seg = text[m.end():end if end > 0 else len(text)]
        pts = [(int(a), int(b), float(c)) for a, b, c in STEP_RE.findall(seg)]
        if len(pts) < 4:
            continue
        key = (m.group(1), pts[0][0], len(pts), pts[-1][2])
        if key in seen:
            continue
        seen.add(key)
        out.append((m.group(1), pts))
    return out


def instants(pts):
    """(step, _, 누적평균) 목록 -> [(구간시작, 순간 ms/step)]."""
    res = []
    for k in range(1, len(pts)):
        n0, _, c0 = pts[k - 1]
        n1, _, c1 = pts[k]
        if n1 > n0:
            res.append((n0 + 1, ((n1 + 1) * c1 - (n0 + 1) * c0) / (n1 - n0)))
    return res


def pct(v, q):
    v = sorted(v)
    i = (len(v) - 1) * q
    lo = int(i)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def main():
    ap = argparse.ArgumentParser(description="ms/step 변동으로 스필 징후를 본다(징후 탐지 전용)")
    ap.add_argument("logs", nargs="+", help="로그 파일(글롭 가능)")
    ap.add_argument("--from-step", type=int, default=100,
                    help="이 스텝 이후만 본다(compile·warmup 제외). 기본 100")
    ap.add_argument("--warn", type=float, default=0.25, help="p90/p10-1 이 이 값을 넘으면 의심. 기본 0.25")
    ap.add_argument("--edge", type=float, default=0.15, help="경계 임계. 기본 0.15")
    a = ap.parse_args()

    paths = []
    for g in a.logs:
        paths.extend(sorted(glob.glob(g)) or [g])

    W = 100
    print("=" * W)
    print("  ms/step 변동폭으로 본 스필 징후  (★징후 탐지이지 증명이 아니다)")
    print("=" * W)
    print(f"  step {a.from_step} 이후 / 지표 = p90/p10 - 1 / 의심 {a.warn:.0%} · 경계 {a.edge:.0%}")
    print("  교정 근거: 정상 런 6.5~15.4% vs 알려진 스필 런 48.5% (docstring 표)")
    print()
    print(f"  {'태그':<22}{'표본':>5}{'중앙값':>10}{'p90/p10':>10}{'max/min':>10}  판정")
    print("  " + "-" * (W - 4))

    n_warn = 0
    for p in paths:
        f = Path(p)
        if not f.exists():
            print(f"  [없음] {f.name}")
            continue
        bs = blocks(f.read_text(encoding="utf-8", errors="replace"))
        if not bs:
            print(f"  [ms/step 없음] {f.name}")
            continue
        print(f"  · {f.name}")
        for tag, pts in bs:
            v = [x for s, x in instants(pts) if s >= a.from_step]
            if len(v) < 4:
                print(f"  {tag:<22}{len(v):>5}{'':>10}{'':>10}{'':>10}  [표본 부족]")
                continue
            spread = pct(v, 0.9) / pct(v, 0.1) - 1.0
            if spread > a.warn:
                verdict = "⚠ 스필 의심 — 이 런의 ms/step 을 속도 비교에 쓰지 말 것"
                n_warn += 1
            elif spread > a.edge:
                verdict = "경계 — 긴 런이면 eval 스파이크일 수 있다"
            else:
                verdict = "정상"
            print(f"  {tag:<22}{len(v):>5}{pct(v,0.5):>10.0f}{spread:>9.1%}"
                  f"{max(v)/min(v)-1:>9.1%}  {verdict}")

    print("  " + "-" * (W - 4))
    print()
    print("  ★읽는 법")
    print("    정상       순간값이 두 값 사이를 규칙적으로 오간다(KD 스텝 1/4 과 비KD 스텝).")
    print("               그 교대 자체가 6~9% 의 고유 변동을 만든다 — 그게 바닥값이다.")
    print("    스필 의심   순간값이 불규칙하게 튄다. 결과 037 의 mC_ct250_nockpt 가 그것이고,")
    print("               VRAM 이 더 적은데도 26% 느렸다.")
    print()
    print("  다음에 할 것: 의심이 뜨면 ①`[vram] peak reserved` 를 본다 ②NVIDIA 제어판의")
    print("    'CUDA - 시스템 메모리 대체' 설정을 확인한다 ③유휴 상태에서 재실행해 재현한다.")
    print("    정책 권고는 docs/methods/09_training_memory.md section 4.")
    print("=" * W)
    return n_warn


if __name__ == "__main__":
    sys.exit(main())
