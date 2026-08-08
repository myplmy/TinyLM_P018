"""배치가 쓰는 CLI 플래그가 **실제로 구현돼 있는지** 정적으로 검사한다. (2026-08-06 신설)

## 왜 필요한가

**같은 사고가 두 번 났다.**

| 날짜 | 배치 | 무슨 일 |
|---|---|---|
| 2026-07-31 | `run_P031_repeat.bat` | `--infer-repeat` 미구현 상태로 최상위에 놓였고 사용자가 실행 → `NOTIMPL` |
| 2026-08-06 | `run_P022_stage1_fp8.bat` | `--fp8` 미구현 상태로 최상위에 놓였고 사용자가 실행 → 가드 정지 |

두 번 다 **가드가 있어서 GPU 는 안 태웠다.** 그래도 **사용자 시간을 썼다.**
그리고 가드는 작성자가 "이건 미구현이다" 를 **기억했을 때만** 붙는다 — 기억에 의존하는 장치다.

`CLAUDE.md` 의 규범("최상위 실험 배치는 지금 돌아갈 때만 만든다")은 옳지만 **강제되지 않았다.**
이 스크립트가 그걸 강제한다. `lint_bat.py` 가 문법을 보듯 이건 **실행 가능성**을 본다.

## 무엇을 검사하나

배치에서 `python run100m.py ...` / `python scripts\X.py ...` 호출을 찾아 **`--flag` 를 전부 뽑고**,
대상 파일의 `add_argument("--flag"` 에 있는지 대조한다. 없으면 **E1(미구현)**.

한계(정직하게):
- **이름이 있는지만** 본다. 플래그가 **동작하는지**는 모른다.
  실제로 `--doc-filter` 는 `train` 파서에 **있었는데 trainer 로 전달되지 않았다**(2026-08-06 발견).
  그런 결함은 이 도구가 못 잡는다 → **스모크가 그 다음 그물**이다.
- `%VAR%` 치환·동적 조립은 못 따라간다(이 저장소는 `%` 금지라 문제되지 않는다).

사용:
    python scripts/check_batch_flags.py             # 최상위 run_*.bat 전부
    python scripts/check_batch_flags.py run_P022_stage1_fp8.bat
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 배치에 **명시적 미구현 가드**(`:NOTIMPL` 라벨 또는 "NOT IMPLEMENTED" 문구)가 있으면
# "작성자가 알고 있다" 로 보고 경고로 낮춘다. 느슨한 패턴을 쓰면 "Nothing is pending" 같은
# 정상 문장에 걸려 진짜 오류를 놓친다(초판이 그랬다).
ACK = re.compile(r"^:NOTIMPL\b|NOT IMPLEMENTED", re.I | re.M)

FLAG = re.compile(r"(?<![\w-])--[a-z0-9][a-z0-9-]*")
CALL = re.compile(r"^\s*python\s+(?:scripts\\runlog\.py[^\n]*?--\s+python\s+)?"
                  r"([\w\\/.-]+\.py)\s+(.*)$", re.I | re.M)


def declared_flags(pyfile: Path):
    if not pyfile.exists():
        return None
    src = pyfile.read_text(encoding="utf-8", errors="replace")
    return set(re.findall(r"add_argument\(\s*[\"'](--[a-z0-9-]+)[\"']", src))


def check(bat: Path):
    text = bat.read_text(encoding="utf-8", errors="replace")
    acked = bool(ACK.search(text))
    errs, warns = [], []
    cache = {}
    for m in CALL.finditer(text):
        target, rest = m.group(1), m.group(2)
        # 배치는 Windows 경로(`scripts\x.py`)를 쓴다 — 어느 OS 에서 검사하든 통하게 정규화한다.
        norm = target.replace("\\", "/")
        # run100m.py 는 tinylm/cli.py 로 위임한다(단일 파서).
        py = ROOT / ("tinylm/cli.py" if Path(norm).name == "run100m.py" else norm)
        if py not in cache:
            cache[py] = declared_flags(py)
        decl = cache[py]
        if decl is None:
            warns.append(f"대상 파일 없음: {target}")
            continue
        # ★따옴표 안은 **데이터지 인자가 아니다.** `runlog.py --note "... --mlp-group ..."`
        #   의 안내문에 든 플래그를 실제 호출로 오인하면 오탐이 난다(초판이 그랬다).
        rest = re.sub(r'"[^"]*"', " ", rest)
        for f in FLAG.findall(rest):
            if f in ("--",):
                continue
            if f not in decl:
                (warns if acked else errs).append(
                    f"미구현 플래그 {f}  ({Path(target).name} 에 add_argument 없음)")
    return errs, warns, acked


def main():
    args = sys.argv[1:]
    bats = ([ROOT / a for a in args] if args
            else sorted(p for p in ROOT.glob("run*.bat") if "-done" not in p.name))
    n_e = n_w = 0
    for b in bats:
        if not b.exists():
            print(f"=== {b.name}  [SKIP] 파일 없음")
            continue
        errs, warns, acked = check(b)
        errs = sorted(set(errs)); warns = sorted(set(warns))
        tag = "FAIL" if errs else ("WARN" if warns else "OK")
        print(f"=== {b.name}  [{tag}]  (E{len(errs)} W{len(warns)})"
              + ("  [미구현 인지 표기 있음]" if acked else ""))
        for e in errs:
            print(f"  [E] {e}")
        for w in warns:
            print(f"  [W] {w}")
        n_e += len(errs); n_w += len(warns)

    print(f"\n총 에러 {n_e}건 / 경고 {n_w}건")
    if n_e:
        print("★최상위 실험 배치는 **지금 돌아갈 때만** 만든다(CLAUDE.md).")
        print("  구현이 선결이면 배치를 만들지 말고 **계획서에만 설계를 남긴다.**")
        print("  불가피하면 헤더 첫 줄에 선결 조건을 적어라 — 그러면 이 검사가 경고로 낮춘다.")
    print("주의: **이름이 있는지만** 본다. 파서에 있는데 실제로 전달 안 되는 결함은 못 잡는다")
    print("      (2026-08-06 `--doc-filter` 가 그랬다). 그건 `run_smoke_check.bat` 의 몫이다.")
    return 1 if n_e else 0


if __name__ == "__main__":
    sys.exit(main())
