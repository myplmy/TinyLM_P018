#!/usr/bin/env python3
"""실행 배치파일(.bat) 린터 — `run-batch` 스킬의 기계 검증부.

검사 항목
  E(에러) 1  비ASCII 바이트          cmd 코드페이지에서 명령이 깨진다
  E       2  chcp 사용                파이썬 출력이 깨진다
  E       3  escape 안 된 <>|%        리다이렉션·변수확장으로 오작동
  E       4  train 명령에 --tag 없음  정본 dense/tied 로그·ckpt 를 조용히 덮어쓴다
  W(경고) 5  train 명령에 --accum 없음  기본 8 이라 절반만 학습된다
  W       6  학습 런 전부가 goto ERROR  한 런 실패로 후속 런이 죽는다(결과 007)
  W       7  --no-ckpt + KD + 태그없는 dense 교사  VRAM 15.7GB+ (OOM 위험)
  W       8  커널 + --compile 병용     코드가 SystemExit 로 막지만 배치가 무의미해진다
  W       9  pause / exit /b 누락      더블클릭 실행 시 창이 닫혀 로그를 잃는다
  I(정보) 10 꼬리 판정 안내(echo) 없음  로그를 받아도 무엇을 읽어야 할지 모른다
  E      11  줄끝이 CRLF 가 아님       `.gitattributes` 가 `*.bat text eol=crlf` 로 선언하는데
                                       작업트리가 LF 면 git 이 매번 재작성·경고한다(CRLF 재발 원인).
                                       또 cmd.exe 는 LF-only 배치에서 `goto`/라벨이 드물게 어긋난다.
                                       → `--fix` 로 일괄 교정

★한계: 문법·규약만 본다. **실험설계가 옳은지는 판단하지 않는다**(그건 exp-preflight).

사용법
  python scripts/lint_bat.py                    # 루트의 모든 .bat
  python scripts/lint_bat.py run100m_P026.bat   # 특정 파일
  python scripts/lint_bat.py --fix              # 줄끝(CRLF)만 자동 교정
  종료코드 = 에러 개수 (0 이면 통과)

★왜 CRLF 검사가 여기 있나: 이 저장소에서 줄끝 문제가 **두 번 재발**했다. 원인은 사람이 아니라
  구조였다 — `.gitattributes` 는 .bat 을 CRLF 로 선언하는데, 이 워크플로의 모든 도구
  (샌드박스 편집·파이썬 재작성)가 LF 를 쓴다. 그래서 배치를 고칠 때마다 어긋난다.
  **배치를 고치면 반드시 이 린터를 돌리므로**, 검사를 여기 두면 자동으로 잡힌다.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def lint(path: Path):
    raw = path.read_bytes()
    txt = raw.decode("utf-8", errors="replace")
    lines = txt.splitlines()
    err, warn, info = [], [], []

    # 11. 줄끝(CRLF) — .gitattributes 선언과 작업트리가 일치해야 한다
    if b"\r\n" not in raw and raw.strip():
        err.append("줄끝이 LF 다 — .bat 은 CRLF 여야 한다(`.gitattributes`). "
                   "`python scripts/lint_bat.py --fix` 로 교정")
    elif raw.replace(b"\r\n", b"").count(b"\n"):
        err.append("CRLF 와 LF 가 섞여 있다 — `--fix` 로 교정")

    # 1. 비ASCII
    bad_lines = [(i + 1, ln) for i, ln in enumerate(lines) if any(ord(c) > 127 for c in ln)]
    for ln, s in bad_lines:
        err.append(f"L{ln} 비ASCII 문자: {s.strip()[:60]}")

    # 2. chcp
    for i, ln in enumerate(lines):
        if re.search(r"\bchcp\b", ln, re.I):
            err.append(f"L{i+1} chcp 사용 금지: {ln.strip()[:60]}")

    # 3. escape 안 된 <>|%
    for i, ln in enumerate(lines):
        for ch in "<>|%":
            if ch in ln and ("^" + ch) not in ln:
                err.append(f"L{i+1} escape 안 된 '{ch}' (echo 면 ^{ch}, 주석이면 다른 표현으로): "
                           f"{ln.strip()[:55]}")
                break

    # train 명령 수집
    trains = [(i + 1, ln) for i, ln in enumerate(lines)
              if re.search(r"run100m\.py\s+train", ln)]
    # 실험 변형을 뜻하는 플래그. 이게 하나라도 있으면 정본이 아니므로 --tag 가 필수다.
    VARIANT = ["--mlp-group", "--sparse34", "--kd", "--init-from", "--no-ckpt", "--sched",
               "--anneal-end", "--decay-frac", "--ema", "--lora-rank", "--mlp-film",
               "--ternary-kernel", "--pool-tokens", "--center-weights"]
    for ln, s in trains:
        if "--tag" not in s:
            variants = [v for v in VARIANT if v in s]
            if variants:
                err.append(f"L{ln} 변형 런({', '.join(variants)})에 --tag 없음 → "
                           f"정본을 덮어쓴다: {s.strip()[:50]}")
            else:
                info.append(f"L{ln} --tag 없음. 플래그가 없으니 그 (preset,data,tokens) 의 "
                            f"**정본 기준선**을 쓰는 것으로 봅니다. 의도한 것이면 OK, "
                            f"아니면 --tag 를 붙이세요")
        if "--accum" not in s:
            warn.append(f"L{ln} train 에 --accum 없음 → 기본 8 로 절반만 학습: {s.strip()[:55]}")
        if "--no-ckpt" in s and "--kd" in s and "--kd-teacher-tag" not in s:
            warn.append(f"L{ln} --no-ckpt + KD + dense 교사 = VRAM 15.7GB+ (OOM 위험). "
                        f"긴 런이면 --no-ckpt 를 빼거나 압축교사를 쓰세요")
        if re.search(r"--ternary-kernel", s) and "--compile" in s:
            warn.append(f"L{ln} 커널 + --compile 병용 (코드가 SystemExit 로 중단시킨다)")

    # 6. errorlevel 정책
    if trains:
        gotos = 0
        for ln, _ in trains:
            nxt = lines[ln] if ln < len(lines) else ""     # train 다음 줄
            if re.search(r"if errorlevel 1 goto ERROR", nxt):
                gotos += 1
        if gotos == len(trains) and len(trains) > 1:
            warn.append(f"학습 런 {len(trains)}개 전부 'goto ERROR' → 한 런 실패로 후속이 전부 죽는다. "
                        f"독립 런은 'if errorlevel 1 echo [WARN] ... - continuing' 로 (결과 007)")

    # 9. pause / exit
    if "pause" not in txt:
        warn.append("pause 없음 → 더블클릭 실행 시 창이 닫혀 로그를 잃는다")

    # ★9b. 무방비 pause — 2026-08-07 야간큐가 여기서 밤새 멈췄다.
    #   `run_night_queue.bat` 은 `TL_NOPAUSE=1` 을 깔고 자식을 `call` 하는데,
    #   자식의 pause 가 그 변수를 안 보면 **키 입력을 기다리며 큐가 정지**한다.
    #   P045·P046 에는 가드가 있었고 P014C 에만 없었다 — 3개 중 1개를 빠뜨린 것이
    #   7시간짜리 무인 실행을 1시간으로 만들었다. 사람이 기억하는 대신 린터가 본다.
    #   **모든 경로의 pause 가 대상이다**(:BADROOT 같은 오류 경로도 큐를 멈춘다).
    for i, ln in enumerate(lines):
        s = ln.strip()
        if re.fullmatch(r"(?i)pause", s):
            err.append(f"L{i+1} 무방비 `pause` → 야간큐(`TL_NOPAUSE=1`)가 여기서 멈춘다. "
                       f"`if not defined TL_NOPAUSE pause` 로 바꾸세요")
        elif re.search(r"(?i)\bpause\b", s) and "TL_NOPAUSE" not in s and not s.startswith("REM"):
            warn.append(f"L{i+1} pause 가 TL_NOPAUSE 가드 없이 쓰였다: {s[:55]}")
    if "exit /b" not in txt and "goto ERROR" in txt:
        warn.append("goto ERROR 는 있는데 'exit /b 0' 이 없음 → 정상 종료도 ERROR 블록으로 흘러간다")

    # ★9c. runlog.py 에 `--note` 와 실행 명령을 **한 줄에** 준 것.
    #   2026-08-07 `run_P047_stage0_spam_rate.bat` 이 이 형태였고, runlog 가 note 만 쓰고
    #   **명령을 조용히 버렸다.** 종료코드 0 이라 `if errorlevel 1` 도 안 걸려
    #   **두 단계가 "성공" 으로 끝났는데 로그는 84바이트뿐**이었다.
    #   runlog.py 쪽에도 거절을 넣었지만(이중 방어), 배치 작성 시점에 잡는 게 더 값싸다.
    for i, ln in enumerate(lines):
        if ln.strip().upper().startswith("REM"):
            continue          # 주석은 이 규칙의 대상이 아니다(사고를 서술한 문장이 걸렸다)
        if "runlog.py" in ln and "--note" in ln and re.search(r"\s--\s", ln):
            err.append(f"L{i+1} runlog.py 에 --note 와 실행 명령을 함께 줬다 → "
                       f"**명령이 실행되지 않는다**(조용한 무동작). 두 줄로 나누세요: "
                       f"{ln.strip()[:45]}")

    # ★9d. runlog `--name` 에 실험계획 번호가 없다.
    #   야간큐 v2 가 `--name mC_g16` 로 돌아 로그가 `log_20260807_mC_g16.txt` 가 됐고
    #   사용자가 `029_log_..._P045_mC_g16.txt` 로 손수 고쳐야 했다(2026-08-07 지적).
    #   `scripts/batch/` 의 도구는 `!TL_LOGNAME!` 로 **호출자가** 이름을 주므로 대상 아님.
    if path.name.startswith("run_"):
        for i, ln in enumerate(lines):
            if ln.strip().upper().startswith("REM"):
                continue
            m = re.search(r"--name\s+([^\s\"]+)", ln)
            if m and not re.match(r"^(P\d{3}|!)", m.group(1)):
                warn.append(f"L{i+1} runlog --name '{m.group(1)}' 에 계획번호(P0NN)가 없다 → "
                            f"로그 파일명만 보고 어느 실험인지 알 수 없다")

    # 10. 꼬리 판정 안내
    tail = "\n".join(lines[-25:]).lower()
    if not any(k in tail for k in ("record", "read", "decision", "compare", "gate", "verdict")):
        info.append("꼬리에 판정/읽는 순서 안내 echo 가 없다 → 로그를 받아도 해석 기준이 없다")

    return err, warn, info


def fix_eol(path: Path) -> bool:
    """줄끝을 CRLF 로 정규화. 내용은 건드리지 않는다."""
    raw = path.read_bytes()
    out = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    if out != raw:
        path.write_bytes(out)
        return True
    return False


def main():
    args = [a for a in sys.argv[1:] if a != "--fix"]
    do_fix = "--fix" in sys.argv[1:]
    files = [Path(a) if Path(a).is_absolute() else ROOT / a for a in args] or \
            sorted(list(ROOT.glob('*.bat')) + list((ROOT / 'scripts' / 'batch').glob('*.bat')))
    if do_fix:
        n = sum(fix_eol(f) for f in files if f.exists())
        print(f"[--fix] 줄끝을 CRLF 로 교정: {n}개 (내용 변경 없음)")
    if not files:
        print("점검할 .bat 이 없습니다.")
        return 0
    total_err = 0
    for f in files:
        if not f.exists():
            print(f"[!] {f} 없음"); continue
        err, warn, info = lint(f)
        total_err += len(err)
        mark = "FAIL" if err else ("WARN" if warn else "OK")
        print(f"\n=== {f.name}  [{mark}]  (E{len(err)} W{len(warn)} I{len(info)})")
        for m in err:
            print(f"  [E] {m}")
        for m in warn:
            print(f"  [W] {m}")
        for m in info:
            print(f"  [i] {m}")
    print(f"\n총 에러 {total_err}건")
    print("주의: 이 린터는 문법·규약만 봅니다. 실험설계 검증은 exp-preflight 스킬로.")
    return total_err


if __name__ == "__main__":
    sys.exit(main())
