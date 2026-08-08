#!/usr/bin/env python3
"""실행 로그 티(tee) — **콘솔에 실시간 출력하면서 `test_result/` 에 즉시 기록**한다.

★왜 필요한가 (사용자 요구, 2026-07-31)
  지금까지 실험 로그는 **사용자가 콘솔에서 복사해 붙여넣는** 방식이었다. 그래서
    · 중간에 크래시하면 그때까지의 출력이 **스크롤 버퍼에만** 남고
    · 장시간 런(P033 은 18~28h)에서는 버퍼를 넘겨 **앞부분이 통째로 사라지며**
    · 붙여넣기 과정에서 잘림·누락이 생긴다.
  **장기 실험에서 로그 유실은 그 시간 전체를 날리는 것과 같다.**

★유실 방지 설계 (이 파일의 존재 이유)
  1. **줄 단위 즉시 기록 + flush** — 버퍼에 쌓아두지 않는다.
  2. **주기적 `os.fsync()`**(기본 2초) — OS 캐시까지 디스크로 내린다.
     BSOD·정전·강제종료에서도 마지막 몇 초를 제외한 전부가 남는다.
  3. **`finally` 로 꼬리말 보장** — 예외·Ctrl+C·자식 프로세스 이상종료 어디서든
     종료코드와 소요시간이 기록된다.
  4. **append 모드가 기본** — 한 배치의 여러 단계가 **한 파일에 누적**된다.
     3단계에서 죽어도 1~2단계 로그는 그대로 남는다.
  5. **인코딩 사고 차단** — 자식에게 `PYTHONIOENCODING=utf-8` 을 강제해 파이프를
     UTF-8 로 고정하고, 파일은 UTF-8, 콘솔은 파이썬이 콘솔 API 로 직접 쓴다.
     (파이프일 때 파이썬 기본 인코딩은 locale=cp949 라 한글이 깨질 수 있다)

★2026-07-31 수정 — **콘솔과 로그파일이 달랐다**(사용자 보고)
  이 래퍼는 **감싼 파이썬 명령의 출력만** 기록한다. 그래서 배치의 `echo` 로 찍히는
  **단계 제목·판정 안내·WHAT TO RECORD 블록이 로그파일에 전혀 남지 않았다.**
  콘솔에는 있고 파일에는 없으니 두 개가 다르게 보이는 것이 당연했다 — 사용자가 계속
  콘솔을 복사해 붙여 온 이유가 이것이다. 세 가지를 고쳤다.

  1. **`--note`** — 명령을 실행하지 않고 **텍스트만** 콘솔과 로그에 동시에 남긴다.
     배치의 단계 제목·판정 안내를 이걸로 흘려보내면 **로그파일만 읽어도 맥락이 산다.**
       python scripts/runlog.py --name P035 --note "[1/2] step anneal at 0.60"
  2. **줄끝을 os.linesep 으로** — 종전에는 `newline=""` 라 파일이 **LF 전용**이었다.
     Windows 메모장·일부 편집기에서 **한 줄로 붙어 보였다**(콘솔과 또 다르게 보이는 원인).
  3. **기록 후 검증** — 닫은 뒤 파일 존재·크기를 다시 확인하고 콘솔에 찍는다.
     0바이트거나 사라졌으면 **경고한다**. "로그 파일: ..." 만 찍고 실제로는 없는
     상황을 조용히 넘기지 않는다.

사용법 — `--` 뒤가 실행할 명령이다
  python scripts/runlog.py --name P030-stage2B -- python scripts/diag_kvcache.py --device cpu
  python scripts/runlog.py --name P036-trapping --num 018 -- python scripts/diag_trapping.py
  python scripts/runlog.py --name P035 --note "[1/2] ..." "control = qb_wsd60"

파일명 — **목적지에 따라 규약이 다르다**(2026-08-07)
  실험(`test_result/`):
    `{num}_log_{YYYYMMDD}_{name}.txt`   (--num 있을 때)
    `log_{YYYYMMDD}_{name}.txt`         (없을 때 — 결과문서 쓸 때 번호를 붙인다)
    **`--name` 은 `P0NN` 으로 시작해야 한다 — 아니면 거절한다.**
    같은 날 같은 이름이면 **한 파일에 이어 붙는다**(한 배치의 여러 단계가 한 파일에).
  비실험(`smoketest_logs/` 등, `--outdir` 또는 `TL_OUTDIR`):
    **`{YYYYMMDDHHMM}_{name}_{sha7}.txt`** — **분 정밀도 + 커밋 SHA7.**
    "어느 커밋에서 돌린 스모크인가" 를 파일명만으로 알아야 한다(사용자 요구).
    ⚠️ 분 정밀도를 그대로 쓰면 한 배치의 13번 호출이 13개 파일이 된다 →
      같은 이름의 최근 파일이 `--stamp-reuse-min`(기본 30) 분 안에 수정됐으면 **이어쓴다.**
      완전히 고정하려면 `TL_STAMP=202608071017`.

★모든 로그 **최상단에 커밋 배너**를 쓴다: `[commit] 5e063b1  +dirty 13개`.
  스모크는 코드를 고친 직후 돌리므로 작업트리가 거의 항상 더럽다 — SHA 만 적으면
  "이 커밋에서 검증했다" 로 **잘못 읽힌다.** 그래서 dirty 개수를 함께 적는다.

종료코드 = 자식 프로세스의 종료코드 그대로(배치의 `if errorlevel` 이 그대로 동작한다).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test_result"


def _split_argv(argv):
    """`--` 앞은 이 스크립트 옵션, 뒤는 실행할 명령. argparse 의 REMAINDER 는 까다로워 직접 나눈다."""
    if "--" not in argv:
        return argv, []
    i = argv.index("--")
    return argv[:i], argv[i + 1:]


def _git_state():
    """(sha7, dirty_count). git 이 없거나 저장소가 아니면 (None, None).

    ★왜 dirty 개수도 같이 보나 (2026-08-07)
      스모크는 **코드를 고친 직후** 돌린다 — 즉 작업트리는 거의 항상 더럽다.
      그래서 SHA 만 적으면 *"이 커밋에서 검증했다"* 로 **잘못 읽힌다.**
      정확한 진술은 *"이 커밋 + 커밋 안 된 변경 N개에서 돌렸다"* 다.
      함정 13("문서의 조치 기록은 코드 변경의 증거가 아니다")과 같은 계열의 정직성이다.
    """
    import subprocess as sp
    try:
        sha = sp.run(["git", "rev-parse", "--short=7", "HEAD"], cwd=ROOT,
                     capture_output=True, text=True, timeout=10)
        if sha.returncode != 0:
            return None, None
        st = sp.run(["git", "status", "--porcelain"], cwd=ROOT,
                    capture_output=True, text=True, timeout=20)
        n = len([l for l in st.stdout.splitlines() if l.strip()]) if st.returncode == 0 else None
        return sha.stdout.strip(), n
    except Exception:                            # noqa: BLE001 — 로그를 못 쓰게 만들면 안 된다
        return None, None


def _stamped_path(out, name, sha, forced_stamp, reuse_min):
    """비실험 로그 경로. `{YYYYMMDDHHMM}_{name}_{sha7}.txt`, 최근 파일이면 이어쓴다."""
    tag = sha or "nogit"
    if forced_stamp:
        return out / f"{forced_stamp}_{name}_{tag}.txt"
    now = time.time()
    hits = [p for p in out.glob(f"*_{name}_*.txt")
            if now - p.stat().st_mtime <= reuse_min * 60]
    if hits:
        newest = max(hits, key=lambda p: p.stat().st_mtime)
        sys.stdout.write(f"[runlog] 같은 이름의 최근 로그에 이어쓴다 -> {newest.name}  "
                         f"({(now - newest.stat().st_mtime)/60:.1f}분 전, "
                         f"--stamp-reuse-min {reuse_min})\n")
        sys.stdout.flush()
        return newest
    return out / f"{datetime.now():%Y%m%d%H%M}_{name}_{tag}.txt"


def _header(path, sha, dirty):
    """파일 최상단에 **한 번만** 커밋 배너를 쓴다. 이미 내용이 있으면 안 쓴다."""
    if path.exists() and path.stat().st_size > 0:
        return
    d = "" if dirty in (None, 0) else f"  +dirty {dirty}개(커밋 안 된 변경)"
    line = (f"[commit] {sha or '(git 없음)'}{d}\n"
            f"[commit] ⚠️ dirty 가 0 이 아니면 **이 커밋 상태로 검증한 것이 아니다** — "
            f"커밋 + 미커밋 변경의 합이다.\n"
            f"{'=' * 78}\n")
    with open(path, "a", encoding="utf-8", newline=os.linesep) as f:
        f.write(line); f.flush(); os.fsync(f.fileno())
    sys.stdout.write(line); sys.stdout.flush()


def _verify(path):
    """★기록 후 실제로 파일이 남았는지 확인한다.

    종전에는 `finally` 가 "로그 파일: ..." 를 **무조건** 찍었다. 그래서 파일이 만들어지지
    않았거나 곧바로 사라져도(동기화 지연·백신·수동 삭제) **콘솔은 성공한 것처럼 보였다.**
    경로만 믿지 말고 크기까지 찍는다.
    """
    try:
        if not path.exists():
            sys.stdout.write(f"[runlog] ★경고: 로그 파일이 존재하지 않는다 — {path}\n"
                             f"[runlog]   경로 권한·동기화·백신을 확인할 것. "
                             f"콘솔 출력이 유일한 사본이다.\n")
            return
        n = path.stat().st_size
        sys.stdout.write(f"[runlog] 로그 파일: {path}  ({n:,} bytes)\n")
        if n == 0:
            sys.stdout.write("[runlog] ★경고: 0바이트다. 기록이 실패했다.\n")
    except OSError as e:                     # noqa: BLE001
        sys.stdout.write(f"[runlog] ★경고: 로그 파일 확인 실패 — {type(e).__name__}: {e}\n")
    sys.stdout.flush()


def main():
    own, cmd = _split_argv(sys.argv[1:])
    ap = argparse.ArgumentParser(description="실행 로그 tee (콘솔 + test_result 즉시 기록)")
    ap.add_argument("--name", required=True, help="로그 파일명에 들어갈 실험 이름(예: P030-stage2B)")
    ap.add_argument("--num", default=None, help="결과 번호를 미리 아는 경우(예: 018)")
    ap.add_argument("--fsync-sec", type=float, default=2.0,
                    help="디스크 동기화 주기(초). 0 이면 매 줄마다 — 느리지만 가장 안전")
    ap.add_argument("--no-append", action="store_true",
                    help="같은 파일이 있어도 덮어쓴다(기본은 이어쓰기 = 유실 방지)")
    ap.add_argument("--outdir", default=None,
                    help="로그를 쓸 폴더(저장소 기준 상대경로). 기본 test_result. "
                         "스모크처럼 **실험 결과가 아닌** 로그는 smoketest_logs 로 보낸다")
    ap.add_argument("--stamp-reuse-min", type=float, default=30.0,
                    help="비실험 로그(smoketest_logs 등)에서 같은 이름의 파일이 이 분 안에 "
                         "수정됐으면 새 파일을 만들지 않고 이어쓴다. 한 배치의 여러 호출이 "
                         "한 파일에 모이게 하려는 것 (TL_STAMP 로 완전 고정 가능)")
    ap.add_argument("--note", nargs="+", default=None,
                    help="명령 대신 텍스트만 기록한다(배치의 단계 제목·판정 안내용). 여러 개면 여러 줄")
    a = ap.parse_args(own)

    if not cmd and not a.note:
        print("[runlog] 실행할 명령이 없습니다. `--` 뒤에 명령을 주거나 --note 를 쓰세요.",
              file=sys.stderr)
        print("  예) python scripts/runlog.py --name X -- python run100m.py train ...", file=sys.stderr)
        return 2

    # ★★2026-08-07 — **`--note` 와 명령을 같이 주면 명령이 조용히 버려졌다.**
    #   아래 `if a.note: ... return 0` 이 먼저 걸려 `cmd` 가 실행되지 않고, 그런데도
    #   **종료코드 0** 을 돌려주므로 배치의 `if errorlevel 1` 이 발화하지 않는다.
    #   → `run_P047_stage0_spam_rate.bat`(2026-08-07)이 정확히 이 형태였다:
    #     두 단계가 "성공" 으로 끝났는데 로그는 **84바이트(제목 두 줄)뿐**이었다.
    #     P045·P046·P048 은 `--note` 와 `--` 를 **별도 호출**로 나눠서 정상 동작했다.
    #   **조용한 무동작이 가장 나쁜 실패**이므로 여기서 거절한다(함정 10·13 계열).
    if cmd and a.note:
        print("[runlog] ★거절: `--note` 와 실행 명령을 함께 줄 수 없습니다.", file=sys.stderr)
        print("[runlog]   종전 구현은 note 만 쓰고 **명령을 조용히 버렸다** — "
              "그리고 종료코드 0 이라 배치가 성공으로 읽었다.", file=sys.stderr)
        print("[runlog]   두 줄로 나누세요:", file=sys.stderr)
        print('[runlog]     python scripts/runlog.py --name X --note "[1/2] 제목"', file=sys.stderr)
        print("[runlog]     python scripts/runlog.py --name X -- python ...", file=sys.stderr)
        return 2

    # ★★기본은 test_result 지만, 스모크·도구 로그는 **실험 결과가 아니다.**
    #   2026-08-07 사용자 보고: `run_smoke_check.bat` 을 돌렸는데 로그가 `test_result/` 로 갔다.
    #   원인 — `tool_smoke.bat` 이 `--outdir` 을 **한 번도 주지 않았다.**
    #   `run_smoke_check.bat` 헤더는 *"Logs land in smoketest_logs, deliberately NOT in
    #   test_result"* 라고 **적어만 뒀고 구현이 없었다**(함정 13: 문서의 조치 기록은
    #   코드 변경의 증거가 아니다).
    #   → 13곳에 `--outdir` 을 손으로 붙이는 대신 **환경변수 `TL_OUTDIR`** 을 읽는다.
    #     한 곳(진입 배치)에서 정하면 그 아래 전부에 걸린다 = "적용 대상을 한 곳에서
    #     정한다"(함정 18)의 적용.
    out = (ROOT / a.outdir) if a.outdir else \
          ((ROOT / os.environ["TL_OUTDIR"]) if os.environ.get("TL_OUTDIR") else OUT)
    out.mkdir(parents=True, exist_ok=True)
    is_experiment_dir = (out.resolve() == OUT.resolve())

    # ★2026-08-07 — `test_result/` 는 **실험 로그만** 담는다. 이제 규약이 아니라 **강제**다.
    #   종전에는 경고였고, 경고는 야간큐 v2 에서 무시됐다(로그 3개를 손으로 개명해야 했다).
    if not re.match(r"^P\d{3}", a.name):
        if is_experiment_dir:
            print(f"[runlog] ★거절: --name '{a.name}' 에 실험계획 번호(P0NN)가 없는데 "
                  f"목적지가 test_result 다.", file=sys.stderr)
            print(f"[runlog]   test_result 는 **실험 로그 전용**이다(번호를 붙여 결과문서와 "
                  f"짝을 맞춘다).", file=sys.stderr)
            print(f"[runlog]   실험이면  --name P0NN_{a.name}", file=sys.stderr)
            print(f"[runlog]   실험이 아니면  --outdir smoketest_logs  또는 "
                  f"set TL_OUTDIR=smoketest_logs", file=sys.stderr)
            return 2
        # 실험 폴더가 아니면 P번호가 없어도 정상이다(스모크·도구 로그).

    # ── 파일명 ────────────────────────────────────────────────────────────────
    #   실험(test_result): `{num}_log_{YYYYMMDD}_{name}.txt` — **종전과 동일**.
    #     같은 날 같은 이름이면 이어쓰기라 한 배치의 여러 단계가 한 파일에 모인다(설계).
    #   비실험(smoketest_logs 등): **`{YYYYMMDDHHMM}_{name}_{sha7}.txt`**
    #     ★분(minute) 정밀도 + 커밋 SHA7 — "어느 커밋에서 돌린 스모크인가" 를
    #       파일명만으로 알 수 있어야 한다(사용자 요구 2026-08-07).
    #     ⚠️ 분 정밀도를 그대로 쓰면 **한 배치의 13번 호출이 13개 파일로 쪼개진다.**
    #       그래서 **같은 이름의 최근 파일이 `--stamp-reuse-min` 분 안에 수정됐으면
    #       그 파일에 이어쓴다.** 스모크 한 판은 수 분이라 한 파일에 모이고,
    #       다음 판은(보통 그 사이에 로그를 읽으므로) 새 파일이 된다.
    #     `TL_STAMP` 를 주면 그 값을 그대로 쓴다(완전 결정적으로 하고 싶을 때).
    sha, dirty = _git_state()
    if is_experiment_dir:
        day = datetime.now().strftime("%Y%m%d")
        stem = f"{a.num}_log_{day}_{a.name}" if a.num else f"log_{day}_{a.name}"
        path = out / f"{stem}.txt"
    else:
        path = _stamped_path(out, a.name, sha, os.environ.get("TL_STAMP"),
                             a.stamp_reuse_min)

    # ★커밋 배너는 **비실험 로그 전용**이다(사용자 확인 2026-08-07).
    #   실험 로그는 조건 재현에 필요한 것이 이미 `runs/logs/*.json`(정본)과 헤더에 있고,
    #   결과문서가 커밋을 따로 적는다. 스모크는 **"어느 코드 상태를 검증했나" 자체가 목적**이라
    #   배너가 필요하다 — 목적이 다르므로 규약도 다르다.
    if not is_experiment_dir:
        _header(path, sha, dirty)      # 파일이 새로 만들어질 때만 쓴다

    # ── --note : 명령 없이 텍스트만. 배치의 echo 블록을 로그에도 남기는 통로다. ──
    if a.note:
        # ★배치는 `^>`·`^<`·`^|` 로 써야 한다(lint_bat.py E3 가 escape 안 된 것을 막는다).
        #   cmd 는 큰따옴표 안의 `^` 를 그대로 넘기므로 여기서 풀어 준다 — 안 그러면
        #   로그에 `-^>` 가 찍힌다. 배치 원문과 로그가 또 달라지는 것을 막는 처리다.
        def _unescape(t):
            for ch in "<>|&^":
                t = t.replace("^" + ch, ch)
            return t
        body = "".join(f"{_unescape(ln)}\n" for ln in a.note)
        with open(path, "a", encoding="utf-8", newline=os.linesep) as nf:
            nf.write(body)
            nf.flush()
            os.fsync(nf.fileno())
        sys.stdout.write(body)
        sys.stdout.flush()
        _verify(path)
        return 0

    started = datetime.now()
    t0 = time.time()
    # 자식이 파이프로 출력하면 파이썬 기본 인코딩이 locale(cp949) 이 되어 한글이 깨진다.
    # UTF-8 로 못박고, 버퍼링도 끈다(-u 와 동일 효과).
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")

    mode = "w" if a.no_append else "a"
    # ★newline=os.linesep — 종전 `newline=""` 는 자식이 보낸 `\n` 을 그대로 써서 파일이
    #   **LF 전용**이 됐다. Windows 메모장에서 전부 한 줄로 붙어 보이고, 콘솔과 또 다르게
    #   보이는 원인이었다. universal-newlines 로 이미 `\n` 로 정규화된 뒤이므로 여기서
    #   플랫폼 줄끝으로 되돌리는 것이 맞다.
    f = open(path, mode, encoding="utf-8", newline=os.linesep)
    rc = -1
    try:
        head = (f"\n{'=' * 78}\n"
                f"[runlog] {started:%Y-%m-%d %H:%M:%S}  name={a.name}\n"
                f"[runlog] cwd={os.getcwd()}\n"
                f"[runlog] cmd={' '.join(cmd)}\n"
                f"{'=' * 78}\n")
        f.write(head); f.flush()
        sys.stdout.write(head); sys.stdout.flush()

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                env=env, bufsize=1, text=True,
                                encoding="utf-8", errors="replace")
        last_sync = time.time()
        assert proc.stdout is not None
        for line in proc.stdout:            # 줄 단위 — 버퍼에 쌓지 않는다
            sys.stdout.write(line)
            sys.stdout.flush()              # 콘솔 실시간
            f.write(line)
            f.flush()                       # 파일 실시간(파이썬 버퍼 비움)
            now = time.time()
            if a.fsync_sec <= 0 or now - last_sync >= a.fsync_sec:
                os.fsync(f.fileno())        # ★OS 캐시까지 디스크로 — 정전·BSOD 대비
                last_sync = now
        rc = proc.wait()
    except KeyboardInterrupt:
        rc = 130
        msg = "\n[runlog] ★Ctrl+C 로 중단됨 — 여기까지의 로그는 보존됩니다.\n"
        f.write(msg); sys.stdout.write(msg)
    except Exception as e:                   # noqa: BLE001 — 어떤 예외든 꼬리말은 남긴다
        rc = 1
        msg = f"\n[runlog] ★래퍼 예외: {type(e).__name__}: {e}\n"
        f.write(msg); sys.stdout.write(msg)
    finally:
        el = time.time() - t0
        tail = (f"{'-' * 78}\n"
                f"[runlog] 종료코드 {rc}  소요 {el/60:.1f}분  "
                f"({datetime.now():%Y-%m-%d %H:%M:%S})\n"
                f"{'=' * 78}\n")
        try:
            f.write(tail); f.flush(); os.fsync(f.fileno())
        finally:
            f.close()
        sys.stdout.write(tail); sys.stdout.flush()
        _verify(path)                    # ★경로만 찍지 않는다 — 실제로 남았는지 본다
    return rc


if __name__ == "__main__":
    sys.exit(main())
