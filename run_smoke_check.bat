@echo off
setlocal enabledelayedexpansion
REM =============================================================================
REM  run_smoke_check.bat  --  RUN THIS AFTER ANY CODE CHANGE, BEFORE ANY LONG RUN
REM =============================================================================
REM
REM  WHAT IT IS
REM    A few minutes of tiny-preset training on synthetic data, followed by a check
REM    that every instrumentation field a result document depends on was actually
REM    written to the json. It is the cheapest net this repo has.
REM
REM  WHY IT EXISTS AS A ROOT BATCH
REM    The smoke logic lives in scripts\batch\tool_smoke.bat, which is a MODULE.
REM    Modules are not for direct invocation - an experiment or a maintenance entry
REM    point calls them. This file is that entry point.
REM
REM  WHY YOU SHOULD ACTUALLY RUN IT (2026-07-31, paid for in real time)
REM    The bpw accounting commit changed report(self, bpw=1.95) to bpw=None but left
REM    one line still using the raw argument. Every single training run then died with
REM        TypeError: unsupported operand type(s) for *: 'int' and 'NoneType'
REM    It surfaced only when P035 was launched, hours later. tool_smoke.bat runs
REM    `train --tiny`, which calls report() - so this batch would have caught it in
REM    minutes. Skipping the smoke was the entire cost of that incident.
REM
REM  READING THE OUTPUT
REM    The answer is the contract check that runs LAST: one block per training run,
REM    then a final total-error-count line. Zero means every instrumentation field a
REM    result document depends on was actually written to the json.
REM    The [VERIFY] banner is only a SECTION TITLE, not the answer - that wording
REM    confused things once already.
REM    Everything else is noise. Losses from 30 steps on synthetic data mean NOTHING:
REM    do not read them, compare them, or record them anywhere.
REM    Logs land in smoketest_logs (via TL_OUTDIR below), NOT in test_result.
REM    Filename carries the minute and the commit sha7.
REM
REM  COST: a few minutes, GPU barely used. Writes sm_* tags into runs\ (throwaway).
REM =============================================================================

if not exist run100m.py goto BADROOT

REM ***2026-08-07: TL_OUTDIR is the fix for logs landing in test_result.***
REM   This header used to CLAIM logs land in smoketest_logs while tool_smoke.bat
REM   never passed --outdir, so all 13 runlog calls wrote to test_result.
REM   Documented but not implemented (trap 13). runlog.py now reads TL_OUTDIR, so
REM   setting it HERE covers every call below it - one place, not thirteen.
REM   Filenames become {YYYYMMDDHHMM}_{name}_{sha7}.txt and every log opens with a
REM   [commit] banner, so you can tell WHICH COMMIT a smoke ran against.
set TL_OUTDIR=smoketest_logs
set TL_LOGNAME=smoke
set TL_NOPAUSE=1
call scripts\batch\tool_smoke.bat
if errorlevel 1 echo [WARN] smoke reported a problem - read which field is missing

echo.
echo =================================================================
echo VERDICT: read the final total-error-count line of the contract check.
echo   zero     -^> the contract holds. Long runs are safe to start.
echo   not zero -^> fix the missing field FIRST. A long run that cannot be
echo               written up is a long run thrown away.
echo   The [VERIFY] banner above it is a section title, not the answer.
echo =================================================================
echo done.
if not defined TL_NOPAUSE pause
exit /b 0

:BADROOT
echo.
echo =================================================================
echo [STOP] run this from the TinyLM working folder (run100m.py not found).
echo =================================================================
if not defined TL_NOPAUSE pause
exit /b 9
