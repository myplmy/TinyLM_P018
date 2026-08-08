@echo off
setlocal enabledelayedexpansion
REM =============================================================================
REM  tool_smoke.bat  --  instrumentation contract smoke test (tiny preset, synthetic data)
REM  *** MODULE. NOT AN EXPERIMENT. ***
REM
REM  Do not point a user at this file to "run experiment X". It has no experiment
REM  number, no plan reference and no place in test_result. An experiment is a
REM  batch in the REPO ROOT that sets the variables below and calls this. That is
REM  what makes a run identifiable later; a bare tool invocation is not.
REM
REM  WHY THIS SHAPE
REM    These files used to be run_P0NN_*.bat - a plan number welded to a tool. So
REM    reusing the tool for a different plan meant either lying in the log name or
REM    copying the file. Both happened. The tool now carries only the function and
REM    the caller carries the identity.
REM
REM  CALLING IT
REM    set TL_LOGNAME=P0NN-stageM
REM    set TL_NOPAUSE=1
REM    call scripts\batch\tool_smoke.bat
REM    if errorlevel 1 echo [WARN] ... - continuing
REM
REM  CONTRACT - inputs are ENVIRONMENT VARIABLES, not arguments.
REM    This repo bans the percent sign in .bat files (lint_bat.py E3), which rules
REM    out both the usual argument syntax AND for-loop variables. Delayed expansion
REM    with exclamation marks is the equivalent that stays inside the rule, so lists
REM    are passed as numbered slots instead of being iterated.
REM
REM    TL_LOGNAME  runlog name   default: smoke
REM    (no other inputs - the whole point is that it is a FIXED contract check)
REM    TL_NOPAUSE  set to anything to skip the pause    (callers should set it)
REM
REM  EXIT CODES: 0 ok / 9 could not find the repo root / see below for others.
REM  Every python call goes through scripts\runlog.py so the log survives a crash.
REM =============================================================================

REM  Locate the repo root. Works from the root or from this folder (double-click).
REM  A marker file is used because the percent sign is banned here.
if not exist run100m.py cd ..\..
if not exist run100m.py goto BADROOT

REM ***A smoke is NEVER an experiment, so its log NEVER belongs in test_result.***
REM   Until 2026-08-07 this file passed no --outdir at all and all 13 runlog calls
REM   wrote to test_result, while run_smoke_check.bat's header claimed otherwise.
REM   The caller normally sets TL_OUTDIR; this default makes the module correct on
REM   its own too, so the invariant does not depend on who calls it.
if not defined TL_OUTDIR set TL_OUTDIR=smoketest_logs
if not defined TL_LOGNAME set TL_LOGNAME=smoke

echo =============================================================
echo [tool] instrumentation contract smoke test
python scripts\runlog.py --name !TL_LOGNAME! --note "[tool] instrumentation contract smoke test"
echo   Runs the tiny preset on synthetic data and then verifies that every
echo   field a result document depends on was actually written to the json.
echo   Minutes here, versus discovering a missing field after a long run.
echo   RUN THIS BEFORE USING ANY NEW SCRIPT.
echo =============================================================

echo.
echo [pre] static attribute check - catches cfg.WRONG_NAME without loading torch
python scripts\runlog.py --name !TL_LOGNAME! --note "[pre] static attribute check - catches cfg.WRONG_NAME without loading torch"
python scripts\runlog.py --name !TL_LOGNAME! -- python scripts\check_attrs.py
if errorlevel 1 echo [WARN] attribute check found problems - FIX THEM FIRST

echo.
echo [1] baseline
python scripts\runlog.py --name !TL_LOGNAME! --note "[1] baseline"
python scripts\runlog.py --name !TL_LOGNAME! -- python run100m.py train --arch tied --tiny --data synthetic --tokens 2M --steps 30 --micro-bs 4 --seq 128 --accum 2 --eval-every 15 --tag sm_base
if errorlevel 1 echo [WARN] sm_base failed - continuing

echo.
echo [2] seed path
python scripts\runlog.py --name !TL_LOGNAME! --note "[2] seed path"
python scripts\runlog.py --name !TL_LOGNAME! -- python run100m.py train --arch tied --tiny --data synthetic --tokens 2M --steps 30 --micro-bs 4 --seq 128 --accum 2 --eval-every 15 --seed 4242 --tag sm_seed
if errorlevel 1 echo [WARN] sm_seed failed - continuing

echo.
echo [3] 3:4 path
python scripts\runlog.py --name !TL_LOGNAME! --note "[3] 3:4 path"
python scripts\runlog.py --name !TL_LOGNAME! -- python run100m.py train --arch tied --tiny --data synthetic --tokens 2M --steps 30 --micro-bs 4 --seq 128 --accum 2 --eval-every 15 --sparse34 --tag sm_s34
if errorlevel 1 echo [WARN] sm_s34 failed - continuing

echo.
echo [4] wsd schedule path
python scripts\runlog.py --name !TL_LOGNAME! --note "[4] wsd schedule path"
python scripts\runlog.py --name !TL_LOGNAME! -- python run100m.py train --arch tied --tiny --data synthetic --tokens 2M --steps 30 --micro-bs 4 --seq 128 --accum 2 --eval-every 15 --sched wsd --anneal-end 0.80 --decay-frac 0.2 --tag sm_sched
if errorlevel 1 echo [WARN] sm_sched failed - continuing

echo.
echo [5] no grad checkpoint path
python scripts\runlog.py --name !TL_LOGNAME! --note "[5] no grad checkpoint path"
python scripts\runlog.py --name !TL_LOGNAME! -- python run100m.py train --arch tied --tiny --data synthetic --tokens 2M --steps 30 --micro-bs 4 --seq 128 --accum 2 --eval-every 15 --no-ckpt --tag sm_nockpt
if errorlevel 1 echo [WARN] sm_nockpt failed - continuing

echo.
echo [6] KD path - needs a tiny dense parent first
python scripts\runlog.py --name !TL_LOGNAME! --note "[6] KD path - needs a tiny dense parent first"
python scripts\runlog.py --name !TL_LOGNAME! -- python run100m.py train --arch dense --tiny --data synthetic --tokens 2M --steps 30 --micro-bs 4 --seq 128 --accum 2 --eval-every 15
if errorlevel 1 echo [WARN] tiny dense parent failed - sm_kd will be skipped
python scripts\runlog.py --name !TL_LOGNAME! -- python run100m.py train --arch tied --tiny --data synthetic --tokens 2M --steps 30 --micro-bs 4 --seq 128 --accum 2 --eval-every 15 --kd --init-from --mlp-group 4 --kd-every 4 --tag sm_kd
if errorlevel 1 echo [WARN] sm_kd failed - continuing

echo.
echo =============================================================
echo [VERIFY] every instrumentation field was recorded
python scripts\runlog.py --name !TL_LOGNAME! --note "[VERIFY] every instrumentation field was recorded"
echo =============================================================
python scripts\runlog.py --name !TL_LOGNAME! -- python scripts\check_smoke.py
if errorlevel 1 echo [WARN] contract check FAILED - read which field is missing

echo.
echo =================================================================
echo VERDICT - read the LAST line of the contract check, which reports the
echo   total error count, plus the per-run blocks above it (each is [OK] or
echo   names the field it could not find). That is the whole result.
echo.
echo   The [VERIFY] banner is just a SECTION TITLE - it is not the answer.
echo   Everything else is noise: losses from 30 steps on synthetic data mean
echo   NOTHING. Do not read them, compare them, or record them anywhere.
echo.
echo   Logs are written to smoketest_logs, deliberately not test_result.
echo =================================================================
goto DONE

:DONE
if not defined TL_NOPAUSE pause
exit /b 0

:BADROOT
echo.
echo =================================================================
echo [STOP] could not locate the repo root (run100m.py not found).
echo   Launch this from the TinyLM working folder, or let a root-level
echo   experiment batch call it. Nothing was executed.
echo =================================================================
if not defined TL_NOPAUSE pause
exit /b 9
