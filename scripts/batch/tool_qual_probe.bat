@echo off
setlocal enabledelayedexpansion
REM =============================================================================
REM  tool_qual_probe.bat  --  qualitative generation probe (scripts\probe_prompts.py)
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
REM    call scripts\batch\tool_qual_probe.bat
REM    if errorlevel 1 echo [WARN] ... - continuing
REM
REM  CONTRACT - inputs are ENVIRONMENT VARIABLES, not arguments.
REM    This repo bans the percent sign in .bat files (lint_bat.py E3), which rules
REM    out both the usual argument syntax AND for-loop variables. Delayed expansion
REM    with exclamation marks is the equivalent that stays inside the rule, so lists
REM    are passed as numbered slots instead of being iterated.
REM
REM    TL_LOGNAME  runlog name                default: qual-probe
REM    TL_MODELS   space separated tags       default: script default
REM    TL_TEMPS    space separated temperatures  default: script default
REM    TL_NOPAUSE  set to anything to skip the pause    (callers should set it)
REM
REM  EXIT CODES: 0 ok / 9 could not find the repo root / see below for others.
REM  Every python call goes through scripts\runlog.py so the log survives a crash.
REM =============================================================================

REM  Locate the repo root. Works from the root or from this folder (double-click).
REM  A marker file is used because the percent sign is banned here.
if not exist run100m.py cd ..\..
if not exist run100m.py goto BADROOT

if not defined TL_LOGNAME set TL_LOGNAME=qual-probe
set TL_MARG=
if defined TL_MODELS set TL_MARG=--models !TL_MODELS!
set TL_TARG=
if defined TL_TEMPS set TL_TARG=--temps !TL_TEMPS!

echo =============================================================
echo [tool] qualitative probe
python scripts\runlog.py --name !TL_LOGNAME! --note "[tool] qualitative probe"
echo   This does NOT choose an architecture. val_loss and REVIEW1 do that.
echo   Five samples is an impression, not a statistic. At 2.3 tokens per
echo   parameter and no SFT, wrong facts are the EXPECTED outcome; what is
echo   being checked is script consistency, particle and ending validity,
echo   register inertia and absence of repetition collapse.
echo =============================================================

echo.
echo [0] prompts and what each one checks
python scripts\runlog.py --name !TL_LOGNAME! --note "[0] prompts and what each one checks"
python scripts\runlog.py --name !TL_LOGNAME! -- python scripts\probe_prompts.py --list
if errorlevel 1 echo [WARN] listing failed - continuing

echo.
echo [1] generating
python scripts\runlog.py --name !TL_LOGNAME! --note "[1] generating"
python scripts\runlog.py --name !TL_LOGNAME! -- python scripts\probe_prompts.py !TL_MARG! !TL_TARG!
if errorlevel 1 echo [WARN] probe failed - see the traceback above

echo.
echo =================================================================
echo WHAT TO RECORD: fill the checklist by eye. Do not score it.
echo   All empty        -^> suspect the pipeline, not the model
echo   Fluent but wrong -^> normal at this token budget
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
