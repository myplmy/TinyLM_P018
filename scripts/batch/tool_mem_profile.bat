@echo off
setlocal enabledelayedexpansion
REM =============================================================================
REM  tool_mem_profile.bat  --  resident vs stored memory profile (scripts\mem_runtime.py)
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
REM    call scripts\batch\tool_mem_profile.bat
REM    if errorlevel 1 echo [WARN] ... - continuing
REM
REM  CONTRACT - inputs are ENVIRONMENT VARIABLES, not arguments.
REM    This repo bans the percent sign in .bat files (lint_bat.py E3), which rules
REM    out both the usual argument syntax AND for-loop variables. Delayed expansion
REM    with exclamation marks is the equivalent that stays inside the rule, so lists
REM    are passed as numbered slots instead of being iterated.
REM
REM    TL_LOGNAME  runlog name                             default: mem-profile
REM    TL_MODELS   space separated checkpoint tags          default: script default
REM    TL_SKIP_CPU   set to skip the cpu pass                 default: run it
REM    TL_SKIP_CUDA  set to skip the cuda pass                default: run it
REM    TL_MAXNEW   tokens to generate while measuring       default: 32
REM    TL_EXTRA    extra flags passed through, e.g. --drop-latent   default: none
REM    TL_NOPAUSE  set to anything to skip the pause    (callers should set it)
REM
REM  EXIT CODES: 0 ok / 9 could not find the repo root / see below for others.
REM  Every python call goes through scripts\runlog.py so the log survives a crash.
REM =============================================================================

REM  Locate the repo root. Works from the root or from this folder (double-click).
REM  A marker file is used because the percent sign is banned here.
if not exist run100m.py cd ..\..
if not exist run100m.py goto BADROOT

if not defined TL_LOGNAME set TL_LOGNAME=mem-profile
if not defined TL_MAXNEW set TL_MAXNEW=32
set TL_MARG=
if defined TL_MODELS set TL_MARG=--models !TL_MODELS!

echo =============================================================
echo [tool] resident memory profile
python scripts\runlog.py --name !TL_LOGNAME! --note "[tool] resident memory profile"
echo   Stored MB is a packing THEORY number. Resident MB is what the
echo   process actually holds. Result 016 measured them 31 to 45 times
echo   apart and in a DIFFERENT ORDER, so never quote one alone.
echo   max_new=!TL_MAXNEW!  extra=!TL_EXTRA!
echo =============================================================

echo.
echo [pre] psutil availability
python scripts\runlog.py --name !TL_LOGNAME! --note "[pre] psutil availability"
python -c "import psutil; print('  psutil OK', psutil.__version__)"
if errorlevel 1 echo [WARN] psutil missing - RSS will be nan. Run: pip install psutil

if defined TL_SKIP_CPU goto SKIPCPU
echo.
echo =============================================================
echo [run] device cpu - the deployment target, so this is the number that counts
python scripts\runlog.py --name !TL_LOGNAME! --note "[run] device cpu - the deployment target, so this is the number that counts"
echo =============================================================
python scripts\runlog.py --name !TL_LOGNAME! -- python scripts\mem_runtime.py --device cpu --max-new !TL_MAXNEW! !TL_MARG! !TL_EXTRA!
if errorlevel 1 echo [WARN] cpu measurement failed - continuing
:SKIPCPU

if defined TL_SKIP_CUDA goto SKIPCUDA
echo.
echo =============================================================
echo [run] device cuda - cross-check against an exact allocator peak
python scripts\runlog.py --name !TL_LOGNAME! --note "[run] device cuda - cross-check against an exact allocator peak"
echo =============================================================
python scripts\runlog.py --name !TL_LOGNAME! -- python scripts\mem_runtime.py --device cuda --max-new !TL_MAXNEW! !TL_MARG! !TL_EXTRA!
if errorlevel 1 echo [WARN] cuda measurement failed - continuing
:SKIPCUDA

echo.
echo =================================================================
echo WHAT TO RECORD
echo   1. per model: stored MB, tensor-sum resident MB, RSS delta,
echo      and the latent versus dequant split
echo   2. the TRANSFER RATE line
echo   3. whether the stored order and the resident order INVERTED
echo   4. with --drop-latent: the logit equivalence gate must read 0.000e+00
echo HOW TO READ IT
echo   CUDA peak is NOT a deployment number - the checkpoint carries AdamW
echo   state, so torch.load inflates it about 3x. Read the tensor sum.
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
