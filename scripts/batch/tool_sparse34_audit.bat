@echo off
setlocal enabledelayedexpansion
REM =============================================================================
REM  tool_sparse34_audit.bat  --  3:4 semi-structured ternary audit (scripts\diag_sparse34.py)
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
REM    call scripts\batch\tool_sparse34_audit.bat
REM    if errorlevel 1 echo [WARN] ... - continuing
REM
REM  CONTRACT - inputs are ENVIRONMENT VARIABLES, not arguments.
REM    This repo bans the percent sign in .bat files (lint_bat.py E3), which rules
REM    out both the usual argument syntax AND for-loop variables. Delayed expansion
REM    with exclamation marks is the equivalent that stays inside the rule, so lists
REM    are passed as numbered slots instead of being iterated.
REM
REM    TL_LOGNAME  runlog name                default: sparse34-audit
REM    TL_MODELS   space separated tags       default: mA_g4s34_k4 mC_g8_k4 p6d
REM    TL_S34MODEL the 3:4 model for the per-layer pass  default: mA_g4s34_k4
REM    TL_LAYERS   layers in the detail pass  default: 8
REM    TL_NOPAUSE  set to anything to skip the pause    (callers should set it)
REM
REM  EXIT CODES: 0 ok / 9 could not find the repo root / see below for others.
REM  Every python call goes through scripts\runlog.py so the log survives a crash.
REM =============================================================================

REM  Locate the repo root. Works from the root or from this folder (double-click).
REM  A marker file is used because the percent sign is banned here.
if not exist run100m.py cd ..\..
if not exist run100m.py goto BADROOT

if not defined TL_LOGNAME set TL_LOGNAME=sparse34-audit
if not defined TL_MODELS set TL_MODELS=mA_g4s34_k4 mC_g8_k4 p6d
if not defined TL_S34MODEL set TL_S34MODEL=mA_g4s34_k4
if not defined TL_LAYERS set TL_LAYERS=8

echo =============================================================
echo [tool] 3:4 ternary audit - implementation, real sparsity, bpw accounting
python scripts\runlog.py --name !TL_LOGNAME! --note "[tool] 3:4 ternary audit - implementation, real sparsity, bpw accounting"
echo   Reminder (result 016 section 7.2): plain TWN already zeroes 43 to 46
echo   percent of weights. 3:4 REDUCES zeros to 25 percent. It is not a
echo   sparsification. The only gain is 5-bit aligned packing, and both the
echo   packing kernel and the paper's Arenas part are unimplemented.
echo   models=!TL_MODELS!
echo =============================================================

echo.
echo [guard] scripts\diag_sparse34.py present
python scripts\runlog.py --name !TL_LOGNAME! --note "[guard] scripts\diag_sparse34.py present"
if not exist scripts\diag_sparse34.py goto NOTIMPL

echo.
echo =============================================================
echo [1/2] full audit, summary level
python scripts\runlog.py --name !TL_LOGNAME! --note "[1/2] full audit, summary level"
echo =============================================================
python scripts\runlog.py --name !TL_LOGNAME! -- python scripts\diag_sparse34.py --models !TL_MODELS!
if errorlevel 2 echo [WARN] some checkpoints were missing - read which ones above
if errorlevel 1 echo [WARN] audit reported a problem - continuing

echo.
echo =============================================================
echo [2/2] per layer detail for the 3:4 model
python scripts\runlog.py --name !TL_LOGNAME! --note "[2/2] per layer detail for the 3:4 model"
echo =============================================================
python scripts\runlog.py --name !TL_LOGNAME! -- python scripts\diag_sparse34.py --models !TL_S34MODEL! --layers !TL_LAYERS!
if errorlevel 1 echo [WARN] per-layer pass failed - continuing

echo.
echo =================================================================
echo WHAT TO RECORD
echo   1. measured zero fraction per model - compare 3:4 against plain TWN
echo   2. bpw under each accounting convention (B is canonical since 26-07-31)
echo   3. whether the reported reduction survives ONE convention throughout
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

:NOTIMPL
echo.
echo =================================================================
echo [STOP] scripts\diag_sparse34.py is missing. Nothing was executed.
echo =================================================================
if not defined TL_NOPAUSE pause
exit /b 3
