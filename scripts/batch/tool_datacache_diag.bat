@echo off
setlocal enabledelayedexpansion
REM =============================================================================
REM  tool_datacache_diag.bat  --  token cache diagnosis (scripts\diag_cache.py)
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
REM    call scripts\batch\tool_datacache_diag.bat
REM    if errorlevel 1 echo [WARN] ... - continuing
REM
REM  CONTRACT - inputs are ENVIRONMENT VARIABLES, not arguments.
REM    This repo bans the percent sign in .bat files (lint_bat.py E3), which rules
REM    out both the usual argument syntax AND for-loop variables. Delayed expansion
REM    with exclamation marks is the equivalent that stays inside the rule, so lists
REM    are passed as numbered slots instead of being iterated.
REM
REM    TL_LOGNAME  runlog name                default: datacache-diag
REM    TL_DATA1    first dataset              default: ko-en
REM    TL_DATA2    second dataset, blank to skip  default: ko-edu-en
REM    TL_TOKENS   pool size label            default: 300M
REM    TL_DOCFILTER set to anything to read the FILTERED cache (P037 stage 2)
REM    TL_NOPAUSE  set to anything to skip the pause    (callers should set it)
REM
REM  EXIT CODES: 0 ok / 9 could not find the repo root / see below for others.
REM  Every python call goes through scripts\runlog.py so the log survives a crash.
REM =============================================================================

REM  Locate the repo root. Works from the root or from this folder (double-click).
REM  A marker file is used because the percent sign is banned here.
if not exist run100m.py cd ..\..
if not exist run100m.py goto BADROOT

set TL_DFARG=
if defined TL_DOCFILTER set TL_DFARG=--doc-filter

if not defined TL_LOGNAME set TL_LOGNAME=datacache-diag
if not defined TL_DATA1 set TL_DATA1=ko-en
if not defined TL_DATA2 set TL_DATA2=ko-edu-en
if not defined TL_TOKENS set TL_TOKENS=300M

echo =============================================================
echo [tool] token cache diagnosis
python scripts\runlog.py --name !TL_LOGNAME! --note "[tool] token cache diagnosis"
echo   data1=!TL_DATA1!  data2=!TL_DATA2!  tokens=!TL_TOKENS!
echo =============================================================

echo.
echo [0] caches present on disk
python scripts\runlog.py --name !TL_LOGNAME! --note "[0] caches present on disk"
python scripts\runlog.py --name !TL_LOGNAME! -- python scripts\diag_cache.py
if errorlevel 1 echo [WARN] listing failed - continuing

echo.
echo =============================================================
echo [1] !TL_DATA1! !TL_TOKENS!
python scripts\runlog.py --name !TL_LOGNAME! --note "[1] !TL_DATA1! !TL_TOKENS!"
echo =============================================================
python scripts\runlog.py --name !TL_LOGNAME! -- python scripts\diag_cache.py --data !TL_DATA1! --tokens !TL_TOKENS! !TL_DFARG!
if errorlevel 1 echo [WARN] first dataset diag failed - continuing

if not defined TL_DATA2 goto NODATA2
echo.
echo =============================================================
echo [2] !TL_DATA2! !TL_TOKENS!
python scripts\runlog.py --name !TL_LOGNAME! --note "[2] !TL_DATA2! !TL_TOKENS!"
echo =============================================================
python scripts\runlog.py --name !TL_LOGNAME! -- python scripts\diag_cache.py --data !TL_DATA2! --tokens !TL_TOKENS! !TL_DFARG!
if errorlevel 1 echo [WARN] second dataset diag failed - continuing
:NODATA2

echo.
echo =================================================================
echo WHAT TO RECORD: train and val token counts, leakage percentage,
echo   bytes per token. Read bytes-per-token to identify which pool an
echo   old log used: 4.3750 is the 600M pool, 4.4778 is the 300M pool.
echo   Runs with different pools must NOT be compared.
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
