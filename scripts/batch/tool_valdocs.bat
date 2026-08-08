@echo off
setlocal enabledelayedexpansion
REM =============================================================================
REM  tool_valdocs.bat  --  per-document val loss decomposition (scripts\diag_val_docs.py)
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
REM    call scripts\batch\tool_valdocs.bat
REM    if errorlevel 1 echo [WARN] ... - continuing
REM
REM  CONTRACT - inputs are ENVIRONMENT VARIABLES, not arguments.
REM    This repo bans the percent sign in .bat files (lint_bat.py E3), which rules
REM    out both the usual argument syntax AND for-loop variables. Delayed expansion
REM    with exclamation marks is the equivalent that stays inside the rule, so lists
REM    are passed as numbered slots instead of being iterated.
REM
REM    TL_LOGNAME  runlog name                default: valdocs
REM    TL_DATA1    first dataset              default: ko-edu-en
REM    TL_DATA2    second dataset, blank to skip  default: ko-en
REM    TL_TOKENS   pool size label            default: 300M
REM    TL_TAG      checkpoint tag             default: dense
REM    TL_ARCH     dense or tied              default: dense
REM    TL_INSPECT  how many top documents to dump text for  default: none
REM    TL_DOCFILTER set to anything to read the FILTERED cache (P037 stage 2)
REM    TL_CKPTTOKENS training-budget label for the checkpoint name, if it DIFFERS
REM                 from the pool size. Result 023 section 7: one flag meant both,
REM                 so a 600M pool went looking for a 600M-budget checkpoint.
REM    TL_DATA2=none  skips the second dataset. Plain 'set TL_DATA2=' does NOT
REM                 work - cmd UNDEFINES the variable and the default comes back.
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
set TL_CKARG=
if defined TL_CKPTTOKENS set TL_CKARG=--ckpt-tokens !TL_CKPTTOKENS!

if not defined TL_LOGNAME set TL_LOGNAME=valdocs
if not defined TL_DATA1 set TL_DATA1=ko-edu-en
if not defined TL_DATA2 set TL_DATA2=ko-en
if not defined TL_TOKENS set TL_TOKENS=300M
if not defined TL_TAG set TL_TAG=dense
if not defined TL_ARCH set TL_ARCH=dense
set TL_IARG=
if defined TL_INSPECT set TL_IARG=--inspect !TL_INSPECT!

echo =============================================================
echo [tool] per-document val loss decomposition
python scripts\runlog.py --name !TL_LOGNAME! --note "[tool] per-document val loss decomposition"
echo   Answers: is the val number carried by a handful of documents.
echo   With TL_INSPECT it also dumps the head and tail of the worst
echo   documents plus character statistics, which is what separates a
echo   BOUNDARY bug from a CONTENT problem.
echo   data1=!TL_DATA1!  data2=!TL_DATA2!  inspect=!TL_INSPECT!
echo =============================================================

echo.
echo =============================================================
echo [1] !TL_DATA1!
python scripts\runlog.py --name !TL_LOGNAME! --note "[1] !TL_DATA1!"
echo =============================================================
python scripts\runlog.py --name !TL_LOGNAME! -- python scripts\diag_val_docs.py --data !TL_DATA1! --tokens !TL_TOKENS! --arch !TL_ARCH! --tag !TL_TAG! !TL_IARG! !TL_DFARG! !TL_CKARG!
if errorlevel 1 echo [WARN] first dataset decomposition failed - continuing

if not defined TL_DATA2 goto NODATA2
if /i "!TL_DATA2!"=="none" goto NODATA2
echo.
echo =============================================================
echo [2] !TL_DATA2!
python scripts\runlog.py --name !TL_LOGNAME! --note "[2] !TL_DATA2!"
echo =============================================================
python scripts\runlog.py --name !TL_LOGNAME! -- python scripts\diag_val_docs.py --data !TL_DATA2! --tokens !TL_TOKENS! --arch !TL_ARCH! --tag !TL_TAG! !TL_IARG! !TL_DFARG! !TL_CKARG!
if errorlevel 1 echo [WARN] second dataset decomposition failed - continuing
:NODATA2

echo.
echo =================================================================
echo WHAT TO RECORD
echo   1. total val loss and the top contributing documents
echo   2. loss after excluding the top 1, 2, 5, 10
echo   3. with TL_INSPECT: eos count vs document count, documents under
echo      16 tokens, and whether the big documents are spam, tables or code
echo   READ: many tiny documents means OVER-SPLITTING (a tool bug). Few but
echo   huge junk documents means a CONTENT filter is what is missing.
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
