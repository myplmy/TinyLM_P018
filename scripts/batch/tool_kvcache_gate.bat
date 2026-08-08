@echo off
setlocal enabledelayedexpansion
REM =============================================================================
REM  tool_kvcache_gate.bat  --  KV cache correctness gate (scripts\diag_kvcache.py) - CANONICAL
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
REM    call scripts\batch\tool_kvcache_gate.bat
REM    if errorlevel 1 echo [WARN] ... - continuing
REM
REM  CONTRACT - inputs are ENVIRONMENT VARIABLES, not arguments.
REM    This repo bans the percent sign in .bat files (lint_bat.py E3), which rules
REM    out both the usual argument syntax AND for-loop variables. Delayed expansion
REM    with exclamation marks is the equivalent that stays inside the rule, so lists
REM    are passed as numbered slots instead of being iterated.
REM
REM    TL_LOGNAME  runlog name                        default: kvcache-gate
REM    TL_MODELS   space separated tags               default: mA_g4s34_k4 mC_g8_k4 p6d
REM    TL_TOL      fp32 tolerance                     default: 1e-3
REM    TL_MAXNEW   decode steps compared              default: 24
REM    TL_FULL     set to also run the cuda and greedy passes (report only)
REM    TL_NOPAUSE  set to anything to skip the pause    (callers should set it)
REM
REM  EXIT CODES: 0 ok / 9 could not find the repo root / see below for others.
REM  Every python call goes through scripts\runlog.py so the log survives a crash.
REM =============================================================================

REM  Locate the repo root. Works from the root or from this folder (double-click).
REM  A marker file is used because the percent sign is banned here.
if not exist run100m.py cd ..\..
if not exist run100m.py goto BADROOT

if not defined TL_LOGNAME set TL_LOGNAME=kvcache-gate
if not defined TL_MODELS set TL_MODELS=mA_g4s34_k4 mC_g8_k4 p6d
if not defined TL_TOL set TL_TOL=1e-3
if not defined TL_MAXNEW set TL_MAXNEW=24

echo =============================================================
echo [tool] KV cache correctness gate - teacher forced logits, fp32, cpu
python scripts\runlog.py --name !TL_LOGNAME! --note "[tool] KV cache correctness gate - teacher forced logits, fp32, cpu"
echo   This is the CANONICAL gate. Do not gate on greedy strings: bf16 ULP
echo   plus autoregressive feedback makes near-tie models fail while the
echo   noisiest model passes (result 014 section 9).
echo   models=!TL_MODELS!  tol=!TL_TOL!
echo =============================================================

echo.
echo [guard] scripts\diag_kvcache.py present
python scripts\runlog.py --name !TL_LOGNAME! --note "[guard] scripts\diag_kvcache.py present"
if not exist scripts\diag_kvcache.py goto NOTIMPL

echo.
echo =============================================================
echo [1] HARD GATE : fp32 on cpu
python scripts\runlog.py --name !TL_LOGNAME! --note "[1] HARD GATE : fp32 on cpu"
echo =============================================================
python scripts\runlog.py --name !TL_LOGNAME! -- python scripts\diag_kvcache.py --models !TL_MODELS! --device cpu --tol !TL_TOL! --max-new !TL_MAXNEW!
if errorlevel 1 goto GATEBAD
echo   [OK] logit equivalence within tolerance. Downstream numbers are meaningful.

if not defined TL_FULL goto TAIL

echo.
echo =============================================================
echo [2] same gate on cuda (bf16 autocast) - sizes the precision gap
python scripts\runlog.py --name !TL_LOGNAME! --note "[2] same gate on cuda (bf16 autocast) - sizes the precision gap"
echo =============================================================
python scripts\runlog.py --name !TL_LOGNAME! -- python scripts\diag_kvcache.py --models !TL_MODELS! --device cuda --tol !TL_TOL! --max-new !TL_MAXNEW!
if errorlevel 1 echo [INFO] bf16 exceeds the fp32 tolerance - record it, this is not a failure

echo.
echo =============================================================
echo [3] greedy divergence, REPORT ONLY, with the tie margin
python scripts\runlog.py --name !TL_LOGNAME! --note "[3] greedy divergence, REPORT ONLY, with the tie margin"
echo =============================================================
python scripts\runlog.py --name !TL_LOGNAME! -- python scripts\diag_kvcache.py --models !TL_MODELS! --device cuda --report-greedy --max-new !TL_MAXNEW!
if errorlevel 1 echo [INFO] greedy strings differ - check the tie margin before calling it a bug

:TAIL
echo.
echo =================================================================
echo VERDICT: pass means the cache path is correct and speed or quality
echo   numbers taken after it can be compared. Fail means stop and fix.
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

:GATEBAD
echo.
echo =================================================================
echo [STOP] fp32 logit equivalence FAILED. This is a real cache bug.
echo   Read the per-prompt error pattern above:
echo     grows with position  -^> RoPE offset
echo     uniform after prefill -^> mask
echo     tied models only      -^> CLA cache sharing
echo   Do not measure speed or quality until this passes.
echo =================================================================
if not defined TL_NOPAUSE pause
exit /b 1

:NOTIMPL
echo.
echo =================================================================
echo [STOP] scripts\diag_kvcache.py is missing. Nothing was executed.
echo =================================================================
if not defined TL_NOPAUSE pause
exit /b 3
