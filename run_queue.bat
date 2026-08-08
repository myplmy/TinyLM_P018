@echo off
setlocal enabledelayedexpansion
REM =============================================================================
REM  run_queue.bat -- INTERACTIVE SCHEDULER. Pick an order, then walk away.
REM  Rule: ai_dev_tool\03 section 6 -- one experiment, one batch. This file only
REM        CALLS them. Never put a training or diagnostic command in here.
REM =============================================================================
REM
REM HOW TO USE
REM   Double-click it, or run it from the TinyLM folder. It prints the menu,
REM   asks for up to 8 slots one at a time, echoes the plan back for
REM   confirmation, and then runs them in that order without asking anything
REM   else. Press ENTER on an empty slot to stop entering ids.
REM
REM   Put  0  in slot 1 to run the smoke check first. ***Do that whenever the
REM   code changed*** - it is the only step that aborts the whole queue.
REM
REM   Nothing is remembered between runs. Re-running it asks again.
REM   An id whose batch file is missing is skipped with a message, not an error,
REM   so a finished-and-renamed experiment does not break the queue.
REM
REM WHY IT ASKS INSTEAD OF HARDCODING
REM   The previous run_night_queue_v4.bat fixed the order at write time, so
REM   changing the plan meant editing a batch file at bedtime. This asks once,
REM   up front, and is unattended from then on.
REM
REM ***THE PERCENT SIGN IS BANNED IN THIS REPO'S BATCH FILES*** (lint rule 3),
REM   so there is no for loop and no positional argument. Slots are read into
REM   numbered variables and dispatched through subroutines. That is why the
REM   code below looks repetitive - it is a constraint, not an oversight.
REM =============================================================================

if not exist run100m.py cd ..\..
if not exist run100m.py goto BADROOT

echo.
echo =================================================================
echo   TinyLM experiment queue -- pick an order
echo =================================================================
echo.
echo    id  batch                                  time    what it answers
echo    --  -------------------------------------  ------  -----------------------------
echo     0  run_smoke_check.bat                    min     code sanity. ABORTS on failure
echo     1  run_P018_stage2_teacher_infer.bat      40 min  do the two teacher levers ADD
echo     2  run_P050_parent_ablation.bat           9.5 h   how much does the parent buy
echo     3  run_P052_seed_replicate.bat            7 h     does the g16 verdict survive seed 2
echo     4  run_P022B_stage2_optstate.bat          4.2 h   AdamW state in BF16, -254 MB
echo     5  run_P046_emb_rank128.bat               3.5 h   E=128 re-measured with SVD
echo.
echo   done and renamed -done: P022B-0, P042-1, P042-2, P051, P018-1.
echo   Their ids are gone. A missing file is skipped, never an error.
echo.
echo   SUGGESTED BUNDLES -- type the digits into the slots, one digit per slot
echo.
echo    A  short night, about 40 min     0 1
echo         cheapest and most decision-relevant thing left. Do this first.
echo    B  one full night, about 10 h    0 1 2
echo         bundle A plus P050 - it corrects how P046/P048/P049 should be read.
echo    C  review preparation, 17 h      0 1 2 3
echo         everything REVIEW2 needs before it can name a winner.
echo    D  memory axes, about 5 h        0 1 4
echo         teacher levers plus the BF16 optimizer state. ***Watch 4.***
echo    E  the lot, about 25 h           0 1 2 3 4 5
echo.
echo   NOTE ***4, P022B stage 2, is the one to babysit.*** A low precision
echo   optimizer can drift quietly instead of failing loudly. Its parts 1 and 2
echo   are cheap gates - look at them before letting part 3 spend 3.5 hours.
echo   The watch list is in test_plan\P022B section 2.3.
echo.
echo   NOTE 1 is a VRAM+SPEED measurement and wants the machine to itself.
echo   ***Set the NVIDIA 'CUDA - System Memory Fallback' to PREFER NO FALLBACK***
echo   before running it, or its main question cannot be answered.
echo   See docs\methods\09_training_memory.md section 4.
echo.

set TL_J1=
set TL_J2=
set TL_J3=
set TL_J4=
set TL_J5=
set TL_J6=
set TL_J7=
set TL_J8=
set /p TL_J1=slot 1 - id, or ENTER to stop:
if not defined TL_J1 goto NOTHING
set /p TL_J2=slot 2 - id, or ENTER to stop:
if not defined TL_J2 goto CONFIRM
set /p TL_J3=slot 3 - id, or ENTER to stop:
if not defined TL_J3 goto CONFIRM
set /p TL_J4=slot 4 - id, or ENTER to stop:
if not defined TL_J4 goto CONFIRM
set /p TL_J5=slot 5 - id, or ENTER to stop:
if not defined TL_J5 goto CONFIRM
set /p TL_J6=slot 6 - id, or ENTER to stop:
if not defined TL_J6 goto CONFIRM
set /p TL_J7=slot 7 - id, or ENTER to stop:
if not defined TL_J7 goto CONFIRM
set /p TL_J8=slot 8 - id, or ENTER to stop:

:CONFIRM
echo.
echo =================================================================
echo   plan
echo =================================================================
set TL_CUR=!TL_J1!
call :SHOW
set TL_CUR=!TL_J2!
call :SHOW
set TL_CUR=!TL_J3!
call :SHOW
set TL_CUR=!TL_J4!
call :SHOW
set TL_CUR=!TL_J5!
call :SHOW
set TL_CUR=!TL_J6!
call :SHOW
set TL_CUR=!TL_J7!
call :SHOW
set TL_CUR=!TL_J8!
call :SHOW
echo.
set TL_GO=
set /p TL_GO=start now, y or n:
if /i not "!TL_GO!"=="y" goto ABORT

REM child batches must not pause, or the queue stalls overnight
set TL_NOPAUSE=1
set TL_ABORT=

echo.
echo [queue] started
date /t
time /t

set TL_CUR=!TL_J1!
call :RUN
if defined TL_ABORT goto SMOKEHALT
set TL_CUR=!TL_J2!
call :RUN
if defined TL_ABORT goto SMOKEHALT
set TL_CUR=!TL_J3!
call :RUN
if defined TL_ABORT goto SMOKEHALT
set TL_CUR=!TL_J4!
call :RUN
if defined TL_ABORT goto SMOKEHALT
set TL_CUR=!TL_J5!
call :RUN
if defined TL_ABORT goto SMOKEHALT
set TL_CUR=!TL_J6!
call :RUN
if defined TL_ABORT goto SMOKEHALT
set TL_CUR=!TL_J7!
call :RUN
if defined TL_ABORT goto SMOKEHALT
set TL_CUR=!TL_J8!
call :RUN

echo.
echo =================================================================
echo [queue] finished
date /t
time /t
echo.
echo   Logs: test_result for experiment runs, smoketest_logs for gates.
echo   Each batch printed its own WHAT TO RECORD block at the end. Read that
echo   before reading any number out of the run itself - several of these
echo   batches deliberately tell you which numbers are NOT to be trusted.
echo =================================================================
set TL_NOPAUSE=
if not defined TL_NOPAUSE pause
exit /b 0

REM ---------------------------------------------------------------- subroutines

:RESOLVE
REM  in: TL_CUR (an id)   out: TL_BAT (filename or empty) and TL_WHAT (label)
REM  ***Parenthesised blocks, not the ampersand form.*** In cmd,
REM      if COND set A=x^& set B=y
REM  runs the second set UNCONDITIONALLY - the ampersand splits at parser level.
set TL_BAT=
set TL_WHAT=
if "!TL_CUR!"=="0" (
  set TL_BAT=run_smoke_check.bat
  set TL_WHAT=smoke check - ABORTS the queue on failure
)
if "!TL_CUR!"=="1" (
  set TL_BAT=run_P018_stage2_teacher_infer.bat
  set TL_WHAT=P018 stage 2 - compressed teacher + infer mode, 40 min, wants the machine alone
)
if "!TL_CUR!"=="2" (
  set TL_BAT=run_P050_parent_ablation.bat
  set TL_WHAT=P050 - parent dependency ablation, 9.5 h
)
if "!TL_CUR!"=="3" (
  set TL_BAT=run_P052_seed_replicate.bat
  set TL_WHAT=P052 - second seed for the g16 verdict, 7 h
)
if "!TL_CUR!"=="4" (
  set TL_BAT=run_P022B_stage2_optstate.bat
  set TL_WHAT=P022B stage 2 - AdamW state in BF16, 4.2 h, babysit parts 1 and 2
)
if "!TL_CUR!"=="5" (
  set TL_BAT=run_P046_emb_rank128.bat
  set TL_WHAT=P046 - embedding E=128 re-measured with SVD, 3.5 h
)
exit /b 0

:SHOW
set TL_CUR=!TL_J2!
call :SHOW
set TL_CUR=!TL_J3!
call :SHOW
set TL_CUR=!TL_J4!
call :SHOW
set TL_CUR=!TL_J5!
call :SHOW
set TL_CUR=!TL_J6!
call :SHOW
set TL_CUR=!TL_J7!
call :SHOW
set TL_CUR=!TL_J8!
call :SHOW
echo.
set TL_GO=
set /p TL_GO=start now, y or n:
if /i not "!TL_GO!"=="y" goto ABORT

REM child batches must not pause, or the queue stalls overnight
set TL_NOPAUSE=1
set TL_ABORT=

echo.
echo [queue] started
date /t
time /t

set TL_CUR=!TL_J1!
call :RUN
if defined TL_ABORT goto SMOKEHALT
set TL_CUR=!TL_J2!
call :RUN
if defined TL_ABORT goto SMOKEHALT
set TL_CUR=!TL_J3!
call :RUN
if defined TL_ABORT goto SMOKEHALT
set TL_CUR=!TL_J4!
call :RUN
if defined TL_ABORT goto SMOKEHALT
set TL_CUR=!TL_J5!
call :RUN
if defined TL_ABORT goto SMOKEHALT
set TL_CUR=!TL_J6!
call :RUN
if defined TL_ABORT goto SMOKEHALT
set TL_CUR=!TL_J7!
call :RUN
if defined TL_ABORT goto SMOKEHALT
set TL_CUR=!TL_J8!
call :RUN

echo.
echo =================================================================
echo [queue] finished
date /t
time /t
echo.
echo   Logs: test_result for experiment runs, smoketest_logs for gates.
echo   Each batch printed its own WHAT TO RECORD block at the end. Read that
echo   before reading any number out of the run itself - several of these
echo   batches deliberately tell you which numbers are NOT to be trusted.
echo =================================================================
set TL_NOPAUSE=
if not defined TL_NOPAUSE pause
exit /b 0

REM ---------------------------------------------------------------- subroutines

:RESOLVE
REM  in: TL_CUR (an id)   out: TL_BAT (filename or empty) and TL_WHAT (label)
REM  ***Parenthesised blocks, not the ampersand form.*** In cmd,
REM      if COND set A=x^& set B=y
REM  runs the second set UNCONDITIONALLY - the ampersand splits at parser level.
set TL_BAT=
set TL_WHAT=
if "!TL_CUR!"=="0" (
  set TL_BAT=run_smoke_check.bat
  set TL_WHAT=smoke check - ABORTS the queue on failure
)
if "!TL_CUR!"=="1" (
  set TL_BAT=run_P022B_stage0_fp8_probe.bat
  set TL_WHAT=P022B stage 0 - FP8 numerical gate, minutes
)
if "!TL_CUR!"=="2" (
  set TL_BAT=run_P042_stage1_speed_vram.bat
  set TL_WHAT=P042 stage 1 - speed and VRAM, 60 min, wants the machine alone
)
if "!TL_CUR!"=="3" (
  set TL_BAT=run_P042_stage2_nockpt.bat
  set TL_WHAT=P042 stage 2 - does --no-ckpt open, 20 min, an OOM is the answer
)
if "!TL_CUR!"=="4" (
  set TL_BAT=run_P050_parent_ablation.bat
  set TL_WHAT=P050 - parent dependency ablation, 9.5 h
)
if "!TL_CUR!"=="5" (
  set TL_BAT=run_P051_micro_group256.bat
  set TL_WHAT=P051 - ternary alpha group 256, 3.5 h
)
if "!TL_CUR!"=="6" (
  set TL_BAT=run_P046_emb_rank128.bat
  set TL_WHAT=P046 - embedding E=128 re-measured with SVD, 3.5 h
)
if "!TL_CUR!"=="7" (
  set TL_BAT=run_P052_seed_replicate.bat
  set TL_WHAT=P052 - second seed for the g16 verdict, 7 h
)
if "!TL_CUR!"=="8" (
  set TL_BAT=run_P018_compressed_teacher.bat
  set TL_WHAT=P018 - distil from mC_wsd instead of dense, 4.2 h
)
if "!TL_CUR!"=="9" (
  set TL_BAT=run_P022B_stage2_optstate.bat
  set TL_WHAT=P022B stage 2 - AdamW state in BF16, 4.2 h, babysit parts 1 and 2
)
exit /b 0

:SHOW
if not defined TL_CUR exit /b 0
call :RESOLVE
if not defined TL_BAT (
  echo   [SKIP] unknown id !TL_CUR!
  exit /b 0
)
if not exist !TL_BAT! (
  echo   [SKIP] !TL_BAT! is not on disk - already done and renamed?
  exit /b 0
)
echo   next: !TL_BAT!
echo         !TL_WHAT!
exit /b 0

:RUN
if not defined TL_CUR exit /b 0
call :RESOLVE
if not defined TL_BAT exit /b 0
if not exist !TL_BAT! (
  echo.
  echo [SKIP] !TL_BAT! is not on disk - continuing
  exit /b 0
)
echo.
echo =================================================================
echo [queue] starting !TL_BAT!
time /t
echo =================================================================
call !TL_BAT!
if errorlevel 1 goto RUNBAD
echo.
echo [queue] !TL_BAT! finished ok
time /t
exit /b 0

:RUNBAD
if "!TL_CUR!"=="0" set TL_ABORT=1
if defined TL_ABORT exit /b 0
echo.
echo [queue] [WARN] !TL_BAT! returned an error - continuing with the next slot
time /t
exit /b 0

:SMOKEHALT
echo.
echo [STOP] the smoke check failed. The tree is broken.
echo   Read the [VERIFY] lines in smoketest_logs and fix it before queuing again.
echo   ***Nothing else was run.*** That is the point - 2026-07-31 left every
echo   training run dead for hours because a report regression went unsmoked.
set TL_NOPAUSE=
if not defined TL_NOPAUSE pause
exit /b 3

:NOTHING
echo.
echo [queue] no ids entered. Nothing to do.
if not defined TL_NOPAUSE pause
exit /b 0

:ABORT
echo.
echo [queue] cancelled. Nothing was run.
if not defined TL_NOPAUSE pause
exit /b 0

:BADROOT
echo.
echo [STOP] run this from the TinyLM working folder (run100m.py not found).
if not defined TL_NOPAUSE pause
exit /b 9
