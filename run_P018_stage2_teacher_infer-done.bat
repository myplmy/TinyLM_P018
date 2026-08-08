@echo off
setlocal enabledelayedexpansion
REM =============================================================================
REM  run_P018_stage2_teacher_infer.bat -- [P018 stage 2] compressed teacher
REM                                       PLUS --kd-teacher-infer. Never tried.
REM  Plan: test_plan\P018 section R9 / Results: 034 section 10, 037 section 5
REM =============================================================================
REM
REM PRECONDITION: both flags exist. Needs mC_wsd (teacher) and the dense parent.
REM
REM ***ABOUT 40 MINUTES. TWO RUNS OF 250 STEPS. RUN IT ALONE.***
REM
REM WHY - two levers that were only ever measured separately
REM   result 034: --kd-teacher-infer          reserved 12.47 -^> 11.45  (-1.03 GiB)
REM   result 037: --kd-teacher-tag mC_wsd     reserved 12.47 -^> 10.59  (-1.88 GiB)
REM   ***Nobody has run both at once.*** If they add:
REM     10.59 - 1.03 = about 9.6 GiB,  and with --no-ckpt (+3.9 to 4.4)
REM     about 13.5 to 14.0 GiB = ***2.0 to 2.5 GiB of headroom.***
REM   Compare result 034 section 10: --no-ckpt with the dense teacher needed
REM   15.37 GiB and left only 0.62 GiB - too thin to make standard.
REM
REM ***AND THERE IS A SECOND QUESTION, MAYBE THE MORE INTERESTING ONE***
REM   Result 037 section 5 found two --no-ckpt runs behaving in OPPOSITE ways:
REM     mC_ct250_nockpt   14.95 GiB   p90/p10 +48.5 pct  ***SPILLED***, +26.4 pct slower
REM     mC_tinf_nockpt    15.37 GiB   p90/p10  +7.4 pct  clean,          -20.5 pct faster
REM   ***The one with LESS memory spilled.*** Hypothesis: --kd-teacher-infer makes
REM   the teacher's allocation pattern STATIC (freeze_quant reuses one _wq,
REM   drop_latent removes the latent), while a compressed teacher without it
REM   rebuilds 6 unique MLP _wq tensors on every teacher forward.
REM   ***If that is right, run [2/2] below is clean and fast.*** If it still
REM   spills, the hypothesis is wrong and the cause is elsewhere - either answer
REM   is worth 40 minutes.
REM
REM ***PREDICTIONS - WRITTEN BEFORE THE RUN***
REM   Q1 [1/2] reserved lands in 9.4 to 9.9 GiB  -^ the two levers ADD.
REM      Above 10.3 -^ they overlap; both touch the teacher, so that is possible.
REM   Q2 [2/2] finishes AND check_spill.py says 'normal'.
REM      ***Finishing is not enough*** - result 037 finished and was still 26 pct
REM      slower. Read the spill verdict, not the exit code.
REM   Q3 steady state of [2/2] beats [1/2] by about 20 percent, matching the
REM      -20.5 percent that result 034 section 10 measured for --no-ckpt.
REM
REM ***SPILL POLICY: run this with system memory fallback DISABLED.***
REM   If fallback is on, question Q2 cannot be answered - the driver hides the
REM   very thing being measured. See docs\methods\09_training_memory.md section 4.
REM
REM ***THE 60 s WAIT BEFORE [2/2] IS DELIBERATE.*** Result 037 section 7.3:
REM   run_P018_compressed_teacher.bat started its --no-ckpt run 1 second after
REM   the previous one exited and died with CUBLAS_STATUS_EXECUTION_FAILED,
REM   while every batch that waited did not. Windows/WDDM holds VRAM for a few
REM   seconds after a process exits.
REM
REM ***DO NOT READ val_loss.*** 250 steps. VRAM and speed only.
REM =============================================================================

if not exist run100m.py cd ..\..
if not exist run100m.py goto BADROOT
if not exist runs\ckpt\m100R1c_ko-en_300M_mC_wsd.pt goto NOTEACHER
if not exist runs\ckpt\m100_ko-en_300M_dense.pt goto NOPARENT

echo =============================================================
echo [P018-2] compressed teacher + teacher inference mode (about 40 min, alone)
echo =============================================================
python scripts\runlog.py --name P018_stage2 --note "[P018 stage 2] --kd-teacher-tag mC_wsd PLUS --kd-teacher-infer. Never run together. Also tests the allocation-pattern hypothesis of result 037 section 5."

echo.
echo [guard] cli.py call-keyword sanity
python scripts\check_call_kwargs.py
if errorlevel 1 goto BADKWARGS

echo.
echo [idle] waiting 120 s so the GPU settles (result 016 section 12.5)
timeout /t 120 /nobreak

echo.
echo =============================================================
echo [1/2] both levers, gradient checkpointing ON
echo =============================================================
python scripts\runlog.py --name P018_stage2 --note "[1/2] compressed teacher + teacher infer mode, ckpt ON. Q1: does reserved land in 9.4-9.9 GiB"
python scripts\runlog.py --name P018_stage2 -- python run100m.py train --preset m100R1c --arch tied --data ko-en --tokens 300M --pool-tokens 600M --exact-cache --steps 250 --micro-bs 8 --accum 16 --seq 1024 --lr 1e-3 --sched wsd --anneal-end 0.80 --decay-frac 0.2 --seed 1337 --eval-every 100 --compile --kd --kd-every 4 --init-from --kd-teacher-tag mC_wsd --kd-teacher-infer --tag mC_cti250
if errorlevel 1 echo [WARN] run 1 failed - continuing

echo.
echo [idle] waiting 60 s. ***This wait is the fix for result 037 section 7.3.***
timeout /t 60 /nobreak

echo.
echo =============================================================
echo [2/2] both levers PLUS --no-ckpt
echo =============================================================
python scripts\runlog.py --name P018_stage2 --note "[2/2] the same plus --no-ckpt. Q2: finishes AND check_spill says normal. Finishing alone is NOT enough."
python scripts\runlog.py --name P018_stage2 -- python run100m.py train --preset m100R1c --arch tied --data ko-en --tokens 300M --pool-tokens 600M --exact-cache --steps 250 --micro-bs 8 --accum 16 --seq 1024 --lr 1e-3 --sched wsd --anneal-end 0.80 --decay-frac 0.2 --seed 1337 --eval-every 100 --compile --kd --kd-every 4 --init-from --kd-teacher-tag mC_wsd --kd-teacher-infer --no-ckpt --tag mC_cti250_nockpt
if errorlevel 1 echo [WARN] run 2 did not finish - an OOM or a CUBLAS error here is a RESULT

echo.
echo =============================================================
echo [check] spill verdict for both runs
echo =============================================================
python scripts\check_spill.py test_result\*P018_stage2*.txt

python scripts\runlog.py --name P018_stage2 --note "=================================================================" "WHAT TO RECORD" "  1. peak reserved for BOTH runs, from the [vram] lines." "  2. ***the check_spill verdict for run [2/2].***  Normal or suspected." "     ***Do not stop at 'it finished'*** - result 037 finished and was 26 percent" "     slower because it spilled quietly." "  3. steady state ms/step for both." "     (running_avg x N - step0) / (N - 1)   with N = 250" "  4. grad_max and skip from the json." "" "HOW TO READ IT" "  [1/2] reserved 9.4 to 9.9 GiB" "      -^> ***the two teacher levers ADD.*** Quote both numbers together." "  [1/2] above 10.3 GiB" "      -^> they overlap. Report the overlap size - it says what the teacher's" "         memory is actually made of." "  [2/2] finishes AND spill verdict is normal" "      -^> ***--no-ckpt becomes usable with real headroom.*** That is -20.5" "         percent on every KD run, and it makes EXPERIMENT_BASELINES section 5" "         revisable at last. It also CONFIRMS the allocation-pattern hypothesis." "  [2/2] finishes but spill is suspected" "      -^> headroom is still not enough. The hypothesis of result 037 section 5" "         is wrong or incomplete - say so, it is a real finding." "  [2/2] OOM or CUBLAS error" "      -^> record the peak and the requested allocation size. Cheap negative." "" "LIMITS: 250 steps, VRAM and speed only. ***Quality is not measured here.***" "  One seed. Spill policy must be DISABLED for [2/2] to mean anything." "================================================================="
echo done.
if not defined TL_NOPAUSE pause
exit /b 0

:BADKWARGS
echo.
echo [STOP] cli.py passes a keyword its target function does not accept.
if not defined TL_NOPAUSE pause
exit /b 6

:NOTEACHER
echo.
echo [STOP] runs\ckpt\m100R1c_ko-en_300M_mC_wsd.pt is missing (it is the teacher).
if not defined TL_NOPAUSE pause
exit /b 5

:NOPARENT
echo.
echo [STOP] parent runs\ckpt\m100_ko-en_300M_dense.pt is missing (--init-from needs it).
if not defined TL_NOPAUSE pause
exit /b 5

:BADROOT
echo.
echo [STOP] run this from the TinyLM working folder (run100m.py not found).
if not defined TL_NOPAUSE pause
exit /b 9
