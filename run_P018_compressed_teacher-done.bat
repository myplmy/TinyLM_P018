@echo off
setlocal enabledelayedexpansion
REM =============================================================================
REM  run_P018_compressed_teacher.bat -- [P018 reopened] distil from mC_wsd
REM                                     instead of the dense parent.
REM  Plan: test_plan\P018 section R (2026-08-07 rewrite - the old premise was wrong)
REM =============================================================================
REM
REM PRECONDITION: --kd-teacher-tag exists (implemented long ago, never used here).
REM   Needs runs\ckpt\m100R1c_ko-en_300M_mC_wsd.pt as BOTH the teacher and the
REM   control, and the dense parent for --init-from.
REM
REM ***ABOUT 4.2 GPU HOURS in four parts: minutes, 20 min, 20 min, 3.5 h.***
REM
REM ***THE OLD PLAN HAD TWO WRONG PREMISES. READ THIS.***
REM   X1  It said a compressed teacher makes the teacher forward '0.55x'.
REM       ***FALSE by our own measurements.*** Tying does not reduce compute -
REM       report() FLOPs has no mlp_group term and the layer count is unchanged.
REM       The measured effect is about -3.15 percent from cache locality
REM       (result 033). ***Do not sell this experiment as KD acceleration.***
REM   X2  It said the compressed teacher is a weaker teacher, so the student is
REM       capped by it. ***That direction is now doubtful.*** Result 021 showed
REM       mC beats p6d on a common val, and the teacher we actually use is not
REM       p6d - it is the 300M-POOL dense, and result 006 measured pool 2x as
REM       worth -0.12.
REM
REM ***SO THE REAL QUESTION FLIPPED***
REM   Not 'can we survive a weaker teacher' but
REM   ***'is the teacher we have been using worse than the student it made?'***
REM   Nobody has ever measured those two checkpoints on the SAME val set.
REM   Part [1/5] does exactly that, in minutes.
REM
REM WHAT IS ACTUALLY BOUGHT - VRAM, not speed
REM   teacher ternary latent  dense 472.5 MiB  to  mC_wsd 209.3 MiB
REM   dequant copy            dense 472.5 MiB  to  mC_wsd 209.3 MiB
REM   teacher residency       dense 945 MiB    to  mC_wsd 419 MiB = about -0.51 GB
REM   plus CLA2 halves the KV-owning layers, 20/20 -^> 10/20.
REM   ***This is P042 section 7 option B*** and P042 listed its risk as 'weaker
REM   teacher hurts quality' - which X2 may reverse.
REM
REM ***PREDICTIONS - WRITTEN BEFORE THE RUN (plan section R4)***
REM   R-P1 part [1/5]: mC_wsd BEATS the dense teacher on the common val by 0.05
REM        or more. ***With a caveat*** - dense trained on a 300M pool, so it
REM        carries a pool handicap. Say so in the result document.
REM   R-P2 student delta -0.02 to +0.03, i.e. same or slightly better.
REM   R-P3 teacher residency about -0.5 GB, measured.
REM   R-P4 wall clock -0 to -5 percent. ***Never quote 0.55x.***
REM   R-P5 --no-ckpt STILL does not open. 0.51 + 0.47 = 0.98 GB against the
REM        1.7 to 2.7 GB needed. Part [3/5] checks it anyway because it is cheap.
REM
REM ***SELF-DISTILLATION WARNING.*** mC_wsd was itself made by dense KD, so the
REM   dense teacher's training cost is already paid. The payoff is reuse across
REM   many future students, not this one run. Same architecture, same parent -
REM   ***do not generalise either direction of the result.***
REM
REM ***PART [4/5] MUST NOT USE --no-ckpt.*** The control mC_wsd ran with
REM   gradient checkpointing ON. Changing it would break the comparison (trap 2).
REM
REM ERRORLEVEL POLICY: part [4/5] is the prerequisite for [5/5]; the rest warn.
REM =============================================================================

if not exist run100m.py cd ..\..
if not exist run100m.py goto BADROOT
if not exist runs\ckpt\m100R1c_ko-en_300M_mC_wsd.pt goto NOCTRL
if not exist runs\ckpt\m100_ko-en_300M_dense.pt goto NOPARENT

echo =============================================================
echo [P018] distil from mC_wsd instead of the dense parent (about 4.2 h)
echo =============================================================
python scripts\runlog.py --name P018_compressed_teacher --note "[P018 reopened] compressed teacher. The old '0.55x forward' premise is false - tying does not reduce compute. This is a VRAM experiment."

echo.
echo [guard] cli.py call-keyword sanity
python scripts\check_call_kwargs.py
if errorlevel 1 goto BADKWARGS

echo.
echo =============================================================
echo [1/5] RQ0 - which of the two teacher candidates is better, on a common val
echo =============================================================
python scripts\runlog.py --name P018_compressed_teacher --note "[1/5] RQ0 : paired(dense, mC_wsd). ***Never measured before.*** Caveat: dense trained on a 300M pool."
python scripts\runlog.py --name P018_compressed_teacher -- python scripts\paired_eval.py --preset m100R1c --data ko-en --tokens 300M --models dense mC_wsd
if errorlevel 1 echo [WARN] RQ0 comparison failed - continuing

echo.
echo =============================================================
echo [2/5] RQ2 and RQ3 - 250 steps with the compressed teacher. VRAM and speed.
echo =============================================================
python scripts\runlog.py --name P018_compressed_teacher --note "[2/5] 250 steps, --kd-teacher-tag mC_wsd. ***Read VRAM and ms/step. DO NOT read val_loss.***"
python scripts\runlog.py --name P018_compressed_teacher -- python run100m.py train --preset m100R1c --arch tied --data ko-en --tokens 300M --pool-tokens 600M --exact-cache --steps 250 --micro-bs 8 --accum 16 --seq 1024 --lr 1e-3 --sched wsd --anneal-end 0.80 --decay-frac 0.2 --seed 1337 --eval-every 100 --compile --kd --kd-every 4 --init-from --kd-teacher-tag mC_wsd --tag mC_ct250
if errorlevel 1 echo [WARN] 250-step run failed - continuing

echo.
echo =============================================================
echo [3/5] bonus - does the smaller teacher open --no-ckpt (an OOM is the answer)
echo =============================================================
python scripts\runlog.py --name P018_compressed_teacher --note "[3/5] 250 steps, compressed teacher plus --no-ckpt. An OOM here is a RESULT, not a failure."
python scripts\runlog.py --name P018_compressed_teacher -- python run100m.py train --preset m100R1c --arch tied --data ko-en --tokens 300M --pool-tokens 600M --exact-cache --steps 250 --micro-bs 8 --accum 16 --seq 1024 --lr 1e-3 --sched wsd --anneal-end 0.80 --decay-frac 0.2 --seed 1337 --eval-every 100 --compile --kd --kd-every 4 --init-from --kd-teacher-tag mC_wsd --no-ckpt --tag mC_ct250_nockpt
if errorlevel 1 echo [WARN] did not finish - read the tail of the log, an OOM here is a RESULT

echo.
echo =============================================================
echo [4/5] RQ1 - full 2289 step run. ***NO --no-ckpt here, on purpose.***
echo =============================================================
python scripts\runlog.py --name P018_compressed_teacher --note "[4/5] full run : 2289 steps, --kd-teacher-tag mC_wsd. Exactly one flag differs from the control."
python scripts\runlog.py --name P018_compressed_teacher -- python run100m.py train --preset m100R1c --arch tied --data ko-en --tokens 300M --pool-tokens 600M --exact-cache --steps 2289 --micro-bs 8 --accum 16 --seq 1024 --lr 1e-3 --sched wsd --anneal-end 0.80 --decay-frac 0.2 --seed 1337 --eval-every 100 --compile --kd --kd-every 4 --init-from --kd-teacher-tag mC_wsd --tag mC_ct
if errorlevel 1 goto TRAINBAD

echo.
echo =============================================================
echo [5/5] paired comparison against mC_wsd
echo =============================================================
python scripts\runlog.py --name P018_compressed_teacher --note "[5/5] paired comparison against mC_wsd (which is also the teacher - self-distillation)"
python scripts\runlog.py --name P018_compressed_teacher -- python scripts\paired_eval.py --preset m100R1c --data ko-en --tokens 300M --models mC_wsd mC_ct
if errorlevel 1 echo [WARN] paired_eval failed - the checkpoint is on disk, re-run this step alone

python scripts\runlog.py --name P018_compressed_teacher --note "=================================================================" "WHAT TO RECORD" "  0. ***the [kd] teacher load line.*** The path must contain mC_wsd. If it" "     says dense, the run retrained the control and 3.5 hours are gone." "  1. part [1/5] : the dense vs mC_wsd delta on the common val." "     ***Write the pool caveat next to it*** - dense trained on 300M, mC_wsd" "     on 600M, and result 006 priced that difference at -0.12." "  2. peak reserved VRAM in [2/5] against the stage-1 numbers of P042." "     Expect the teacher's share to drop by about 0.5 GB." "  3. steady state ms/step. ***Expect -0 to -5 percent, NOT 0.55x.***" "  4. did [3/5] finish or OOM, plus the peak either way." "  5. final val of [4/5], NOT best, plus the paired delta and t." "  6. grad_max from the json." "" "HOW TO READ IT" "  paired delta under 0.024" "      -^> ***the compressed teacher is a drop-in.*** Every future KD run gets" "         about 0.5 GB back for free, and P042 section 7 option B stops being" "         a fallback and becomes the default." "  delta 0.024 to 0.05" "      -^> a teacher ceiling exists. Report the size; it is the first direct" "         measurement of one in this repo." "  above 0.05" "      -^> reject. And check part [1/5] - if mC_wsd is the BETTER teacher and" "         the student still got worse, the cause is not teacher quality." "         ***That would be the interesting outcome***, pointing at diversity" "         or self-distillation collapse rather than capacity." "" "LIMITS: self-distillation - same architecture, same parent, one seed. Do not" "  generalise either way. Part [1/5] carries a pool handicap. --kd-every 2 was" "  deliberately NOT combined here (one variable at a time)." "================================================================="
echo done.
if not defined TL_NOPAUSE pause
exit /b 0

:BADKWARGS
echo.
echo [STOP] cli.py passes a keyword its target function does not accept.
if not defined TL_NOPAUSE pause
exit /b 6

:NOCTRL
echo.
echo [STOP] runs\ckpt\m100R1c_ko-en_300M_mC_wsd.pt is missing. It is both the teacher and the control.
if not defined TL_NOPAUSE pause
exit /b 5

:NOPARENT
echo.
echo [STOP] parent runs\ckpt\m100_ko-en_300M_dense.pt is missing (--init-from needs it).
if not defined TL_NOPAUSE pause
exit /b 5

:TRAINBAD
echo.
echo [STOP] the full run failed. Nothing to compare.
if not defined TL_NOPAUSE pause
exit /b 4

:BADROOT
echo.
echo [STOP] run this from the TinyLM working folder (run100m.py not found).
if not defined TL_NOPAUSE pause
exit /b 9
