# TinyLM P018-R1 agent instructions

This file applies to the whole repository. It is the root working context for Codex/GPT-5.6 Sol; `.codex/config.toml` selects that project-default model. More specific nested `AGENTS.md` files, if added later, may narrow these rules but must not weaken the execution-safety or research-integrity rules below.

## Current objective

Prepare and conduct P018-R1, which estimates how much independently trained MobileLLM-125M teachers and their students differ when the teacher uses MLP-only g6 parameter sharing, while measuring the resulting KD-teacher memory reduction.

Before planning or changing P018-R1 work, read in this order:

1. `P018-R1 작업지시서.md` — research question, hypotheses, controls, and required endpoints.
2. `docs/P018-R1_현황점검_20260821.md` — current implementation audit, gaps, and unresolved decisions.
3. `CLAUDE.md` — durable project conventions and user-execution policy.
4. `ai_dev_tool/03_실험착수_절차.md` and `ai_dev_tool/01_계측함정_원장.md` — preflight and measurement rules.
5. Relevant files under `test_plan/`, `test_result/`, and `handoff/` — historical context only; never treat old results or status as current without verification.

The `.claude/` configuration and skills are useful workflow references, but they are not a substitute for these Codex instructions.

## Execution boundary: user runs resource-intensive work

Codex/AI must **not** directly run model training, model inference, model construction, forward/backward passes, parity workloads, GPU/CPU benchmarks, data-tokenization jobs, or checkpoint loading/saving when they can consume material CPU, GPU, RAM, VRAM, disk, or time.

In particular, do not execute any of the following yourself:

- `test_mobile_make_reference.py`
- `test_mobile_weight_parity.py`
- `scripts/validation/validate_mobile_architecture.py --confirm-heavy`
- training, evaluation, generation, KD-cache, tokenizer/cache-preparation, profiling, or benchmark commands under `tinylm/` or `scripts/`

When such execution is required:

1. Finish all safe static checks first.
2. Give the user the exact command, working directory, Python executable/environment, expected log destination, resource expectation, and pass/fail criteria.
3. Ask the user to run it.
4. Analyze only the returned or saved log/artifact.

Allowed lightweight checks include reading/searching source and documents, read-only Git inspection, file metadata and hashes, `python --version`, importing packages only to print versions/device availability, and syntax/AST checks that do not instantiate models or load checkpoints/datasets.

## Python environment

- Named environment: `p018`
- Explicit interpreter: `W:\miniforge3\envs\p018\python.exe`
- Never assume shell activation succeeded; use the explicit interpreter in user-run commands.
- The 2026-08-21 re-check found Python 3.10.20, PyTorch 2.13.0+cu130, Transformers 4.41.2, CUDA available, and an NVIDIA GeForce RTX 4070 Ti SUPER. Treat this as a dated observation and re-check before execution planning.
- Do not install, upgrade, or replace packages without explicit user approval.

## Change scope and repository care

- The user authorized the minimum P018-R1 integration on 2026-08-18/21. Keep changes family-routed so legacy P018 presets and caches retain their existing defaults.
- `test_mobile.py` was authorized for removal after its useful checks were migrated to a clearly named manual validation script. Preserve `test_mobile_make_reference.py` and `test_mobile_weight_parity.py` unless separately authorized.
- Preserve existing P018 behavior while adding MobileLLM support. Avoid silently changing legacy presets, data caches, checkpoints, or result documents.
- `MobileLLM/` is an upstream submodule/reference implementation. Do not edit, update, clean, or commit inside it without explicit approval.
- Never delete or overwrite experiment logs, caches, checkpoints, reference artifacts, or user changes.
- Inspect the live Git status before work. Historical handoffs and logs are not evidence of the current worktree.

## P018-R1 scientific invariants

- Compare a dense MobileLLM-125M teacher with an MLP-tied teacher while holding hidden size, layer count, attention, vocabulary/tokenizer, activation, normalization, data/cache, token budget, optimizer/schedule, precision, and KD/student settings fixed.
- Train the dense and tied teachers independently under the same pre-registered budget. Do not convert a trained dense teacher into the tied arm, and do not post-hoc equalize consumed tokens.
- Use the same student architecture and paired seed/cache/evaluation protocol for both KD arms.
- Do not use a fixed equivalence margin for R1. Report raw paired differences, mean, standard deviation, confidence intervals, and effect sizes. A nonsignificant difference is not evidence of equality or equivalence.
- Use at least three seeds, paired evaluation on identical examples/tokens, and report mean, standard deviation, confidence intervals, and effect sizes.
- Report parameter count and unique parameter count separately. Report allocated and reserved GPU memory separately, plus latency/throughput and wall time; do not imply a speedup from parameter sharing without measurement.
- Old P018 dense-versus-compressed results used confounded data pools and are motivation only, not R1 evidence.
- Primary tied condition is MLP-only g6: 30 MobileLLM layers form five equal groups (L0-5, L6-11, L12-17, L18-23, L24-29). g10/g15 are optional later sharing-strength studies, not primary arms.
- Give every run a unique tag and isolated cache/output path. Save the exact command, commit/config/environment banner, stdout/stderr, and failure reason.

## Architecture and evidence boundaries

- Canonical implementation lives under `tinylm/`; `MobileLLM/` is the upstream comparison source.
- The saved 2026-08-09 parity log covers the dense model's selected eager CPU path after an explicit weight copy. It does not establish trainer, tokenizer/data, checkpoint, KD, tied-teacher, memory, throughput, or multi-seed readiness.
- `parity_reference/mobilellm_125m_state_dict.pt` is ignored by Git and currently lacks a recorded content hash/provenance chain. Do not assume it is the artifact exercised by `test_mobile_weight_parity.py`; the current script does not load it.

## Standard workflow

1. Re-read the work order and current audit.
2. Confirm unresolved protocol choices with the user.
3. Perform a static preflight and write an exact implementation/validation plan.
4. Make only the authorized code/document changes.
5. Run lightweight static validation only.
6. Hand resource-intensive commands to the user with explicit pass/fail gates and log paths.
7. Convert returned logs into evidence-backed result documents; distinguish observed facts, static estimates, and hypotheses.

Write user-facing explanations and experiment documents in Korean unless the user requests otherwise. Keep code identifiers and exact commands unchanged.
