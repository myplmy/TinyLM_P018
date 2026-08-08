#!/usr/bin/env python3
"""스모크 결과 검증 — 신규 계측 필드가 실제로 기록됐는지 json 으로 확인한다.

왜: 필드를 추가해놓고 "기록되겠지" 하고 긴 런을 돌린 뒤에야 비어 있는 걸 발견하면
    그 GPU 시간이 통째로 낭비된다. 스모크(수 분)로 먼저 계약을 검증한다.
    (P021B 에서 nvidia-smi VRAM 을 사람이 적기로 했다가 통째로 빠뜨린 것이 계기)

검사
  1. 필수 필드 존재 — seed·micro_bs·accum·pool_tokens·mlp_group·grad_ckpt·deploy_mb·
     vram_reserved_gb·tokens_per_microbatch ...
  2. 값 정합 — eff_batch == micro_bs*accum*seq, tokens == steps*eff_batch,
     tokens_per_microbatch == micro_bs*seq, deploy_mb == parts 합
  3. 플래그 반영 — --seed/--sparse34/--anneal-end 가 실제로 json 에 반영됐는지
  4. VRAM 이 0 이 아닌지(cuda 런일 때)

사용법
  python scripts/check_smoke.py                 # tiny_synthetic_* 전부
  python scripts/check_smoke.py --tag sm_seed
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "runs" / "logs"

REQUIRED = ["seed", "micro_bs", "accum", "eff_batch", "pool_tokens", "exact_cache",
            "mlp_group", "grad_ckpt", "kd_teacher", "init_from_src", "anneal_end",
            "decay_frac", "deploy_mb", "mem_parts_mb", "mem_params",
            "vram_alloc_gb", "vram_reserved_gb", "tokens_per_microbatch",
            "sparse34", "bpw", "grad_max", "grad_peak_warmup"]


def check(name, d, expect=None):
    errs, warns, oks = [], [], []

    # 1. 필드 존재 (None 허용 필드는 키 존재만 확인)
    nullable = {"pool_tokens", "kd_teacher", "init_from_src"}
    for k in REQUIRED:
        if k not in d:
            errs.append(f"필드 누락: {k}")
        elif d[k] is None and k not in nullable:
            errs.append(f"필드가 None: {k}")
    if not errs:
        oks.append(f"필수 필드 {len(REQUIRED)}개 모두 존재")

    # 2. 정합성
    if all(k in d for k in ("micro_bs", "accum", "seq", "eff_batch")):
        want = d["micro_bs"] * d["accum"] * d["seq"]
        (oks if d["eff_batch"] == want else errs).append(
            f"eff_batch {d['eff_batch']} == mb*accum*seq {want}")
    if all(k in d for k in ("steps", "eff_batch", "tokens")):
        want = d["steps"] * d["eff_batch"]
        (oks if d["tokens"] == want else errs).append(
            f"tokens {d['tokens']} == steps*eff_batch {want}")
    if all(k in d for k in ("micro_bs", "seq", "tokens_per_microbatch")):
        want = d["micro_bs"] * d["seq"]
        (oks if d["tokens_per_microbatch"] == want else errs).append(
            f"M(tokens_per_microbatch) {d.get('tokens_per_microbatch')} == mb*seq {want}")
    if "mem_parts_mb" in d and d.get("deploy_mb"):
        tot = sum(d["mem_parts_mb"].values())
        (oks if abs(tot - d["deploy_mb"]) < 1e-6 else errs).append(
            f"deploy_mb {d['deploy_mb']:.4f} == parts 합 {tot:.4f}")

    # 3. VRAM
    if d.get("vram_reserved_gb") is not None:
        v = d["vram_reserved_gb"]
        if v <= 0:
            errs.append(f"vram_reserved_gb 가 {v} (0 이하)")
        else:
            oks.append(f"VRAM peak reserved {v:.3f}GB (alloc {d.get('vram_alloc_gb', 0):.3f}GB)")
    else:
        warns.append("VRAM 미기록 — CPU 런이면 정상, CUDA 런이면 버그")

    # 4. 기대값(플래그가 반영됐는지)
    for k, want in (expect or {}).items():
        got = d.get(k)
        (oks if got == want else errs).append(f"{k} = {got!r} (기대 {want!r})")

    mark = "FAIL" if errs else ("WARN" if warns else "OK")
    print(f"\n=== {name}  [{mark}]")
    for m in errs:
        print(f"  [E] {m}")
    for m in warns:
        print(f"  [W] {m}")
    for m in oks:
        print(f"  [ok] {m}")
    return len(errs)


EXPECT = {   # 태그 접미사 -> 그 런이 반드시 만족해야 하는 값
    "sm_base":   {"seed": 1337, "sparse34": False, "anneal_end": 0.60, "grad_ckpt": True},
    "sm_seed":   {"seed": 4242},
    "sm_s34":    {"sparse34": True, "bpw": 1.25},
    "sm_sched":  {"anneal_end": 0.80, "sched": "wsd", "decay_frac": 0.2},
    "sm_nockpt": {"grad_ckpt": False},
    "sm_kd":     {"kd": True, "init_from": True, "kd_every": 4},
}


def main():
    ap = argparse.ArgumentParser(description="스모크 결과 계측필드 검증")
    ap.add_argument("--tag")
    a = ap.parse_args()
    files = sorted(LOGS.glob(f"*{a.tag}.json")) if a.tag else sorted(LOGS.glob("tiny_*sm_*.json"))
    if not files:
        print(f"[!] 검사할 스모크 로그가 없습니다({LOGS}). 먼저 run_smoke.bat 을 돌리세요.")
        return 1
    total = 0
    for p in files:
        d = json.loads(p.read_text())
        exp = next((v for k, v in EXPECT.items() if p.stem.endswith(k)), None)
        total += check(p.stem, d, exp)
    print(f"\n{'='*60}\n총 에러 {total}건 — 0 이면 계측 계약이 지켜지고 있습니다.")
    print("주의: 이 검사는 '필드가 기록되는가'만 본다. 값이 물리적으로 옳은지(예: VRAM 절대값)는")
    print("      실제 장기 런과 nvidia-smi 로 1회 대조해야 한다.")
    return total


if __name__ == "__main__":
    sys.exit(main())
