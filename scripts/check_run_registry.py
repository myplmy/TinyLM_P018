#!/usr/bin/env python3
"""런 레지스트리 점검 — 태그 충돌·조건 중복·필드 누락을 `runs/logs/*.json` 에서 찾는다.

`exp-preflight` 스킬의 기계 검사 부분. 사람이 `docs/EXPERIMENT_BASELINES.md` 를 읽는 것과
병행한다(문서는 판단 근거, 이 스크립트는 사실 조회).

★한계: 이 스크립트는 **이미 존재하는 로그만** 본다. 새 실험이 좋은 실험인지, 가설이 참인지는
전혀 말해주지 않는다. "중복 아님"은 "해볼 가치 있음"이 아니다.

사용법
  python scripts/check_run_registry.py                      # 전체 요약 + 충돌·중복 점검
  python scripts/check_run_registry.py --tag mA_g4s34_k4    # 이 태그가 이미 있는지
  python scripts/check_run_registry.py --plan "tied g8 sparse34 kd4 pool600 steps2289"
                                                            # 자유문 조건과 유사한 기존 런 찾기
  python scripts/check_run_registry.py --steps 2289 --micro-bs 8 --accum 16 --seq 1024
                                                            # 학습토큰 환산만
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "runs" / "logs"
CKPT = ROOT / "runs" / "ckpt"

# docs/EXPERIMENT_BASELINES.md §2.6 과 동기화해야 하는 목록.
PROTECTED = {"dense", "tied"}          # 정본 — 덮어쓰기 금지
RESERVED_PREFIX = {                    # 계획됐지만 아직 안 돈 접두사
    "qb_": "P026 cooldown-QAT",
    "mA_": "REVIEW1 후보 A", "mB_": "REVIEW1 후보 B", "mC_": "REVIEW1 후보 C",
    "nz_": "P021 추가검증 A 노이즈 캘리브레이션",
}
# 런 재구성에 필요한 필드(구 로그엔 없음 → 경고만)
NEEDED = ["pool_tokens", "mlp_group", "micro_bs", "accum", "deploy_mb", "kd_teacher"]

# 조건 동일성 판정에 쓰는 축. 여기 값이 모두 같으면 "같은 실험"이다.
IDENTITY = ["preset", "data", "arch", "mlp_group", "sparse34", "steps", "micro_bs", "accum",
            "seq", "lr", "sched", "anneal_end", "decay_frac", "pool_tokens", "kd", "kd_every",
            "kd_dynamic", "kd_teacher", "init_from_src", "ema", "lora_rank", "grad_ckpt"]


def load():
    if not LOGS.is_dir():
        print(f"[!] {LOGS} 없음 — 아직 학습 로그가 없습니다.")
        return []
    out = []
    for p in sorted(LOGS.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception as e:
            print(f"[!] {p.name} 파싱 실패: {e}")
            continue
        if "final" not in d:
            continue                     # lrfind 등
        d["_file"] = p.name
        d["_name"] = p.stem
        out.append(d)
    return out


def tokens_of(steps, mbs, accum, seq):
    return steps * mbs * accum * seq


def cmd_calc(a):
    eff = a.micro_bs * a.accum * a.seq
    tot = a.steps * eff
    print(f"유효배치 = {a.micro_bs} x {a.accum} x {a.seq} = {eff:,} 토큰/step")
    print(f"학습토큰 = {a.steps:,} step x {eff:,} = {tot/1e6:.1f}M")
    import math as _m
    pool = int(_m.ceil(tot * 2 / 1e6 / 100) * 100)      # 2배 미만이 되지 않게 100M 단위 올림
    print(f"권장 데이터 풀(2x 이상) = {tot*2/1e6:.0f}M  "
          f"--> --pool-tokens {pool}M --exact-cache  (기존 캐시 재사용 가능하면 600M 등 큰 쪽으로)")
    if a.accum == 8:
        print("[!] accum 8 은 CLI 기본값입니다. 의도한 값인지 확인하세요 "
              "(300M 라인 표준은 --accum 16).")
    return 0


def cmd_tag(runs, tag):
    hits = [r for r in runs if r.get("tag", "").endswith("_" + tag) or r["_name"].endswith("_" + tag)]
    if tag in PROTECTED:
        print(f"[STOP] '{tag}' 는 정본 태그입니다. 절대 이 이름으로 새 런을 돌리지 마세요.")
    for pre, why in RESERVED_PREFIX.items():
        if tag.startswith(pre) and not hits:
            print(f"[i] '{tag}' 는 예약 접두사 '{pre}' ({why}) 이고 아직 실행 기록이 없습니다 — 신규 실행 OK.")
    if not hits:
        print(f"[OK] '{tag}' 로 저장된 로그 없음 → 이름 충돌 없음.")
        # 체크포인트도 확인(로그 없이 ckpt 만 있는 경우)
        if CKPT.is_dir():
            cks = [p.name for p in CKPT.glob(f"*_{tag}.pt")] + [p.name for p in CKPT.glob(f"*_{tag}_best.pt")]
            if cks:
                print(f"[!] 그런데 체크포인트는 있습니다: {cks} — 중단된 런의 잔재일 수 있습니다.")
        return 0
    print(f"[STOP] '{tag}' 는 이미 사용 중입니다. 그대로 돌리면 **조용히 덮어씁니다**:")
    for r in hits:
        f = r["final"]
        print(f"    {r['_file']}  val {f['val_loss']:.4f}  steps {r['steps']}  "
              f"pool {r.get('pool_tokens')}  {r['wall_sec']/60:.0f}분")
    return 1


def _ident(r):
    return tuple((k, r.get(k)) for k in IDENTITY)


def cmd_audit(runs):
    print(f"로그 {len(runs)}건\n")

    print("== 1. 조건이 동일한 런(중복 학습 후보) ==")
    seen = {}
    dup = 0
    for r in runs:
        # 재구성 필드가 없는 구 로그는 동일성 판정 불가 → 제외
        if any(r.get(k) is None for k in ("micro_bs", "accum", "pool_tokens")):
            continue
        seen.setdefault(_ident(r), []).append(r)
    for _, group in seen.items():
        if len(group) > 1:
            dup += 1
            names = ", ".join(g["_name"] for g in group)
            vals = ", ".join(f"{g['final']['val_loss']:.4f}" for g in group)
            print(f"  [DUP] {names}\n         val: {vals}  <- 동일 조건. "
                  f"의도한 재현 실험이면 σ 측정에 쓸 수 있습니다.")
    if not dup:
        print("  없음 (또는 구 로그라 판정 불가 — 아래 3번 참조)")

    print("\n== 2. 같은 파일명으로 덮어쓸 위험 ==")
    stems = {}
    for r in runs:
        stems.setdefault(r["_name"], []).append(r["_file"])
    print("  로그 파일명은 유일합니다(중복 불가). 위험은 '앞으로' 같은 이름을 쓸 때 발생합니다.")
    print("  현재 사용중인 이름 수:", len(stems))
    prot = [r["_name"] for r in runs if r["_name"].split("_")[-1] in PROTECTED]
    if prot:
        print(f"  정본(덮어쓰기 금지): {prot}")

    print("\n== 3. 재구성 필드 누락(구 로그) ==")
    bad = []
    for r in runs:
        miss = [k for k in NEEDED if r.get(k) is None]
        if miss:
            bad.append((r["_name"], miss))
    if not bad:
        print("  없음 — 모든 로그가 재구성 가능합니다.")
    else:
        print(f"  {len(bad)}/{len(runs)}건. 이 런들은 조건을 로그만으로 복원할 수 없어 중복 판정에서 제외됩니다.")
        # 누락 패턴이 하나뿐이면(= v6 이전 로그 전체) 목록 대신 요약만.
        pats = {tuple(m) for _, m in bad}
        if len(pats) == 1:
            print(f"  전부 같은 패턴 누락: {','.join(next(iter(pats)))}")
            if len(bad) == len(runs):
                print("  → 즉 **모든 기존 로그**가 구 포맷입니다. 조건 대조는 반드시")
                print("     docs/EXPERIMENT_BASELINES.md §2 레지스트리(수동 기록)로 하세요.")
        else:
            for n, miss in bad[:20]:
                print(f"    {n:38} 누락: {','.join(miss)}")
            if len(bad) > 20:
                print(f"    ... 외 {len(bad)-20}건")
        print("  → 재학습하면 자동으로 기록됩니다. 과거 조건은 test_result 문서·배치파일로 추적하세요.")

    print("\n== 4. sparse34 런의 메모리 보고 ==")
    s34 = [r for r in runs if r.get("sparse34")]
    if not s34:
        print("  sparse34 런 없음")
    for r in s34:
        if r.get("deploy_mb"):
            print(f"  {r['_name']:38} deploy_mb {r['deploy_mb']:.2f}MB (정확)")
        else:
            approx = r["params"] * r.get("bpw", 1.95) / 8 / 1024 ** 2
            print(f"  {r['_name']:38} 근사 {approx:.2f}MB [!] 과소 — 정확값은 학습 로그 report() 블록")

    print("\n== 5. 안정성 점검 (grad_max, warmup 이후) ==")
    for r in sorted(runs, key=lambda x: -(x.get("grad_max") or 0))[:8]:
        gm, gp = r.get("grad_max"), r.get("grad_peak_warmup")
        if gm is None:
            continue
        flag = "  <-- 10 초과: 학습 문제 의심" if gm > 10 else ""
        print(f"  {r['_name']:38} grad_max {gm:7.2f}  warmup_peak {(gp or 0):7.2f}"
              f"  skip {r.get('n_skip')}{flag}")

    print("\n== 6. 풀 기록이 있는 런끼리만 격차 비교가 유효 ==")
    pools = {}
    for r in runs:
        pools.setdefault((r["data"], r.get("pool_tokens")), []).append(r["_name"])
    for (data, pool), names in sorted(pools.items(), key=lambda x: str(x[0])):
        tag = "미기록(구 로그)" if pool is None else f"{pool/1e6:.0f}M"
        print(f"  {data:10} 풀={tag:16} {len(names)}건")
    print("  → 풀이 다른 그룹끼리 val 을 직접 비교하면 무효입니다(결과 006).")
    return 0


def cmd_plan(runs, text):
    """자유문 조건과 토큰이 많이 겹치는 기존 런을 보여준다(느슨한 유사도)."""
    want = set(t.lower().strip("-") for t in text.replace(",", " ").split() if t)
    scored = []
    for r in runs:
        blob = " ".join(str(r.get(k)) for k in IDENTITY) + " " + r["_name"]
        blob = blob.lower().replace("_", " ")
        score = sum(1 for w in want if w in blob)
        if score:
            scored.append((score, r))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        print("[OK] 비슷한 기존 런을 찾지 못했습니다. (그래도 EXPERIMENT_BASELINES.md §2 를 눈으로 확인하세요)")
        return 0
    print("조건이 겹치는 기존 런(위가 더 유사):")
    for score, r in scored[:8]:
        f = r["final"]
        print(f"  [{score:2}] {r['_name']:38} val {f['val_loss']:.4f}  steps {r['steps']}  "
              f"g{r.get('mlp_group')} s34={int(bool(r.get('sparse34')))} "
              f"kd={r.get('kd_every') if r.get('kd') else '-'} pool={r.get('pool_tokens')}")
    print("\n→ 위와 조건이 완전히 같다면 새로 돌리지 말고 기존 로그를 쓰세요.")
    print("→ 일부만 같다면 '무엇이 다른지' 를 실험계획서에 명시하세요(그게 곧 독립변수입니다).")
    return 0


def main():
    ap = argparse.ArgumentParser(description="런 레지스트리 점검(exp-preflight 보조)")
    ap.add_argument("--tag", help="이 태그가 이미 쓰였는지 확인")
    ap.add_argument("--plan", help="자유문 조건과 유사한 기존 런 검색")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--seq", type=int, default=1024)
    a = ap.parse_args()

    if a.steps:
        return cmd_calc(a)
    runs = load()
    if a.tag:
        return cmd_tag(runs, a.tag)
    if a.plan:
        return cmd_plan(runs, a.plan)
    return cmd_audit(runs)


if __name__ == "__main__":
    sys.exit(main() or 0)
