"""비교 표. 기본은 dense vs tied(태그), --vs 주면 tied vs tied."""
from __future__ import annotations

import json

from .. import paths

LOGS = paths.RUNS / "logs"


def _resolve(base, kind, tagname=None):
    if kind == "dense":
        cands = [f"{base}_dense"] if base else ["dense"]
    elif tagname:
        cands = [f"{base}_{tagname}", tagname] if base else [tagname]
    else:
        cands = [f"{base}_tied"] if base else ["tied"]
    for c in cands:
        if (LOGS / f"{c}.json").exists():
            return c
    return cands[0]


def _deploy_mb(res):
    """배포 메모리(MB) — **정확값 우선**. `(값, 주석)` 반환.

    ★ 왜 이 함수가 필요한가(결과 008 §2-(6) 버그 수정): 예전 구현은 `전체 파라미터 × 단일 bpw` 로
    근사했다. 그런데 3:4 희소(P016)는 **삼진 가중치에만** 적용되고 임베딩은 대상이 아니다.
    그래서 sparse34 런만 메모리 과소(-0.8MB)·감축비율 과대(3.26× vs 실제 3.00×)로 출력됐다.

    이제 `trainer` 가 `report()` 와 같은 소스(`mem_breakdown`)로 `deploy_mb` 를 json 에 기록한다.
    그 값이 있으면 그대로 쓰고, **없으면(구 로그)** 근사로 되돌리되 sparse34 인 경우 경고를 붙인다.
    """
    # (2026-07-31) 회계 통일 후 신규 로그는 packed_mb(안 B) 를 쓴다.
    if res.get("packed_mb"):
        return float(res["packed_mb"]), ""
    if res.get("deploy_mb"):
        # 구 로그 — 규약이 다르다(삼진 1.95 / sparse34 1.25 = 혼합 회계).
        note = "  [구규약]" if res.get("sparse34") else ""
        return float(res["deploy_mb"]), note
    n, bpw = res["params"], res.get("bpw", 1.95)
    approx = n * bpw / 8 / 1024 ** 2
    if res.get("sparse34"):
        # 임베딩은 3:4 대상이 아니라 1.95bpw. 구 로그엔 분해가 없어 정확 복원 불가 → 명시 경고.
        return approx, "~(근사·과소)"
    return approx, "~"


def compare(base=None, tied_tag=None, vs_tag=None):
    if vs_tag:                                  # tied vs tied
        llabel = tied_tag or "tied"; lname = _resolve(base, "tied", tied_tag)
        rlabel = vs_tag;             rname = _resolve(base, "tied", vs_tag)
        verdict = False
    else:                                       # dense vs tied
        llabel = "dense"; lname = _resolve(base, "dense")
        rlabel = f"tied:{tied_tag}" if tied_tag else "tied"
        rname = _resolve(base, "tied", tied_tag)
        verdict = True

    out = {}
    for role, nm in (("L", lname), ("R", rname)):
        p = LOGS / f"{nm}.json"
        if not p.exists():
            print(f"[compare] {p} 없음. 먼저 학습하세요."); return
        out[role] = json.loads(p.read_text())
    d, t = out["L"], out["R"]
    dmem, dmnote = _deploy_mb(d)
    tmem, tmnote = _deploy_mb(t)
    dl, tl = d["final"]["val_loss"], t["final"]["val_loss"]

    # 긴 태그명이 컬럼(16)을 넘쳐 서로 붙는 문제 방지: base 프리픽스 제거 + 15자 절단(≥1칸 여백 보장).
    def _short(lab):
        lab = str(lab)
        if base and lab.startswith(f"{base}_"):
            lab = lab[len(base) + 1:]
        return lab if len(lab) <= 15 else lab[:13] + ".."
    llabel, rlabel = _short(llabel), _short(rlabel)

    print(f"  (좌={lname}.json  우={rname}.json)")
    print("=" * 68)
    print(f"  {llabel} 기준 vs {rlabel}   (동일 토큰·동일 시드)")
    print("=" * 68)
    print(f"  {'':<18}{llabel:>16}{rlabel:>16}{'차이(우-좌)':>14}")
    print("  " + "-" * 62)
    print(f"  {'파라미터':<18}{d['params']/1e6:>15.1f}M{t['params']/1e6:>15.1f}M"
          f"{d['params']/t['params']:>13.2f}x")
    print(f"  {'배포 메모리':<18}{dmem:>14.1f}MB{tmem:>14.1f}MB{dmem/tmem:>13.2f}x")
    if dmnote or tmnote:
        _who = " / ".join(f"{lab}{nt}" for lab, nt in ((llabel, dmnote), (rlabel, tmnote)) if nt)
        print(f"  {'':<18}!! 근사값 포함 ({_who}) — 구 로그라 deploy_mb 미기록.")
        if "과소" in dmnote + tmnote:
            print(f"  {'':<18}   sparse34 는 임베딩까지 1.25bpw 로 세어 **과소·비율 과대**다.")
        print(f"  {'':<18}   정확값 = 학습 로그 상단 report() 블록. 재학습 시 자동 기록.")
    print(f"  {'val loss':<18}{dl:>16.4f}{tl:>16.4f}{tl-dl:>+14.4f}")
    print(f"  {'perplexity':<18}{d['final']['ppl']:>16.2f}{t['final']['ppl']:>16.2f}"
          f"{t['final']['ppl']/d['final']['ppl']:>13.2f}x")
    print(f"  {'학습 시간(분)':<18}{d['wall_sec']/60:>16.1f}{t['wall_sec']/60:>16.1f}"
          f"{d['wall_sec']/t['wall_sec']:>13.2f}x")
    print("  " + "-" * 62)
    gap = tl - dl
    for role, lab in (("L", llabel), ("R", rlabel)):
        if out[role].get("grad_max"):
            print(f"  {'최대 |g| ('+lab+')':<20}{out[role]['grad_max']:>14.1f}"
                  f"{'  (>10이면 불안정)' if out[role]['grad_max'] > 10 else ''}")

    if out["R"].get("data") == "synthetic":
        print(f"\n  손실 격차 {gap:+.4f}")
        print("  !! 합성 데이터는 품질 판정에 쓸 수 없습니다(파이프라인 전용).")
        print("=" * 68); return

    if verdict:
        mag = "작음" if gap <= 0.07 else ("보통" if gap <= 0.15 else "큼")
        print(f"\n  손실 격차 {gap:+.4f}  ({mag})")
    else:
        print(f"\n  손실 차이(우-좌) {gap:+.4f}   ({rlabel}가 {llabel}보다 "
              f"{'낮음(좋음)' if gap < 0 else '높음'})")
    print("=" * 68)
