#!/usr/bin/env python3
"""3:4 희소 삼진(`--sparse34`) 구현·회계 감사 — REVIEW1 승자 판정의 선결 진단.

★왜 필요한가 (사용자 지적, 2026-07-31)
  결과 016 에서 `mA(g4+s34)` 의 **상주** 메모리가 `mC(g8)` 보다 컸고, 결과 014 에서 `mA` 의
  CPU 추론이 dense 보다도 **느렸다**. 두 모델의 차이는 타잉 정도(g4/g8)와 **3:4 희소**다.
  그러므로 승자를 정하기 전에 **3:4 경로 자체에 구현·회계 문제가 있는지** 먼저 봐야 한다.

★이 스크립트가 검사하는 4가지

  [A] 유니크 파라미터 분해 — 상주 차이가 정말 g4/g8 때문인지 산수로 확인한다.
      (예측: 3:4 는 파라미터 **개수**를 1개도 줄이지 않는다. 상주 차이는 전부 타잉 몫이다)

  [B] 실제 비영(nonzero) 비율 — ★가장 중요.
      표준 경로는 TWN 임계(`|w| >= 0.7·mean|w|`)라 **희소율이 데이터에 따라 결정**된다.
      3:4 는 **정확히 25%** 만 0으로 만든다. 만약 표준 경로의 실제 0 비율이 25% 보다 **크면**,
      "3:4 희소"는 희소성을 **늘리는 게 아니라 줄이면서** 구조만 규칙적으로 만드는 것이다.
      그렇다면 이름과 달리 이 기법의 본질은 **희소화가 아니라 패킹 가능성**이고,
      품질 비용(결과 008 의 +0.035~0.056)의 출처도 "0이 많아서"가 아니라
      **"TWN 이 살렸을 큰 가중치를 블록 규칙 때문에 죽여서"** 가 된다.

  [C] 결정 뒤집힘 — TWN 이 살릴 것을 3:4 가 죽인 개수 / 그 반대. [B]의 기전 확인.

  [D] ★bpw 회계 규약 검산 — 저장 MB 의 근거.
      문서 근거: g128 삼진 = **1.71bpw**, 배포(GGUF) 밀도 **1.95** (`docs/custom_ternary_kernel.md`,
      `handoff/202607230703_HANDOFF.md`). 즉 1.95 = 코드 + **그룹 스케일** + **컨테이너 오버헤드**.
      그런데 sparse34 의 **1.25** 는 `C(4,3)·2^3 = 32 = 2^5` 라는 **순수 코드공간**이고
      그룹 스케일도 컨테이너도 포함하지 않는다.
      **분자와 분모가 다른 규약이면 감축비(2.64×)가 과대평가된다.** 이 스크립트가
      세 가지 회계로 저장 MB 를 다시 계산해 순위가 유지되는지 본다.

사용법 (GPU 불필요)
  python scripts/diag_sparse34.py
  python scripts/diag_sparse34.py --models mA_g4s34_k4 mC_g8_k4 p6d --layers 6
종료코드: 0 정상 / 2 체크포인트 없음
"""
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_MODELS = [("mA_g4s34_k4", "tied"), ("mC_g8_k4", "tied"), ("p6d", "dense")]

# --- 회계 규약 상수 (전부 문서 근거. 바꾸면 여기만 바꾼다) ---
SCALE_BITS = 16.0          # 그룹 스케일 alpha 를 fp16 로 저장한다고 볼 때
DOC_TERNARY_BPW = 1.71     # 문서의 g128 삼진 밀도(코드 + 스케일)
DOC_DEPLOY_BPW = 1.95      # 문서의 배포(GGUF) 밀도
S34_CODE_BPW = 1.25        # 5비트 / 4가중치 (무낭비 포화)


def banner(s, ch="="):
    print("\n" + ch * 92)
    print(f"  {s}")
    print(ch * 92)


def _arch_of(tag):
    return "dense" if tag.startswith(("p6d", "dense", "p12d")) else "tied"


def masks_for(w, group, thr_ratio):
    """같은 latent weight 에 대해 표준 TWN 마스크와 3:4 마스크를 **둘 다** 계산한다.

    `ternary.py:_TernarySTE.forward` 의 두 분기를 그대로 옮긴 것이 아니라 **같은 식**을 쓴다.
    (구현을 복제하지 말라는 규약은 '실행 경로'에 대한 것이고, 여기는 두 분기를 **동시에**
     비교해야 해서 한 함수 안에 둘 다 필요하다. 식이 어긋나면 이 진단이 무의미하므로
     ternary.py 를 고칠 때 여기도 함께 고친다.)
    """
    import torch
    O, I = w.shape
    G = I // group
    aw = w.reshape(O, G, group).abs()
    twn = (aw >= thr_ratio * aw.mean(dim=2, keepdim=True))
    b = aw.reshape(O, G, group // 4, 4)
    keep = torch.ones_like(b, dtype=torch.bool)
    keep.scatter_(3, b.argmin(dim=3, keepdim=True), False)
    s34 = keep.reshape(O, G, group)
    return aw, twn, s34


def recon_err(aw, sign, mask):
    """상대 재구성 오차 ||W - Ŵ||_F / ||W||_F. alpha 는 유지된 것들의 평균 |w| (TWN 식)."""
    import torch
    m = mask.to(aw.dtype)
    cnt = m.sum(dim=2, keepdim=True).clamp_min(1.0)
    alpha = (aw * m).sum(dim=2, keepdim=True) / cnt
    q = sign * m * alpha
    num = (aw * sign - q).pow(2).sum().sqrt()
    den = (aw * sign).pow(2).sum().sqrt().clamp_min(1e-12)
    return float(num / den), alpha


def main():
    ap = argparse.ArgumentParser(description="3:4 희소 삼진 구현·회계 감사")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--data", default="ko-en")
    ap.add_argument("--tokens", default="300M")
    ap.add_argument("--preset", default="m100")
    ap.add_argument("--layers", type=int, default=0,
                    help="층별 상세를 앞에서 N개만 출력(0=요약만)")
    a = ap.parse_args()

    import torch
    from tinylm import paths
    from tinylm.infer.generate import load_model
    from tinylm.model.ternary import TLinear

    models = [(t, _arch_of(t)) for t in a.models] if a.models else DEFAULT_MODELS
    base = f"{a.preset}_{a.data}_{a.tokens}"

    banner("3:4 희소 삼진 감사 — 구현 · 실제 희소율 · bpw 회계", "#")
    print("  전부 CPU. 학습·GPU 없음. 체크포인트의 latent weight 만 읽는다.")

    rows = []
    for tag, arch in models:
        ck = paths.RUNS / "ckpt" / f"{base}_{tag}.pt"
        if not ck.exists():
            print(f"\n  [건너뜀] 체크포인트 없음: {ck.name}")
            continue
        model, cfg, _ = load_model(arch=arch, ckpt_path=str(ck), device="cpu")
        group, thr = cfg.micro_group, cfg.twn_thr_ratio
        s34_on = bool(getattr(cfg, "sparse34", False))

        banner(f"{tag}  ({arch}, g={cfg.mlp_group if cfg.tie_mlp else 1}, "
               f"cla={cfg.cla_group}, sparse34={s34_on})")

        tl = [m for m in model.modules() if isinstance(m, TLinear)]
        n_tern = sum(m.weight.numel() for m in tl)

        tot = dict(n=0, twn_nz=0, s34_nz=0, both=0, twn_only=0, s34_only=0, neither=0)
        err_twn_w, err_s34_w = 0.0, 0.0
        shown = 0
        with torch.no_grad():
            for i, m in enumerate(tl):
                w = m.weight.detach().float()
                aw, twn, s34 = masks_for(w, group, thr)
                sign = torch.sign(w.reshape(*aw.shape))
                n = aw.numel()
                tw, sw = int(twn.sum()), int(s34.sum())
                both = int((twn & s34).sum())
                tot["n"] += n; tot["twn_nz"] += tw; tot["s34_nz"] += sw
                tot["both"] += both
                tot["twn_only"] += tw - both
                tot["s34_only"] += sw - both
                tot["neither"] += n - tw - sw + both
                e_t, _ = recon_err(aw, sign, twn)
                e_s, _ = recon_err(aw, sign, s34)
                err_twn_w += e_t * n; err_s34_w += e_s * n
                if a.layers and shown < a.layers:
                    print(f"    L{i:<3} {tuple(w.shape)!s:<16} "
                          f"TWN 비영 {tw/n*100:5.1f}%  3:4 비영 {sw/n*100:5.1f}%  "
                          f"재구성오차 TWN {e_t:.4f} / 3:4 {e_s:.4f}")
                    shown += 1

        n = tot["n"]
        twn_nz, s34_nz = tot["twn_nz"] / n, tot["s34_nz"] / n
        e_t, e_s = err_twn_w / n, err_s34_w / n

        print(f"\n  [A] 유니크 삼진 파라미터 : {n_tern/1e6:9.2f}M   "
              f"(상주 = 이 값 x 4B x 2벌 + 33MB — 결과 016)")
        print(f"      3:4 는 파라미터 **개수**를 줄이지 않는다 → 상주 차이는 전부 **타잉(g)** 몫이다.")

        print(f"\n  [B] 실제 비영 비율 (같은 latent weight 에 두 규칙을 적용)")
        print(f"      표준 TWN(|w| >= {thr}·mean|w|) : 비영 {twn_nz*100:5.2f}%  "
              f"(= 0 이 {100-twn_nz*100:5.2f}%)")
        print(f"      3:4 희소                       : 비영 {s34_nz*100:5.2f}%  "
              f"(= 0 이 {100-s34_nz*100:5.2f}%)  ← 항상 정확히 25%")
        if s34_nz > twn_nz:
            print(f"      ★3:4 가 표준보다 **비영이 많다**({s34_nz*100:.1f}% vs {twn_nz*100:.1f}%).")
            print(f"        즉 이 기법은 희소성을 **늘리는 것이 아니라 구조를 규칙화**하는 것이다.")
            print(f"        메모리 이득의 출처는 '0 이 많아서'가 아니라 '5비트 무낭비 패킹'이다.")
        else:
            print(f"      3:4 가 표준보다 희소하다 — 이 경우 이득의 일부는 실제 희소화에서 온다.")

        print(f"\n  [C] 결정 뒤집힘 (전체 {n/1e6:.1f}M 가중치 기준)")
        print(f"      둘 다 유지          {tot['both']/n*100:6.2f}%")
        print(f"      ★TWN 유지 → 3:4 제거 {tot['twn_only']/n*100:6.2f}%   "
              f"= 블록 규칙 때문에 죽인 **큰 가중치**. 품질 비용의 후보")
        print(f"      TWN 제거 → 3:4 유지 {tot['s34_only']/n*100:6.2f}%   "
              f"= 블록 규칙 때문에 살린 작은 가중치")
        print(f"      둘 다 제거          {tot['neither']/n*100:6.2f}%")
        print(f"      상대 재구성오차   TWN {e_t:.4f}  vs  3:4 {e_s:.4f}  "
              f"({'3:4 가 나쁨' if e_s > e_t else '3:4 가 좋음'}, 비 {e_s/max(e_t,1e-12):.3f}x)")

        # ---- [D] bpw 회계 ----
        p0_twn = 1.0 - twn_nz
        h = (-(p0_twn*math.log2(max(p0_twn, 1e-12)) + (1-p0_twn)*math.log2(max(1-p0_twn, 1e-12)))
             + (1 - p0_twn))                       # 0/비영 엔트로피 + 비영의 부호 1비트
        scale_bpw = SCALE_BITS / group
        rows.append(dict(tag=tag, n_tern=n_tern, s34=s34_on, twn_nz=twn_nz,
                         h=h, scale=scale_bpw, model=model, cfg=cfg))
        print(f"\n  [D] bpw 회계 (이 모델)")
        print(f"      측정 엔트로피(코드)  {h:.3f} bpw   "
              f"(0 비율 {p0_twn*100:.1f}% 기준. 문서 1.71 = 코드+스케일)")
        print(f"      그룹 스케일          {scale_bpw:.3f} bpw   (fp16 1개 / {group} 가중치)")
        print(f"      코드+스케일          {h+scale_bpw:.3f} bpw   "
              f"vs 문서값 {DOC_TERNARY_BPW}  → 컨테이너 오버헤드 "
              f"{DOC_DEPLOY_BPW-(h+scale_bpw):.3f} bpw")
        del model

    if not rows:
        return 2

    # ---- 회계 3안으로 저장 MB 재계산 ----
    banner("[D] ★저장 MB 재계산 — 회계 규약 3안. 순위가 유지되는가", "#")
    print("  현행(A)  : 삼진 1.95 / sparse34 1.25.  ← 분자·분모 규약이 다르다(이것이 문제)")
    print("  (B) 코드+스케일  : 표준=측정엔트로피+scale, 3:4=1.25+scale")
    print("  (C) B + 컨테이너 : (B) 에 현행 1.95 가 품고 있는 컨테이너 오버헤드를 **양쪽 다** 가산")
    ovh = None
    for r in rows:                                   # 컨테이너 오버헤드는 비-s34 모델에서 역산
        if not r["s34"]:
            ovh = DOC_DEPLOY_BPW - (r["h"] + r["scale"])
            break
    if ovh is None:
        ovh = DOC_DEPLOY_BPW - DOC_TERNARY_BPW
    print(f"  역산한 컨테이너 오버헤드 = {ovh:.3f} bpw (비-s34 모델 기준)\n")

    MB = lambda cnt, bpw: cnt * bpw / 8 / 1024 ** 2
    print(f"    {'모델':<16}{'유니크삼진':>11}{'현행A':>10}{'(B)':>10}{'(C)':>10}   bpw A / B / C")
    print("    " + "-" * 78)
    out = []
    for r in rows:
        cfg = r["cfg"]
        emb = MB(cfg.vocab_size * cfg.emb_rank + cfg.emb_rank * cfg.dim, DOC_DEPLOY_BPW)
        if r["s34"]:
            bA, bB = S34_CODE_BPW, S34_CODE_BPW + r["scale"]
        else:
            bA, bB = DOC_DEPLOY_BPW, r["h"] + r["scale"]
        bC = bB + ovh
        mA_, mB_, mC_ = (MB(r["n_tern"], b) + emb for b in (bA, bB, bC))
        out.append((r["tag"], mA_, mB_, mC_))
        print(f"    {r['tag']:<16}{r['n_tern']/1e6:>10.2f}M{mA_:>10.1f}{mB_:>10.1f}{mC_:>10.1f}"
              f"   {bA:.2f} / {bB:.2f} / {bC:.2f}")
    print("    (임베딩은 3:4 대상이 아니므로 세 안 모두 1.95bpw 로 동일하게 더했다)")

    print()
    for lbl, idx in (("현행 A ", 1), ("(B)    ", 2), ("(C)    ", 3)):
        order = sorted(out, key=lambda x: x[idx])
        print(f"  {lbl} 저장 작은 순: "
              + "  <  ".join(f"{o[0]}({o[idx]:.1f}MB)" for o in order))
    ranks = {lbl: [o[0] for o in sorted(out, key=lambda x: x[idx])]
             for lbl, idx in (("A", 1), ("B", 2), ("C", 3))}
    if len({tuple(v) for v in ranks.values()}) > 1:
        print("\n  ★회계 규약에 따라 **순위가 바뀐다** — 저장 MB 로 승자를 고른 근거가 규약 의존이다.")
    else:
        print("\n  세 회계 모두 같은 순위 — 저장 기준 순위는 규약에 견고하다(간격은 달라진다).")
    banner("읽는 법 / 한계")
    print("""  · [A] 가 g4/g8 로 상주 차이를 설명하면 **3:4 는 상주 문제의 원인이 아니다**.
  · [B] 에서 3:4 의 비영이 표준보다 많다면, 이 기법의 정체는 '희소화'가 아니라
    '패킹 가능한 구조 강제'다. 문서·리뷰의 표현을 그에 맞게 고쳐야 한다.
  · [C] 의 'TWN 유지 → 3:4 제거' 비율이 크면 그것이 결과 008 의 품질 비용 기전이다.
    (단 이 진단은 **상관**이지 인과가 아니다. 인과는 학습 런으로만 확인된다)
  · [D] 의 (C) 에서 순위가 뒤집히면 **REVIEW1 의 '메모리가 작은 쪽' 근거가 회계 규약에
    의존**했다는 뜻이다. (B)/(C) 의 컨테이너 오버헤드는 **역산 추정**이며, 실제 GGUF/
    커스텀 포맷이 3:4 코드를 어떻게 담는지는 **미검증**이다(P016 §한계와 같은 유보).
  · 이 스크립트는 **학습된 latent weight** 를 본다. mA 는 3:4 로 학습됐으므로 그 가중치에
    TWN 을 적용한 값은 '만약 TWN 이었다면'이 아니라 '지금 가중치에 TWN 을 씌우면'이다.
    두 모델의 절대 비교보다 **각 모델 안에서 두 규칙의 차이**를 읽는 것이 안전하다.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
