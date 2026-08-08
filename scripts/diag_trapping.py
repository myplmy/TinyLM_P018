#!/usr/bin/env python3
"""P036 단계1 — **weight trapping 이 우리 스케일에 실재하는가**를 학습 없이 관측한다.

★왜 (결과 016 §7.5·§8.6, 논문 arXiv:2601.07892 §3.2)
  Sherry 논문은 3:4 준정형 제약을 그냥 씌우면 **weight trapping** 이 난다고 명시한다:
      3:4 는 0 이 균일하게 흩어져 희소 행렬이 **dense binary 처럼** 행동한다
      → `∂L/∂X = (∂L/∂Y)(Tα)ᵀ` 가 **미분화되지 않는다**(gradient homogenization)
      → gradient 의 **유효랭크(ER)가 붕괴**(논문 Fig.4: ER < 750, 차원 4096인데도)
      → latent weight 가 **이진처럼 양극화**되어 삼진의 표현력을 못 쓴다
  대응책이 **Arenas**(`Y = XTα + λ_t·XW`)인데 **우리는 그걸 빼고 P016 을 측정했다.**

  ★그러나 논문은 1B/3B·10B 토큰이고 우리는 **132M·300M 토큰**(파라미터당 2.3 토큰)이다.
  **기전이 우리에게 없으면 대응책도 필요 없다.** 그래서 학습(4.5h+) 전에 이걸 먼저 본다.

★이 스크립트가 재는 것 (전부 기존 체크포인트로, 학습 없음)

  [1] latent weight 분포의 **양극화** — 논문 Fig.3/Fig.10 의 정성 관찰을 스칼라로
      · 그룹 스케일 α 로 정규화한 |w|/α 의 히스토그램
      · **골짜기 지표**: |w| < 0.5α 인 비율. 양극화되면 이 '가운데'가 비어간다
      · 이봉성(bimodality) = |w|/α 분포의 사분위 기반 지표

  [2] **삼진 값 점유율** {−α, 0, +α} — 0 이 비정상적으로 적으면 '이진화'다

  [3] ★**gradient 유효랭크(ER)** — 논문의 핵심 지표
      ER = exp(−Σ pᵢ log pᵢ),  pᵢ = σᵢ / Σσⱼ   (Roy & Vetterli 2007)
      TLinear 입력 활성의 **gradient** 행렬에 대해 계산한다. 낮을수록 균질화.

★판정
  `mA`(3:4) 만 [1] 골짜기가 깊고 [3] ER 이 낮으면 → **trapping 실재** → P036 단계2 진행
  세 모델이 비슷하면 → **우리 스케일엔 없다** → 단계2 보류, 3:4 비용은 구조 제약 그 자체

사용법 (GPU 를 아주 조금 쓴다 — 배치 4개 forward+backward)
  python scripts/diag_trapping.py
  python scripts/diag_trapping.py --models mA_g4s34_k4 mC_g8_k4 p6d --batches 4 --device cuda
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


def banner(s, ch="="):
    print("\n" + ch * 92)
    print(f"  {s}")
    print(ch * 92)


def _arch_of(tag):
    return "dense" if tag.startswith(("p6d", "dense", "p12d")) else "tied"


def effective_rank(M, max_rows=4096):
    """ER = exp(엔트로피(정규화 특이값)). 논문 Fig.4 와 같은 정의(Roy & Vetterli 2007).

    M: (N, d). N 이 크면 무작위 부분표본으로 자른다(SVD 비용 때문).
    ★해석 주의: ER 은 **차원 d 로 상한**된다. 절대값보다 **모델 간 비교**로 읽는다.
    """
    import torch
    if M.shape[0] > max_rows:
        idx = torch.randperm(M.shape[0])[:max_rows]
        M = M[idx]
    M = M.float()
    if M.abs().max() == 0:
        return 0.0, 0
    s = torch.linalg.svdvals(M)
    s = s[s > 0]
    p = s / s.sum()
    ent = -(p * p.log()).sum()
    return float(ent.exp()), int(M.shape[1])


def weight_stats(model, TLinear):
    """[1][2] latent 분포 양극화 + 삼진 점유율. 학습된 가중치만 읽는다."""
    import torch
    from tinylm.model.ternary import ternary as _ternary
    cfg = model.cfg
    g = cfg.micro_group
    tot = dict(n=0, valley=0, neg=0, zero=0, pos=0)
    q_all = []
    with torch.no_grad():
        for m in (mm for mm in model.modules() if isinstance(mm, TLinear)):
            w = m.weight.detach().float()
            O, I = w.shape
            wg = w.reshape(O, I // g, g)
            aw = wg.abs()
            # 그 층이 실제로 쓰는 마스크·alpha 로 정규화한다(= ternary() 와 같은 식)
            wq = _ternary(w, cfg).reshape(O, I // g, g)
            alpha = wq.abs().amax(dim=2, keepdim=True).clamp_min(1e-12)
            r = (aw / alpha).reshape(-1)
            tot["n"] += r.numel()
            tot["valley"] += int((r < 0.5).sum())          # 가운데가 비면 양극화
            q_all.append(r[torch.randperm(r.numel(), device=r.device)[:200_000]].cpu())
            z = wq.reshape(-1)
            tot["zero"] += int((z == 0).sum())
            tot["neg"] += int((z < 0).sum())
            tot["pos"] += int((z > 0).sum())
    r = torch.cat(q_all)
    # ★2026-07-31 수정 — `torch.quantile` 은 입력이 2^24(=16,777,216) 를 넘으면
    #   "input tensor is too large" 로 **죽는다**. dense(p6d)는 TLinear 가 많아
    #   층당 200k 표본만 모아도 30M 을 넘겼고, 그래서 결과 019 에서 **dense 대조군이
    #   통째로 빠졌다**(타잉 두 모델만 비교됨 = 3:4 귀속이 더 약해졌다).
    #   층별 표본을 이미 뽑았으므로 여기서 한 번 더 줄여도 분위 추정은 그대로다.
    QMAX = 8_000_000
    if r.numel() > QMAX:
        r = r[torch.randperm(r.numel())[:QMAX]]
    qs = torch.quantile(r, torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90]))
    return tot, [float(x) for x in qs]


def grad_effective_rank(model, cfg, loader, device, batches, TLinear, topk=6):
    """[3] TLinear 입력 활성의 gradient 에 대한 ER. 논문 Fig.4 대응.

    ★왜 '입력 활성의 gradient' 인가: 논문 식 (6) 이 `∂L/∂X` 를 지목한다.
      3:4 구조가 `(Tα)ᵀ` 를 균질화해 **앞 층으로 가는 신호가 미분화되지 않는 것**이 기전이다.
    """
    import torch
    import torch.nn.functional as F

    tls = [(n, m) for n, m in model.named_modules() if isinstance(m, TLinear)]
    # 중간층 MLP 쪽을 균등 간격으로 topk 개 고른다(전부 보면 느리고 정보도 중복)
    # ★2026-07-31 주의(결과 019): 균등 간격은 **모델마다 다른 층을 고른다**. g4 와 g8 은
    #   TLinear 개수가 달라 뽑히는 shape 집합이 달라졌고, 그래서 **raw 평균 ER 을 모델 간
    #   비교에 쓰면 안 된다**. 이름을 함께 돌려주어 무엇이 뽑혔는지 로그에 남긴다.
    picked = [tls[i] for i in
              (range(0, len(tls), max(1, len(tls) // topk)))][:topk]
    picks = [m for _n, m in picked]
    names = [n for n, _m in picked]
    store = {id(m): [] for m in picks}
    handles = []

    def mk_hook(m):
        def hook(mod, gin, gout):
            # gin[0] = ∂L/∂(입력). (B,T,d) → (B*T,d)
            if gin and gin[0] is not None:
                store[id(mod)].append(gin[0].detach().reshape(-1, gin[0].shape[-1]).float().cpu())
        return hook

    for m in picks:
        handles.append(m.register_full_backward_hook(mk_hook(m)))

    model.train()                     # dropout 없음. grad 경로만 살린다
    model.clear_quant()               # ★freeze 해제 — backward 에 STE 가 필요하다
    for _ in range(batches):
        x, y = loader()
        model.zero_grad(set_to_none=True)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        loss.backward()
    for h in handles:
        h.remove()
    model.zero_grad(set_to_none=True)

    out = []
    for i, m in enumerate(picks):
        chunks = store[id(m)]
        if not chunks:
            continue
        M = torch.cat(chunks, dim=0)
        er, d = effective_rank(M)
        out.append((names[i], tuple(m.weight.shape), er, d, er / max(d, 1)))
    return out


def main():
    ap = argparse.ArgumentParser(description="P036 단계1 weight trapping 기전 관측")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--data", default="ko-en")
    ap.add_argument("--tokens", default="300M")
    ap.add_argument("--preset", default="m100")
    ap.add_argument("--device", default=None, help="기본: cuda 있으면 cuda")
    ap.add_argument("--batches", type=int, default=4, help="ER 용 forward+backward 횟수")
    ap.add_argument("--micro-bs", type=int, default=2)
    ap.add_argument("--seq", type=int, default=512)
    a = ap.parse_args()

    import torch
    from tinylm import paths
    from tinylm.data import prepare, Loader
    from tinylm.infer.generate import load_model
    from tinylm.model.ternary import TLinear

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    models = [(t, _arch_of(t)) for t in a.models] if a.models else DEFAULT_MODELS
    base = f"{a.preset}_{a.data}_{a.tokens}"
    n_tok = int(float(a.tokens.rstrip("MmBb")) * (1e9 if a.tokens[-1] in "Bb" else 1e6))

    banner("P036 단계1 — weight trapping 기전 관측 (학습 없음)", "#")
    print(f"  device={dev}  ER 배치 {a.batches}회 x mb{a.micro_bs} x seq{a.seq} "
          f"= {a.batches*a.micro_bs*a.seq:,} 토큰만 흘린다")
    print("  ★논문(arXiv:2601.07892 Fig.3·4)의 두 지표를 우리 체크포인트에 적용한다.")
    print("  ★절대값이 아니라 **모델 간 비교**로 읽는다 — 논문과 스케일·데이터가 다르다.")

    meta = prepare(a.data, n_tok)
    rows = []
    for tag, arch in models:
        ck = paths.resolve_ckpt(a.preset, a.data, a.tokens, tag)
        if not ck.exists():
            print(f"\n  [건너뜀] 체크포인트 없음: {ck.name}")
            continue
        model, cfg, _ = load_model(arch=arch, ckpt_path=str(ck), device=dev)
        s34 = bool(getattr(cfg, "sparse34", False))
        banner(f"{tag}  (g={cfg.mlp_group if cfg.tie_mlp else 1}, cla={cfg.cla_group}, "
               f"sparse34={s34})")

        tot, qs = weight_stats(model, TLinear)
        n = tot["n"]
        valley = tot["valley"] / n
        zero = tot["zero"] / n
        print(f"\n  [1] latent 분포 양극화  (|w|/alpha 기준, alpha = 그룹 스케일)")
        print(f"      ★골짜기 지표 |w| < 0.5·alpha : {valley:6.2%}   "
              f"(작을수록 가운데가 비었다 = 양극화)")
        print(f"      분위 |w|/alpha : 10% {qs[0]:.3f}  25% {qs[1]:.3f}  "
              f"50% {qs[2]:.3f}  75% {qs[3]:.3f}  90% {qs[4]:.3f}")

        print(f"\n  [2] 삼진 값 점유율")
        print(f"      -alpha {tot['neg']/n:6.2%}   0 {zero:6.2%}   +alpha {tot['pos']/n:6.2%}")
        print(f"      (0 이 유난히 적으면 '이진화' — 삼진의 세 번째 상태를 안 쓰는 것)")

        va = Loader("val", a.micro_bs, a.seq, dev, meta["dir"], seed=99)
        ers = grad_effective_rank(model, cfg, va, dev, a.batches, TLinear)
        print(f"\n  [3] ★gradient 유효랭크 (dL/dX, 논문 Fig.4 대응)")
        print(f"      {'층(이름)':<28} {'shape':>16} {'ER':>9} {'차원':>6} {'ER/차원':>9}")
        for nm, shp, er, d, ratio in ers:
            print(f"      {nm[-28:]:<28} {str(shp):>16} {er:>9.1f} {d:>6} {ratio:>8.2%}")
        er_mean = sum(e[2] for e in ers) / max(len(ers), 1)
        ratio_mean = sum(e[4] for e in ers) / max(len(ers), 1)
        print(f"      평균 ER {er_mean:.1f}  평균 ER/차원 {ratio_mean:.2%}")
        print(f"      ★평균 ER 은 **모델 간 비교용이 아니다**(뽑힌 shape 이 다르다). "
              f"판정은 ER/차원 과 shape 일치 쌍으로 한다.")

        rows.append((tag, s34, valley, zero, er_mean, ratio_mean, ers))
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        return 2

    banner("★모델 간 비교 — 이것이 판정의 근거다", "#")
    print(f"  {'모델':<16}{'s34':>6}{'골짜기<0.5a':>12}{'0 비율':>9}{'평균ER(참고)':>13}{'ER/차원':>9}")
    print("  " + "-" * 68)
    for tag, s34, valley, zero, er, ratio, _e in rows:
        print(f"  {tag:<16}{str(s34):>6}{valley:>11.2%}{zero:>9.2%}{er:>12.1f}{ratio:>8.2%}")

    # ★'0 비율'은 3:4 모델에서 **측정값이 아니라 제약값**이다 — 4개마다 정확히 1개를 0 으로
    #   강제하므로 25.00% 가 구조적으로 확정된다. 지표로 읽으면 순환논증이 된다.
    if any(r[1] for r in rows):
        print("\n  ⚠️ '0 비율' 은 3:4 모델에서 **구조가 정한 값(25.00%)** 이다. 관측 지표가 아니므로")
        print("     '3:4 가 0 을 덜 쓴다'는 식으로 읽지 말 것 — 그건 3:4 의 정의 그 자체다.")

    # ★shape 일치 쌍 비교 — 뽑힌 층 집합이 모델마다 달라 raw 평균 ER 은 비교 불가다.
    if len(rows) >= 2:
        banner("shape 일치 층만 골라낸 ER/차원 비교 (raw 평균 ER 의 대체)")
        by_shape = {}
        for tag, s34, _v, _z, _e, _r, ers in rows:
            for _nm, shp, er, d, ratio in ers:
                by_shape.setdefault(shp, {}).setdefault(tag, []).append(ratio)
        shared = [s for s, d in by_shape.items() if len(d) == len(rows)]
        if not shared:
            print("  공통 shape 이 없다 — 이 체크포인트 조합으로는 ER 을 모델 간 비교할 수 없다.")
        else:
            tags = [r[0] for r in rows]
            print(f"  {'shape':>16} " + " ".join(f"{t:>14}" for t in tags))
            for s in sorted(shared, key=lambda x: -x[0]):
                cells = [sum(by_shape[s][t]) / len(by_shape[s][t]) for t in tags]
                print(f"  {str(s):>16} " + " ".join(f"{c:>13.2%}" for c in cells))
            print("  ★이 표만이 ER 의 모델 간 비교로 유효하다.")

    s34rows = [r for r in rows if r[1]]
    other = [r for r in rows if not r[1]]
    print()
    if s34rows and other:
        v_s, v_o = s34rows[0][2], sum(r[2] for r in other) / len(other)
        # ★판정은 raw 평균 ER 이 아니라 **차원 정규화한 ER/차원** 으로 한다(위 경고 참조).
        e_s, e_o = s34rows[0][5], sum(r[5] for r in other) / len(other)
        print(f"  3:4 모델 vs 비-3:4 평균")
        print(f"    골짜기 지표 {v_s:.2%} vs {v_o:.2%}  "
              f"({'3:4 가 더 양극화' if v_s < v_o else '차이 없음/반대'})")
        print(f"    ER/차원     {e_s:.2%} vs {e_o:.2%}  "
              f"({'3:4 가 더 낮음 = 균질화' if e_s < e_o * 0.9 else '차이 없음/반대'})")
        print()
        if v_s < v_o * 0.85 and e_s < e_o * 0.9:
            print("  → ★**trapping 실재.** 두 지표가 함께 3:4 를 지목한다. P036 단계2 진행.")
        elif e_s < e_o * 0.9 or v_s < v_o * 0.85:
            print("  → 한 지표만 지목한다. **약한 증거** — 단계2 를 돌릴지 사용자 판단.")
        else:
            print("  → **우리 스케일엔 trapping 이 관측되지 않는다.** 단계2 보류.")
            print("     3:4 의 품질 비용(결과 008)은 **구조 제약 그 자체**로 확정된다.")
    banner("한계")
    print("""  · ER 은 배치 몇 개의 gradient 표본이다. 논문의 학습 전 구간 추적과 다르다.
  · ER 은 차원으로 상한되므로 **절대값을 논문(4096차원)과 직접 비교하지 말 것**
    — 우리는 d=768 / ffn=2048 이다. 읽을 것은 **모델 간 비율**이다.
  · ★뽑는 층이 모델마다 다르다(TLinear 개수가 g 에 따라 달라 균등간격이 어긋난다).
    그래서 **raw 평균 ER 은 참고값**이고, 판정은 ER/차원 과 위의 shape 일치 표로 한다.
  · 이 세 모델은 **학습 조건이 조금씩 다르다**(g4/g8, s34 유무). 차이를 3:4 단독에
    귀속하려면 g 를 맞춘 대조가 필요하지만, 지금 있는 체크포인트로는 불가능하다.
    → **단계2 가 그 대조를 학습으로 만든다**(같은 조건에 Arenas 만 on/off).
  · 골짜기 지표의 임계(0.5·alpha)는 **우리가 정한 것**이지 논문 값이 아니다.
  · '0 비율' 은 3:4 에서 구조가 정한 25.00% — 관측 지표가 아니다.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
