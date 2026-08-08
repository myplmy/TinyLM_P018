"""자동 LR 탐색.

  기본(range test)  1회 짧은 런에서 LR을 지수적으로 올리며 발산 지점을 찾고,
                    그 지점의 1/safety 를 권장 LR로 제시한다. 빠르다(~수 분).
  고급(grid sweep)  여러 LR을 각각 짧게 돌려 |g|<thr & loss 단조감소하는 가장 큰
                    LR을 고른다. 사용자가 수동으로 하던 방식의 자동화.

발산은 삼진 어닐 이전(완전 FP) 구간에서 나므로, 두 방식 모두 anneal=0 으로 측정한다
(빠르고, 우리가 실제로 겪은 발산 조건과 일치).
"""
from __future__ import annotations

import json
import math

import torch
import torch.nn.functional as F

from .. import paths
from ..config import build_config
from ..model import TiedMLPTransformer
from ..data import prepare, Loader

LOGS = paths.RUNS / "logs"


def _make(preset, arch, seq, ckpt, device):
    cfg = build_config(preset, arch, seq, ckpt)
    model = TiedMLPTransformer(cfg).to(device)
    model.set_anneal(0.0)                      # 완전 FP — 원 발산 조건
    return cfg, model


def _step(model, opt, tr, accum, device, cfg):
    model.train()
    tot = 0.0
    for _ in range(accum):
        x, y = tr()
        with torch.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1)) / accum
        loss.backward()
        tot += loss.item()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    model.set_anneal(0.0)
    if torch.isfinite(gn):
        opt.step()
    opt.zero_grad(set_to_none=True); model.clear_quant()
    return tot, float(gn)


def range_test(preset="m100", arch="tied", data="ko-en", n_tokens="50M", *,
               micro_bs=8, seq=1024, accum=8, steps=150,
               lr_min=1e-5, lr_max=1e-1, grad_thr=10.0, diverge_mult=1.3,
               safety=3.0, ckpt=True, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    meta = prepare(data, int(float(str(n_tokens).rstrip("MmBb")) *
                             (1e9 if str(n_tokens)[-1] in "Bb" else 1e6)))
    torch.manual_seed(1337)
    cfg, model = _make(preset, arch, seq, ckpt, device)
    opt = torch.optim.AdamW(model.param_groups(lr_min), betas=(0.9, 0.95), eps=1e-8)
    tr = Loader("train", micro_bs, seq, device, meta["dir"], seed=1234)

    mult = (lr_max / lr_min) ** (1.0 / max(steps - 1, 1))
    lrs, losses, gnorms = [], [], []
    best = math.inf
    diverge_lr = None
    print(f"[lrfind:range] {arch} {preset}  lr {lr_min:.1e}->{lr_max:.1e}  {steps} step (anneal=0)")
    for s in range(steps):
        lr = lr_min * (mult ** s)
        for g in opt.param_groups:
            g["lr"] = lr          # range test는 그룹별 1/sqrt(g) 없이 동일 lr로 단순화
        loss, gn = _step(model, opt, tr, accum, device, cfg)
        lrs.append(lr); losses.append(loss); gnorms.append(gn)
        finite = math.isfinite(loss) and math.isfinite(gn)
        if finite:
            best = min(best, loss)
        if s % 10 == 0 or not finite:
            print(f"  s{s:>3}  lr {lr:.2e}  loss {loss:.3f}  |g| {gn:.2f}")
        if diverge_lr is None and (not finite or gn > grad_thr or
                                   (best < math.inf and loss > diverge_mult * best and s > 10)):
            diverge_lr = lr
            print(f"  -> 발산 감지 @ lr {lr:.2e} (loss {loss:.3f} |g| {gn:.2f})")
            break

    suggested = (diverge_lr / safety) if diverge_lr else lr_max / safety
    out = {"method": "range", "arch": arch, "preset": preset,
           "diverge_lr": diverge_lr, "suggested_lr": suggested,
           "lrs": lrs, "losses": losses, "gnorms": gnorms}
    print(f"[lrfind:range] 발산 lr={diverge_lr}  ->  권장 lr≈{suggested:.2e} "
          f"(발산의 1/{safety:g})")
    return out


def grid_sweep(preset="m100", arch="tied", data="ko-en", n_tokens="50M", *,
               lrs=(3e-4, 6e-4, 1e-3, 2e-3), micro_bs=8, seq=1024, accum=8,
               steps=60, grad_thr=10.0, ckpt=True, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    meta = prepare(data, int(float(str(n_tokens).rstrip("MmBb")) *
                             (1e9 if str(n_tokens)[-1] in "Bb" else 1e6)))
    print(f"[lrfind:grid] {arch} {preset}  lrs={list(lrs)}  각 {steps} step (anneal=0)")
    rows = []
    for lr in lrs:
        torch.manual_seed(1337)
        cfg, model = _make(preset, arch, seq, ckpt, device)
        opt = torch.optim.AdamW(model.param_groups(lr), betas=(0.9, 0.95), eps=1e-8)
        base = [g["lr"] for g in opt.param_groups]
        warm = max(5, steps // 5)     # 실제 학습처럼 warmup 램프 — 목표 LR을 cold로 때려
                                       # 오발산하는 것을 막는다(안 그러면 grid가 부당하게 가혹).
        tr = Loader("train", micro_bs, seq, device, meta["dir"], seed=1234)
        gmax, first, last, diverged = 0.0, None, None, False
        for s in range(steps):
            f = min(1.0, (s + 1) / warm)
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f
            loss, gn = _step(model, opt, tr, accum, device, cfg)
            if first is None:
                first = loss
            last = loss
            if not (math.isfinite(loss) and math.isfinite(gn)) or gn > grad_thr:
                diverged = True; break
            gmax = max(gmax, gn)
        drop = (first - last) if (first is not None and last is not None and not diverged) else None
        rows.append({"lr": lr, "diverged": diverged, "gmax": gmax,
                     "loss_first": first, "loss_last": last, "loss_drop": drop})
        tag = "발산" if diverged else f"loss {first:.2f}->{last:.2f}  |g|max {gmax:.2f}"
        print(f"  lr {lr:.2e}  {tag}")
    ok = [r for r in rows if not r["diverged"]]
    pick = max(ok, key=lambda r: r["lr"])["lr"] if ok else None
    print(f"[lrfind:grid] 권장 lr = {pick}  (발산 없이 가장 큰 LR)")
    return {"method": "grid", "arch": arch, "preset": preset, "rows": rows, "suggested_lr": pick}


def lr_find(method="range", *, preset="m100", arch="tied", data="ko-en", n_tokens="50M",
            micro_bs=8, seq=1024, accum=8, ckpt=True, device=None,
            range_steps=150, lr_min=1e-5, lr_max=1e-1,
            grid_lrs=(3e-4, 6e-4, 1e-3, 2e-3), grid_steps=60, grad_thr=10.0):
    """method: 'range'(기본) / 'grid'(고급) / 'both'. 함수별 인자를 명시적으로 라우팅한다."""
    LOGS.mkdir(parents=True, exist_ok=True)
    results = {}
    common = dict(preset=preset, arch=arch, data=data, n_tokens=n_tokens,
                  micro_bs=micro_bs, seq=seq, accum=accum, ckpt=ckpt, device=device)
    if method in ("range", "both"):
        results["range"] = range_test(steps=range_steps, lr_min=lr_min, lr_max=lr_max,
                                      grad_thr=grad_thr, **common)
    if method in ("grid", "both"):
        results["grid"] = grid_sweep(lrs=tuple(grid_lrs), steps=grid_steps,
                                     grad_thr=grad_thr, **common)
    (LOGS / "lrfind.json").write_text(json.dumps(results, indent=2, default=str))
    sug = (results.get("grid") or results.get("range") or {}).get("suggested_lr")
    print(f"\n[lrfind] 최종 권장 lr ≈ {sug}   (본 실행 --lr 로 사용)")
    return results
