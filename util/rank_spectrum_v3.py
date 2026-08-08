"""rank_spectrum_v3 — 널 대조군이 있는 랭크 스펙트럼 측정.

v2의 치명적 결함: 활성 가중(--calib)을 켜면 수치가 크게 오르는데, 그 상승분의
대부분이 '레이어가 서로 비슷해서'가 아니라 '활성 스케일이 몇 채널에 몰려서'다.

  구조가 전혀 없는 랜덤 행렬 4개, r=16 포착률
    가중 없음                    6%
    완만한 가중 (18x)           13%
    LLM 유사 가중 (heavy)       81%   <-- 구조가 0인데 81%
    16채널만 100배 (spiky)      99%

따라서 활성 가중 수치는 반드시 **널 대조군과의 차이**로 읽어야 한다.
v3은 동일한 S, 동일한 그룹 구성에서 잔차의 저랭크 구조만 파괴한 대조군을
같이 계산하고, 그 차이(excess)를 보고한다.

  excess = measured - null
    excess가 10%p 미만  -> 레이어 유사성 신호 없음. delta 무의미.
    excess가 30%p 이상  -> 진짜 구조가 있다. delta 유효.

사용:
  python rank_spectrum_v3.py Qwen/Qwen3-1.7B --blocks 7
  python rank_spectrum_v3.py Qwen/Qwen3-1.7B --blocks 7 --calib
"""
from __future__ import annotations
import argparse, torch


def null_residual(R: torch.Tensor) -> torch.Tensor:
    """잔차의 저랭크 구조만 파괴하고 열 노름은 보존한다.

    각 열의 원소를 열 안에서 무작위 셔플 -> ||column||은 그대로이므로
    diag(S)와의 상호작용(=활성 가중이 만드는 편중)은 완전히 동일하게 유지되고,
    열들 사이의 정렬(=저랭크성)만 사라진다. 이것이 올바른 귀무가설이다.
    """
    out = torch.empty_like(R)
    for i in range(R.shape[0]):
        for j in range(R.shape[2]):
            out[i, :, j] = R[i, torch.randperm(R.shape[1]), j]
    return out


def fit_lowrank(R, rank, S):
    low = []
    Sinv = torch.linalg.inv(S) if S is not None else None
    for r_ in R:
        M = r_ @ S if S is not None else r_
        U, s, Vh = torch.linalg.svd(M, full_matrices=False)
        a = (U[:, :rank] * s[:rank]) @ Vh[:rank]
        low.append(a @ Sinv if S is not None else a)
    return torch.stack(low)


def solve(Ws, rank, mode, S, iters=10):
    W = torch.stack(Ws)
    base = W.mean(0)
    for _ in range(iters if mode == "als" else 1):
        low = fit_lowrank(W - base, rank, S)
        if mode == "mean":
            break
        base = (W - low).mean(0)
    low = fit_lowrank(W - base, rank, S)     # base와 low를 항상 정합시킨다
    return base, low


def captured(R, low, S):
    A = R @ S if S is not None else R
    B = (R - low) @ S if S is not None else (R - low)
    num = (A**2).sum() - (B**2).sum()
    return (num / (A**2).sum().clamp_min(1e-12)).item()


def measure(Ws, rank, mode, S):
    """-> (실측 포착률, 널 포착률, 차이)"""
    base, low = solve(Ws, rank, mode, S)
    R = torch.stack(Ws) - base
    real = captured(R, low, S)

    Rn = null_residual(R)                     # 구조만 제거, 열 노름 보존
    null = captured(Rn, fit_lowrank(Rn, rank, S), S)
    return real, null, real - null


def cluster_layers(mats, n_blocks, mode):
    L = len(mats)
    if mode == "adjacent":
        g = L // n_blocks
        return [list(range(b*g, min((b+1)*g, L))) for b in range(n_blocks)]
    X = torch.stack([m.flatten() for m in mats])
    X = X / X.norm(dim=1, keepdim=True)
    cen = [0]
    for _ in range(n_blocks - 1):
        d = torch.stack([1 - X @ X[c] for c in cen]).min(0).values
        cen.append(int(d.argmax()))
    C = X[cen].clone()
    for _ in range(50):
        a = (X @ C.T).argmax(1)
        for k in range(n_blocks):
            if (a == k).any():
                C[k] = torch.nn.functional.normalize(X[a == k].mean(0), dim=0)
    return [[i for i in range(L) if a[i] == k] for k in range(n_blocks) if (a == k).any()]


def activation_scales(model, tok, n_batches=32, seqlen=1024):
    """채널별 RMS. v2의 torch.maximum(=배치 최댓값)은 아웃라이어를 과대평가하므로
    제곱 누적 평균으로 바꿨다."""
    from datasets import load_dataset
    acc, cnt, hooks = {}, {}, []
    def hook(nm):
        def f(mod, inp, out):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            acc[nm] = acc.get(nm, 0) + x.pow(2).sum(0)
            cnt[nm] = cnt.get(nm, 0) + x.shape[0]
        return f
    for nm, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and "layers" in nm:
            hooks.append(m.register_forward_hook(hook(nm)))
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    buf, used = "", 0
    with torch.no_grad():
        for row in ds:                        # 짧은 행이 많으므로 이어붙여서 채운다
            buf += row["text"]
            if len(buf) < seqlen * 4:
                continue
            ids = tok(buf[:seqlen*4], return_tensors="pt",
                      truncation=True, max_length=seqlen).input_ids
            buf = ""
            if ids.shape[1] > 32:
                model(ids); used += 1
            if used >= n_batches:
                break
    for h in hooks:
        h.remove()
    return {k: (acc[k]/cnt[k]).sqrt() for k in acc}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_id"); p.add_argument("--blocks", type=int, default=7)
    p.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--calib", action="store_true")
    p.add_argument("--clip", type=float, default=10.0,
                   help="활성 스케일 동적범위 상한. 무제한이면 지표가 포화해 정보가 사라진다.")
    a = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(a.model_id, torch_dtype=torch.float32)
    layers = model.model.layers
    names = [n for n, q in layers[0].named_parameters() if q.dim() == 2]
    print(f"{a.model_id}: {len(layers)}층 -> {a.blocks}블록"
          f"{'  (활성 가중)' if a.calib else '  (Frobenius)'}\n")

    act = {}
    if a.calib:
        tok = AutoTokenizer.from_pretrained(a.model_id)
        act = activation_scales(model, tok)
        print(f"활성 스케일 수집 완료 ({len(act)}개 선형층)\n")

    # 먼저 레이어끼리 실제로 얼마나 비슷한지부터 본다
    print("[레이어 쌍별 코사인 유사도 — 여기에 신호가 없으면 그룹 구성은 무의미]")
    for nm in names:
        Ws = torch.stack([dict(l.named_parameters())[nm].data.flatten() for l in layers])
        Ws = Ws / Ws.norm(dim=1, keepdim=True)
        sim = (Ws @ Ws.T).fill_diagonal_(0)
        adj = torch.tensor([sim[i, i+1] for i in range(len(layers)-1)])
        print(f"  {nm[:26]:<28} 전체평균 {sim.sum()/(len(layers)**2-len(layers)):+.3f}  "
              f"인접평균 {adj.mean():+.3f}  최대 {sim.max():+.3f}")
    print()

    for nm in names:
        Ws_all = [dict(l.named_parameters())[nm].data for l in layers]
        S = None
        if a.calib:
            key = [k for k in act if k.endswith(nm.replace(".weight", ""))]
            if key:
                v = torch.stack([act[k] for k in key]).mean(0).clamp_min(1e-8)
                v = v / v.median()
                S = torch.diag(v.clamp(1.0/a.clip, a.clip))   # 동적범위 제한
        print(f"[{nm}]" + (f"  활성 동적범위 {(S.diag().max()/S.diag().min()).item():.0f}x"
                           if S is not None else ""))
        print(f"  {'그룹':<11}{'base':<6}" +
              "".join(f"{f'r={r} 실측/널/차이':>22}" for r in a.ranks))
        for gmode in ("adjacent", "similarity"):
            groups = cluster_layers(Ws_all, a.blocks, gmode)
            for bmode in ("mean", "als"):
                cells = []
                for r in a.ranks:
                    rs, ns, es = [], [], []
                    for g in groups:
                        if len(g) < 2:
                            continue
                        x, y, z = measure([Ws_all[i] for i in g], r, bmode, S)
                        rs.append(x); ns.append(y); es.append(z)
                    n = max(len(rs), 1)
                    cells.append(f"{sum(rs)/n:>6.0%}/{sum(ns)/n:>4.0%}/{sum(es)/n:>+6.0%}")
                print(f"  {gmode:<11}{bmode:<6}" + "".join(f"{c:>22}" for c in cells))
        print()
    print("판정: '차이(excess)'만 보시면 됩니다. 실측값은 활성 편중에 크게 오염됩니다.")
    print("  excess < 10%p  -> 레이어 유사 구조 없음. delta 대신 블록 수를 늘릴 것.")
    print("  excess > 30%p  -> 구조 있음. 해당 랭크로 delta 유효.")


if __name__ == "__main__":
    main()
