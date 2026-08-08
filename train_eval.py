"""검증 + from-scratch 학습/평가 스캐폴드.

  python train_eval.py test          # 스모크 테스트만
  python train_eval.py train         # 타이드 모델 학습
  python train_eval.py train --dense # 동일 shape dense 기준선 학습

핵심: 절대 점수가 아니라 **동일 토큰 예산의 dense 기준선과의 차이**가 성적표다.
"""
import argparse, math, sys, time
import torch, torch.nn.functional as F
from tied_mlp_transformer import TMTConfig, TiedMLPTransformer, dense_baseline


def tiny(**kw):
    base = dict(vocab_size=512, dim=256, ffn_dim=512, n_q_heads=4, n_kv_heads=2,
                emb_rank=64, n_prelude=1, n_middle=4, n_coda=1, mlp_group=2,
                cla_group=2, micro_group=128, max_seq_len=64)
    base.update(kw); return TMTConfig(**base)


def smoke():
    torch.manual_seed(0)
    ok = lambda c: "OK" if c else "FAIL"

    print("### 1. forward / backward / 파라미터 커버리지")
    m = TiedMLPTransformer(tiny()); m.train()
    x = torch.randint(0, 512, (2, 16))
    logits = m(x)
    F.cross_entropy(logits.reshape(-1, 512), x.reshape(-1)).backward()
    nograd = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    pg = m.param_groups(1e-3)
    covered = sum(len(g["params"]) for g in pg)
    total = sum(1 for p in m.parameters() if p.requires_grad)
    ids = [id(p) for g in pg for p in g["params"]]
    print(f"  logits {tuple(logits.shape)}   grad 없는 파라미터: {nograd or '없음'}")
    print(f"  param_groups 커버 {covered}/{total}, 중복 {len(ids)-len(set(ids))}개  -> "
          f"{ok(covered == total and len(ids) == len(set(ids)))}")

    print("\n### 2. MLP 타잉이 실제로 공유되는가")
    c = tiny(n_middle=4, mlp_group=2)
    mm = TiedMLPTransformer(c)
    mids = [id(l.mlp[0]) for l in mm.layers[c.n_prelude:c.n_prelude+c.n_middle]]
    n_unique = len(set(mids))
    print(f"  중간 {c.n_middle}층의 MLP 인스턴스 {n_unique}개 (기대 {c.n_mlp_groups})  -> "
          f"{ok(n_unique == c.n_mlp_groups)}")
    p_tied = sum(p.numel() for p in mm.parameters())
    p_dense = sum(p.numel() for p in TiedMLPTransformer(dense_baseline(c)).parameters())
    print(f"  파라미터 {p_dense/1e3:.0f}K -> {p_tied/1e3:.0f}K  ({p_dense/p_tied:.2f}배 감축)")

    print("\n### 3. CLA — 인접 층이 같은 K/V 텐서를 쓰는가")
    c = tiny(cla_group=2)
    mc = TiedMLPTransformer(c)
    owners = mc.owner
    has_kv = [hasattr(l.attn, "k_proj") for l in mc.layers]
    n_owner = sum(has_kv)
    print(f"  owner map {owners}")
    print(f"  K/V 프로젝션 보유 층 {n_owner}/{c.n_layers}  -> "
          f"{ok(n_owner == math.ceil(c.n_layers/c.cla_group))}")

    print("\n### 4. gradient checkpointing 등가성")
    def g(ck):
        torch.manual_seed(0)
        mm = TiedMLPTransformer(tiny(grad_checkpoint=ck)); mm.train()
        F.cross_entropy(mm(x).reshape(-1, 512), x.reshape(-1)).backward()
        return {n: p.grad.clone() for n, p in mm.named_parameters() if p.grad is not None}
    a, b = g(True), g(False)
    w = max(((a[k]-b[k]).abs().max()/(b[k].abs().max()+1e-12)).item() for k in a)
    print(f"  최대 상대오차 {w:.2e}  -> {ok(w < 1e-5)}")

    print("\n### 5. AMP 안전성 (loss scale 동차성)")
    m2 = TiedMLPTransformer(tiny())
    def gs(s):
        m2.zero_grad()
        (F.cross_entropy(m2(x).reshape(-1,512), x.reshape(-1)) * s).backward()
        return m2.layers[0].attn.q_proj.weight.grad.clone() / s
    r = (gs(1.0)-gs(65536.0)).abs().max().item() / (gs(1.0).abs().max().item()+1e-12)
    print(f"  scale 1 vs 65536 상대오차 {r:.2e}  -> {ok(r < 1e-5)}")

    print("\n### 6. 모드 제어 (adaLN 전용 / mode-delta 옵션)")
    for rank in (0, 16):
        cm = tiny(n_modes=3, mode_rank=rank)
        mo = TiedMLPTransformer(cm); mo.eval()
        with torch.no_grad():
            for mod in mo.modules():          # 둘 다 0 초기화라 학습된 상태를 흉내
                if hasattr(mod, "use_mode") and mod.use_mode:
                    mod.mode_b.normal_(0, 0.05)
                if hasattr(mod, "use_mode_ln") and mod.use_mode_ln:
                    mod.mode_scale.normal_(0, 0.1)
            sch = torch.eye(3)[torch.tensor([0,0,1,2,0,1])[:cm.n_layers]]
            d = (mo(x) - mo(x, mode_override=sch)).abs().mean().item()
        n = sum(p.numel() for p in mo.parameters())
        print(f"  mode_rank={rank:<3} 파라미터 {n/1e3:>6.0f}K   스케줄 강제 시 출력 변화 {d:.4f}")
    print("  (rank=0은 adaLN만으로 모드 제어 — 기본 설정)")

    print("\n### 7. 삼진 어닐링")
    ma = TiedMLPTransformer(tiny())
    outs = []
    for an in (0.0, 0.5, 1.0):
        ma.cfg.quant_anneal = an
        with torch.no_grad():
            outs.append(ma(x).clone())
    print(f"  anneal 0.0(FP) vs 1.0(삼진) 출력 차이 {(outs[0]-outs[2]).abs().mean():.4f}")
    print(f"  anneal 0.5 는 그 중간인가: "
          f"{ok((outs[1]-outs[0]).abs().mean() < (outs[2]-outs[0]).abs().mean())}")


def train(dense=False, steps=200, bs=8, seq=512, lr=3e-3, warmup=20):
    cfg = TMTConfig()
    if dense:
        cfg = dense_baseline(cfg)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = TiedMLPTransformer(cfg).to(dev)
    model.cfg.grad_checkpoint = True
    print(model.report())
    print(f"학습: {'dense 기준선' if dense else '타이드'}  device={dev}  "
          f"{steps}step x {bs}x{seq} = {steps*bs*seq/1e6:.1f}M 토큰\n")

    opt = torch.optim.AdamW(model.param_groups(lr), betas=(0.9, 0.95))
    sched = lambda s: (s/warmup if s < warmup else
                       0.1 + 0.45*(1+math.cos(math.pi*(s-warmup)/(steps-warmup))))
    scaler = torch.amp.GradScaler(dev) if dev == "cuda" else None

    # TODO: 실제 데이터로 교체.  from datasets import load_dataset ...
    def batch():
        return torch.randint(0, cfg.vocab_size, (bs, seq+1), device=dev)

    model.train(); t0 = time.time()
    for s in range(steps):
        # 삼진 어닐링: 초반 10%는 FP, 이후 40% 구간에 걸쳐 삼진으로 이행
        model.cfg.quant_anneal = min(1.0, max(0.0, (s/steps - 0.10) / 0.40))
        for g in opt.param_groups:
            g["lr"] = g.get("_base", g["lr"]) * sched(s); g.setdefault("_base", g["lr"]/sched(s))

        d = batch()
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logits, aux = model(d[:, :-1], return_aux=True)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), d[:, 1:].reshape(-1))
            loss = loss + 0.01 * aux["router_loss"]

        if scaler:
            scaler.scale(loss).backward(); scaler.unscale_(opt)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler:
            scaler.step(opt); scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        model.clear_quant()                      # _wq가 붙잡은 그래프 해제

        if s % 20 == 0 or s == steps-1:
            print(f"  step {s:>4}  loss {loss.item():.4f}  anneal {model.cfg.quant_anneal:.2f}"
                  f"  {(time.time()-t0)/(s+1)*1000:.0f} ms/step")
    print(f"\n최종 loss {loss.item():.4f}   (dense 기준선과의 차이가 성적표)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["test", "train"])
    p.add_argument("--dense", action="store_true")
    p.add_argument("--steps", type=int, default=200)
    a = p.parse_args()
    smoke() if a.cmd == "test" else train(a.dense, a.steps)
