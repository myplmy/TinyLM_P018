#!/usr/bin/env python3
"""P022B 단계0 — **FP8(E4M3) 로 내려가면 우리 모델이 얼마나 나빠지는가.** 학습 0.

## 왜 이 진단이 FP8 의 방향을 정하는가

DeepSeek-V3(arXiv:2412.19437) §3.3 은 **Linear 의 GEMM 3개만** FP8 로 내리고
임베딩·헤드·어텐션·정규화·optimizer state 는 고정밀로 남겼다. 정확도는
**미세 입도 스케일**(활성 1x128 타일 / 가중치 128x128 블록)로 지켰다.

★**우리는 그 미세 입도를 이미 갖고 있다.** 어닐 완료 후

    wq[o,i] = sign[o,i] * alpha[o, g(i)]   sign in {-1,0,+1}   g = micro_group = 128

이고 **alpha 로 나누면 값이 정확히 {-1,0,+1}** 이다. E4M3 는 이 셋을 **정확히** 표현하므로
**스케일 입도가 alpha 입도와 같으면 가중치 FP8 오차는 0** 이다.

⚠️ **그런데 그것이 곧 장벽이다.** 논문 §3.3.2 원문:
   *"per-group scaling factors along the inner dimension of GEMM ...
     is not directly supported in the standard FP8 GEMM"*
→ DeepSeek 은 CUTLASS 커널을 직접 만들었다(DeepGEMM). 우리 alpha 도 **정확히 K축 그룹
스케일**이고, `torch._scaled_mm` 은 통상 per-tensor / per-row 만 받는다.
**P014B 의 U2 부정(bitnet.cpp 는 텐서당 스케일 1개)과 같은 벽**이다.

## 그래서 무엇을 재는가 — 네 부분

  [A] `torch._scaled_mm` 이 **이 GPU 에서** 어떤 스케일 입도를 실제로 받는가.
      ★이름만 보지 않고 **실제로 호출**한다(결과 028 section 11 이 `_weight_int8pack_mm` 에서
        한 것과 같은 방식). 되는 것과 안 되는 것을 예외 메시지까지 적는다.
  [B] 가중치를 스케일 입도별로 **E4M3 왕복**시키고 공통 원문 bpb 를 잰다.
      ★**g128 행의 상대오차는 0.00000 이어야 한다** — 아니면 도구가 틀린 것이고
        나머지 행을 읽으면 안 된다(결과 028 의 자기검증과 같은 장치).
  [C] **활성**을 입도별로 E4M3 왕복시킨다(per-tensor / per-token / 1x128 타일).
      논문이 활성에만 1x128 타일을 쓴 이유가 여기 있다.
  [D] 가중치 + 활성 **동시** — 실제 FP8 GEMM 에 가장 가까운 조건. **판정은 이 표로 한다.**

## 재학습하지 않는다 — 그래서 이 값은 **하한이 아니라 상한**이다

g128 로 학습된 모델을 사후에 왕복시키는 것이므로, QAT 로 적응시키면 더 좋아진다.
반대로 **실제 GEMM 의 누산 오차(Ada Tensor Core 의 가수 폭)는 이 방법으로 안 보인다** —
그쪽으로는 실제가 더 나쁠 수 있다. **두 방향의 미지가 모두 있다는 것을 명시한다.**

사용:
    python scripts/diag_fp8_precision.py --models mC_wsd --preset m100R1c
    python scripts/diag_fp8_precision.py --models mC_wsd mC_g16 --preset m100R1c --max-docs 2000
    python scripts/diag_fp8_precision.py --probe-only        # [A] 만(모델 불필요, 수 초)
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common_bpb import load_squad_contexts, SQUAD    # noqa: E402  (원문 로딩 단일 소스)

E4M3_MAX = 448.0        # float8_e4m3fn 의 최대 유한값. 448 = 1.75 * 2^8 로 E4M3 에 정확히 들어간다

# 가중치 스케일 입도. 정수 = 그 블록 크기(K축), None = per-row, 0 = per-tensor
W_GRANS = [("g128(자기검증)", 128), ("g256", 256), ("per-row", None), ("per-tensor", 0)]
# 활성 스케일 입도. 'tile' = 1x128(논문 규약), 'token' = per-row(=per-token), 'tensor' = 텐서당 1개
A_GRANS = [("없음", None), ("1x128 타일", "tile"), ("per-token", "token"), ("per-tensor", "tensor")]


# ---------------------------------------------------------------------------
# E4M3 왕복 (양자화 -> 역양자화). 실제 FP8 GEMM 이 겪는 표현 오차를 그대로 재현한다.
# ---------------------------------------------------------------------------
def _e4m3_roundtrip(x, scale):
    """`x` 를 `scale` 로 나눠 E4M3 로 내렸다가 되돌린다. `scale` 은 브로드캐스트 가능해야 한다."""
    import torch
    q = (x / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q.to(x.dtype) * scale


def quant_weight(wq, gran):
    """가중치 `_wq` 를 스케일 입도 `gran` 으로 E4M3 왕복.

    gran: 정수 = K축 블록 크기 / None = per-row / 0 = per-tensor
    반환: (새 텐서, fell_back). `fell_back=True` 면 `I % gran != 0` 이라 **그 텐서만 per-row**.

    ★스케일은 amax/E4M3_MAX 로 잡는다(표준 absmax 규약). 삼진 + g128 이면
      amax 가 그 그룹의 alpha 와 정확히 같으므로 `wq/scale` 이 **정확히 448*{-1,0,+1}** 이 되고
      E4M3 가 이를 정확히 표현한다 -> **상대오차 0**. 그게 [B] 의 자기검증이다.
    """
    import torch
    wq = wq.detach()
    O, I = wq.shape

    def _per_row():
        s = wq.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / E4M3_MAX
        return _e4m3_roundtrip(wq, s)

    if gran == 0:                                    # per-tensor
        s = wq.abs().amax().clamp_min(1e-12) / E4M3_MAX
        return _e4m3_roundtrip(wq, s), False
    if gran is None:                                 # per-row
        return _per_row(), False
    if I % gran != 0:                                # 나눠지지 않으면 그 텐서만 per-row 로 내린다
        return _per_row(), True                      # (결과 028 에서 g512 행이 통째로 비었던 사고 방지)
    G = I // gran
    w = wq.reshape(O, G, gran)
    s = w.abs().amax(dim=2, keepdim=True).clamp_min(1e-12) / E4M3_MAX
    return _e4m3_roundtrip(w, s).reshape(O, I), False


def quant_act(x, kind):
    """활성 `x`([..., K])를 E4M3 왕복. kind = 'tile'(1x128) / 'token'(행당) / 'tensor'."""
    import torch
    if kind is None:
        return x
    xf = x.float()
    if kind == "tensor":
        s = xf.abs().amax().clamp_min(1e-12) / E4M3_MAX
    elif kind == "token":
        s = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / E4M3_MAX
    elif kind == "tile":
        K = xf.shape[-1]
        t = 128 if K % 128 == 0 else K               # 나눠지지 않으면 per-token 으로 내린다
        sh = xf.shape[:-1] + (K // t, t)
        v = xf.reshape(sh)
        s = v.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / E4M3_MAX
        return _e4m3_roundtrip(v, s).reshape(xf.shape).to(x.dtype)
    else:
        raise ValueError(kind)
    return _e4m3_roundtrip(xf, s).to(x.dtype)


# ---------------------------------------------------------------------------
# [A] `torch._scaled_mm` 능력 조사 — 이름이 아니라 **호출**로 확인한다
# ---------------------------------------------------------------------------
def probe_scaled_mm(dev):
    import torch
    W = 96
    print("=" * W)
    print("  [A] torch._scaled_mm 이 이 GPU 에서 받는 스케일 입도 — **실제로 호출해서** 확인")
    print("=" * W)
    if not (hasattr(torch, "float8_e4m3fn") and hasattr(torch, "_scaled_mm")):
        print("  [STOP] 이 torch 빌드에 FP8/_scaled_mm 이 없다.")
        return
    cap = torch.cuda.get_device_capability() if dev == "cuda" else (0, 0)
    print(f"  device={dev}  sm_{cap[0]}{cap[1]}  torch {torch.__version__}")
    print("  ★DeepSeek-V3 section 3.3.2: K축 그룹 스케일은 '표준 FP8 GEMM 이 지원하지 않는다'.")
    print("    우리 alpha 가 정확히 그것이므로, 아래 blockwise 행이 이 계획의 갈림길이다.")
    print()
    shapes = [("MLP gate/up", 8192, 768, 2048),
              ("MLP down   ", 8192, 2048, 768),
              ("Attn q/o   ", 8192, 768, 768)]
    print(f"  {'shape':<14}{'M':>6}{'K':>6}{'N':>6}  {'recipe':<16}{'결과':<44}")
    print("  " + "-" * (W - 4))
    for name, M, K, N in shapes:
        a = torch.randn(M, K, device=dev).clamp(-1, 1).to(torch.float8_e4m3fn)
        b = torch.randn(N, K, device=dev).clamp(-1, 1).to(torch.float8_e4m3fn).t()   # [K,N] col-major
        recipes = [
            ("per-tensor", lambda: (torch.ones(1, device=dev), torch.ones(1, device=dev))),
            ("rowwise", lambda: (torch.ones(M, 1, device=dev), torch.ones(1, N, device=dev))),
            ("blockwise 1x128", lambda: (torch.ones(M, K // 128, device=dev),
                                         torch.ones(K // 128, N, device=dev))),
            ("blockwise 128x128", lambda: (torch.ones(M, K // 128, device=dev),
                                           torch.ones(K // 128, (N + 127) // 128, device=dev))),
        ]
        for rname, mk in recipes:
            try:
                sa, sb = mk()
                out = torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
                msg = f"OK  out={tuple(out.shape)} {out.dtype}"
            except Exception as e:                      # noqa: BLE001 (무엇이든 그대로 적는다)
                msg = "FAIL " + " ".join(str(e).split())[:38]
            print(f"  {name:<14}{M:>6}{K:>6}{N:>6}  {rname:<16}{msg:<44}")
    print("  " + "-" * (W - 4))
    print("  ★읽는 법")
    print("    blockwise 가 OK  -^ 우리 alpha 를 그대로 살릴 수 있다. 가중치 오차 0 경로가 열린다")
    print("    rowwise 만 OK    -^ [B] per-row 행이 그대로 실현 가능한 최선이다")
    print("    per-tensor 만 OK -^ 결과 028 이 이미 부정적(0.141~0.261 nats). 커널 자체 제작 외 길 없음")
    print()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="P022B 단계0 — FP8(E4M3) 수치 게이트")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--data", default="ko-en")
    ap.add_argument("--tokens", default="300M")
    ap.add_argument("--preset", default="m100R1c")
    ap.add_argument("--arch", nargs="+", default=None)
    ap.add_argument("--max-docs", type=int, default=4000)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--probe-only", action="store_true", help="[A] 만 실행(모델 불필요, 수 초)")
    a = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from tinylm import paths
    from tinylm.infer.generate import load_model
    from tinylm.data import tokenizer_path
    from tinylm.model.ternary import TLinear
    from tokenizers import Tokenizer

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    probe_scaled_mm(dev)
    if a.probe_only:
        return
    if not a.models:
        raise SystemExit("[STOP] --models 를 주거나 --probe-only 를 쓰세요.")
    if not SQUAD.exists():
        raise SystemExit(f"[STOP] SQuAD 원문이 없다: {SQUAD}")

    docs = load_squad_contexts(limit=a.max_docs)
    text = "\n\n".join(docs)
    n_bytes = len(text.encode("utf-8"))

    W = 96
    print("=" * W)
    print("  [B~D] E4M3 왕복 오차 — 가중치 / 활성 / 동시")
    print("=" * W)
    print(f"  device={dev}  SQuAD context {len(docs):,}개  {n_bytes/1e6:.2f} MB  seq={a.seq}")
    print("  ★재학습하지 않는다. 배포되는 `_wq` 와 실제 활성을 **그대로** E4M3 로 왕복시킨다.")
    print("  ★판정은 bpb 로 한다. 가중치 상대오차는 참고값이다(품질과 단조가 아니다).")
    print("  ⚠️ 실제 GEMM 의 **누산** 오차는 이 방법으로 안 보인다 — 실제는 더 나쁠 수 있다.")
    print()

    archs = a.arch or ["dense" if t.startswith(("p6d", "dense", "p12d")) else "tied"
                       for t in a.models]

    def run_bpb(model, ids, bpt, act_kind):
        """활성 왕복 훅을 걸고 bpb 를 잰다. act_kind=None 이면 훅 없음."""
        handles = []
        if act_kind is not None:
            def pre(mod, args):
                return (quant_act(args[0], act_kind),) + tuple(args[1:])
            for m in model.modules():
                if isinstance(m, TLinear):
                    handles.append(m.register_forward_pre_hook(pre))
        n_crop = (len(ids) - 1) // a.seq
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, n_crop, a.micro_bs):
                ch = [ids[j*a.seq:(j+1)*a.seq + 1]
                      for j in range(i, min(i + a.micro_bs, n_crop))]
                t = torch.from_numpy(np.stack(ch)).to(dev)
                x, y = t[:, :-1], t[:, 1:]
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
                    logits = model(x)
                l = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
                tot += float(l) * y.numel(); cnt += y.numel()
        for h in handles:
            h.remove()
        loss = tot / cnt
        return loss, loss / math.log(2) / bpt

    for tag, arch in zip(a.models, archs):
        ck = paths.resolve_ckpt(a.preset, a.data, a.tokens, tag)
        if not ck.exists():
            print(f"  [건너뜀] 체크포인트 없음: {ck.name}")
            continue
        tok = Tokenizer.from_file(str(tokenizer_path(a.data)))
        ids = np.array(tok.encode(text).ids, dtype=np.int64)
        bpt = n_bytes / max(len(ids), 1)
        print(f"  ── {tag} ({arch})   토큰 {len(ids):,}  bytes/token {bpt:.4f}")

        # ---- 기준선(왕복 없음) ----
        model, cfg, _ = load_model(arch=arch, ckpt_path=str(ck), device=dev)
        model.set_anneal(1.0); model.eval(); model.freeze_quant()
        base_loss, base_bpb = run_bpb(model, ids, bpt, None)
        print(f"     기준선(왕복 없음)                                손실 {base_loss:.4f}  bpb {base_bpb:.4f}")
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

        # ---- [B] 가중치만 ----
        print()
        print(f"     [B] 가중치 E4M3 왕복")
        print(f"     {'입도':<16}{'스케일 개수':>14}{'가중치 상대오차':>16}{'손실':>10}{'bpb':>10}{'Δbpb':>10}")
        print("     " + "-" * (W - 8))
        for name, gran in W_GRANS:
            model, cfg, _ = load_model(arch=arch, ckpt_path=str(ck), device=dev)
            model.set_anneal(1.0); model.eval(); model.freeze_quant()
            n_s, num, den, n_fb = 0, 0.0, 0.0, 0
            with torch.no_grad():
                for m in [x for x in model.modules() if isinstance(x, TLinear)]:
                    wq = m._wq.detach()
                    new, fb = quant_weight(wq, gran)
                    num += float((new - wq).pow(2).sum()); den += float(wq.pow(2).sum())
                    O, I = wq.shape
                    if fb:
                        n_fb += 1; n_s += O
                    else:
                        n_s += (1 if gran == 0 else (O if gran is None else O * (I // gran)))
                    m._wq = new
            rel = math.sqrt(num / max(den, 1e-12))
            nm = f"{name}*{n_fb}" if n_fb else name
            loss, bpb = run_bpb(model, ids, bpt, None)
            print(f"     {nm:<16}{n_s:>14,}{rel:>16.5f}{loss:>10.4f}{bpb:>10.4f}{bpb-base_bpb:>+10.4f}")
            del model
            if dev == "cuda":
                torch.cuda.empty_cache()

        # ---- [C] 활성만 ----
        print()
        print(f"     [C] 활성 E4M3 왕복 (TLinear 입력)")
        print(f"     {'입도':<16}{'':>14}{'':>16}{'손실':>10}{'bpb':>10}{'Δbpb':>10}")
        print("     " + "-" * (W - 8))
        for name, kind in A_GRANS[1:]:
            model, cfg, _ = load_model(arch=arch, ckpt_path=str(ck), device=dev)
            model.set_anneal(1.0); model.eval(); model.freeze_quant()
            loss, bpb = run_bpb(model, ids, bpt, kind)
            print(f"     {name:<16}{'':>14}{'':>16}{loss:>10.4f}{bpb:>10.4f}{bpb-base_bpb:>+10.4f}")
            del model
            if dev == "cuda":
                torch.cuda.empty_cache()

        # ---- [D] 동시 — 판정은 이 표로 한다 ----
        print()
        print(f"     [D] 가중치 + 활성 동시  ★판정표")
        print(f"     {'가중치':<16}{'활성':<16}{'':>14}{'손실':>10}{'bpb':>10}{'Δbpb':>10}")
        print("     " + "-" * (W - 8))
        for wname, gran in [("g128", 128), ("per-row", None)]:
            for aname, kind in [("1x128 타일", "tile"), ("per-token", "token")]:
                model, cfg, _ = load_model(arch=arch, ckpt_path=str(ck), device=dev)
                model.set_anneal(1.0); model.eval(); model.freeze_quant()
                with torch.no_grad():
                    for m in [x for x in model.modules() if isinstance(x, TLinear)]:
                        m._wq = quant_weight(m._wq.detach(), gran)[0]
                loss, bpb = run_bpb(model, ids, bpt, kind)
                print(f"     {wname:<16}{aname:<16}{'':>14}{loss:>10.4f}{bpb:>10.4f}"
                      f"{bpb-base_bpb:>+10.4f}")
                del model
                if dev == "cuda":
                    torch.cuda.empty_cache()
        print()

    print("=" * W)
    print("  ★어떻게 읽는가 (게이트 G0)")
    print("=" * W)
    print("  0. **[B] g128 행의 상대오차가 0.00000 이 아니면 도구가 틀린 것**이다.")
    print("     삼진 코드는 {-1,0,+1} 이고 E4M3 는 이를 정확히 표현한다. 다른 행을 읽지 말 것.")
    print("  1. [D] 최선 조합 Δbpb ^< 0.008(우리 실무 분해능의 bpb 환산)")
    print("       -^ FP8 수치 게이트 통과. 단계1(실 구현) 검토 대상")
    print("  2. 0.008 ~ 0.02")
    print("       -^ 사후 왕복으로는 부족하다 = **재학습(QAT)이 필요**. 비용이 크게 오른다")
    print("  3. ^> 0.02")
    print("       -^ ★여기서 종결한다. 커널을 만들어야만 성립하고 그건 P014B 에서 접은 것과 같은 작업")
    print()
    print("  ★[A] 와 함께 읽는다: blockwise 가 FAIL 이면 [B] 의 g128 행은 **실현 불가능한 최선**이다.")
    print("     그때 실현 가능한 값은 rowwise 가 OK 일 때의 per-row 행이다.")
    print("  ⚠️ 이 표는 **재학습 없는 사후 왕복**이다. QAT 로 적응시키면 더 좋아지고,")
    print("     실제 GEMM 의 누산 오차는 안 보이므로 더 나빠질 수도 있다. **양방향 미지**.")
    print("=" * W)


if __name__ == "__main__":
    main()
