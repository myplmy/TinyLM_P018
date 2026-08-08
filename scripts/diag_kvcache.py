#!/usr/bin/env python3
"""P030 단계1.5-B — KV 캐시 정확성의 **진짜 게이트**: teacher-forced 로짓 동등성.

★왜 기존 게이트를 대체하는가 (결과 014 §8)
  P030 계획 단계1 이 요구한 것은 **"캐시 유/무의 로짓이 일치"** 다.
  `check_cache_equivalence()` 는 그것을 **"디코딩된 문자열이 완전히 같다"** 로 대리했다.
  그 대리는 세 가지를 통과시킨다:
    (a) bf16 autocast(cuda) — 가수 8비트
    (b) **자기회귀 24스텝** — near-tie 한 번 뒤집히면 이후 전부가 달라진다(혼돈 증폭)
    (c) 토크나이저 decode
  실제로 `mA_g4s34_k4` 가 2/5 불일치, `p6d` 가 5/5 일치로 갈렸는데,
  RoPE 슬라이스와 SDPA 마스크는 **dense 와 tied 가 공유하는 코드**다. p6d 가 통과했으므로
  그 둘은 이미 무죄이고, 게이트는 **아키텍처가 아니라 near-tie 빈도**를 재고 있었다.

★이 스크립트가 재는 것
  같은 토큰열을 두 경로로 흘리되 **샘플링도 피드백도 없다(teacher forcing)**:
    path A : 전체 시퀀스 1회 forward, 캐시 없음
    path B : 앞부분을 prefill(use_cache) → 나머지를 **알려진 다음 토큰**으로 1개씩
  위치별 로짓을 비교한다. **결정론적이고 증폭이 없다.**

  하드 게이트 : fp32 에서 max|Δlogit| < tol (기본 1e-3)
  보고 전용   : bf16 그리디 불일치 + **최초 분기점의 top-2 로짓 간격**
                간격이 관측된 로짓 오차보다 작으면 **타이브레이크**이지 버그가 아니다.

★오차 패턴이 버그 위치를 지목한다 (기존 게이트는 못 하던 것)
    위치에 따라 증가            → RoPE 절대위치 오프셋
    prefill 경계 이후 균일하게 큼 → SDPA 마스크 정렬
    tied 에서만 큼              → CLA owner 키 캐시(kv_bank)
    평탄한 1e-6                 → 정상

사용법
  python scripts/diag_kvcache.py --models mA_g4s34_k4 mC_g8_k4 p6d --device cpu --tol 1e-3
  python scripts/diag_kvcache.py --models mA_g4s34_k4 --device cuda --report-greedy
종료코드: 0 통과 / 1 허용오차 초과 / 2 체크포인트 없음
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 한글 프롬프트가 필요해 .bat 이 아니라 여기 산다(ASCII 규약). P029 프롬프트와 같은 집합.
PROMPTS = [
    "대한민국의 수도 서울은",
    "조선 시대의 과학 기술은 세종대왕 때",
    "물이 끓는 온도는 섭씨",
    "The Industrial Revolution began in",
    "인공지능 기술의 발전은 사회에",
]

DEFAULT_MODELS = [("mA_g4s34_k4", "tied"), ("mC_g8_k4", "tied"), ("p6d", "dense")]


def banner(s, ch="="):
    print("\n" + ch * 88)
    print(f"  {s}")
    print(ch * 88)


def _arch_of(tag):
    return "dense" if tag.startswith(("p6d", "dense", "p12d")) else "tied"


def teacher_forced_delta(model, cfg, tok, prompt, max_new, device, use_autocast):
    """★핵심. 같은 토큰열을 두 경로로 흘려 위치별 로짓 차이를 잰다.

    토큰열은 **캐시 없는 경로의 그리디 결과**로 만든다(둘 중 하나를 골라야 하고,
    어느 쪽을 고르든 두 경로 모두 **같은 입력**을 받는다는 점이 중요하다).
    반환: (per_step_max, per_step_mean, argmax_agree, n_steps, prefill_len)
    """
    import torch

    ids = tok.encode(prompt).ids
    T0 = len(ids)
    dev_t = device if isinstance(device, str) else device.type
    ac = dict(device_type=dev_t, dtype=torch.bfloat16, enabled=bool(use_autocast))

    # 1) 캐시 없는 경로로 토큰열을 만든다(그리디). 이 열이 두 경로의 공통 입력이 된다.
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_new):
            with torch.autocast(**ac):
                lg = model(x[:, -cfg.max_seq_len:])[:, -1, :].float()
            x = torch.cat([x, lg.argmax(-1, keepdim=True)], dim=1)
    seq = x                                              # (1, T0+max_new)

    # 2) path A — 전체 1회 forward, 캐시 없음
    with torch.no_grad(), torch.autocast(**ac):
        logits_A = model(seq)[0].float()                 # (T, V)

    # 3) path B — prefill(T0) 후 나머지를 1토큰씩. **teacher forcing** 이라 피드백이 없다.
    outs, past = [], None
    with torch.no_grad():
        with torch.autocast(**ac):
            lg, past = model(seq[:, :T0], past_kv=None, use_cache=True)
        outs.append(lg[0].float())                       # prefill 구간 전체
        for t in range(T0, seq.shape[1]):
            with torch.autocast(**ac):
                lg, past = model(seq[:, t:t + 1], past_kv=past, use_cache=True)
            outs.append(lg[0].float())
    logits_B = torch.cat(outs, dim=0)                    # (T, V)

    assert logits_A.shape == logits_B.shape, f"{logits_A.shape} vs {logits_B.shape}"
    d = (logits_A - logits_B).abs()
    per_max = d.max(dim=-1).values                       # (T,)
    per_mean = d.mean(dim=-1)
    agree = (logits_A.argmax(-1) == logits_B.argmax(-1))
    return per_max, per_mean, agree, seq.shape[1], T0


def greedy_tie_margin(model, cfg, tok, prompt, max_new, device, use_autocast):
    """보고 전용: 캐시 유/무 그리디를 **각자** 돌려 최초 분기 위치와 그 지점의 top-2 간격."""
    import torch

    ids = tok.encode(prompt).ids
    dev_t = device if isinstance(device, str) else device.type
    ac = dict(device_type=dev_t, dtype=torch.bfloat16, enabled=bool(use_autocast))

    def run(use_cache):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        gaps, past = [], None
        with torch.no_grad():
            for _ in range(max_new):
                if use_cache:
                    xin = x if past is None else x[:, -1:]
                    with torch.autocast(**ac):
                        lg, past = model(xin, past_kv=past, use_cache=True)
                    lg = lg[:, -1, :].float()
                else:
                    with torch.autocast(**ac):
                        lg = model(x[:, -cfg.max_seq_len:])[:, -1, :].float()
                top2 = lg.topk(2, dim=-1).values[0]
                gaps.append(float(top2[0] - top2[1]))
                x = torch.cat([x, lg.argmax(-1, keepdim=True)], dim=1)
        return x[0].tolist(), gaps

    a_ids, a_gaps = run(True)
    b_ids, _ = run(False)
    n0 = len(ids)
    for i in range(n0, min(len(a_ids), len(b_ids))):
        if a_ids[i] != b_ids[i]:
            return i - n0, a_gaps[i - n0]                # (분기 스텝, 그 지점 top-2 간격)
    return None, None


def main():
    ap = argparse.ArgumentParser(description="KV 캐시 teacher-forced 로짓 동등성 게이트")
    ap.add_argument("--models", nargs="*", help="태그 목록(기본 mA_g4s34_k4 mC_g8_k4 p6d)")
    ap.add_argument("--device", default="cpu",
                    help="cpu 면 autocast 가 꺼져 fp32(하드 게이트용). cuda 는 bf16")
    ap.add_argument("--tol", type=float, default=1e-3, help="fp32 하드 게이트 허용 max|Δlogit|")
    ap.add_argument("--max-new", type=int, default=24, help="teacher-forced 스텝 수")
    ap.add_argument("--report-greedy", action="store_true",
                    help="그리디 분기 위치와 top-2 간격을 함께 보고(판정에는 쓰지 않는다)")
    ap.add_argument("--data", default="ko-en")
    ap.add_argument("--tokens", default="300M")
    ap.add_argument("--preset", default="m100")
    a = ap.parse_args()

    from tinylm import paths
    from tinylm.infer.generate import load_model
    from tokenizers import Tokenizer
    from tinylm.data import tokenizer_path

    models = [(t, _arch_of(t)) for t in a.models] if a.models else DEFAULT_MODELS
    tok = Tokenizer.from_file(str(tokenizer_path(a.data)))
    base = f"{a.preset}_{a.data}_{a.tokens}"
    use_ac = (a.device == "cuda")

    banner(f"P030 1.5-B — teacher-forced 로짓 동등성  device={a.device}  "
           f"{'bf16 autocast' if use_ac else 'fp32(autocast off)'}  tol={a.tol:g}")
    print("  ★자기회귀 피드백을 끊었으므로 이 수치는 결정론적이다.")
    print("  ★오차 패턴 읽는 법: 위치증가=RoPE / prefill 이후 균일=마스크 / tied만=CLA / 평탄 1e-6=정상")
    if use_ac:
        print("  ※ cuda 는 bf16 이라 tol 을 넘는 것이 정상일 수 있다. **하드 게이트는 cpu(fp32)** 다.")

    worst, missing, rows = {}, [], []
    for tag, arch in models:
        ck = paths.RUNS / "ckpt" / f"{base}_{tag}.pt"
        if not ck.exists():
            print(f"\n  [건너뜀] 체크포인트 없음: {ck.name}")
            missing.append(tag)
            continue
        model, cfg, device = load_model(arch=arch, ckpt_path=str(ck), device=a.device)
        print(f"\n  ── {tag} ({arch}, cla_group={cfg.cla_group}, "
              f"mlp_group={cfg.mlp_group if cfg.tie_mlp else 1}, "
              f"sparse34={bool(getattr(cfg, 'sparse34', False))}) "
              + "─" * 20)
        m_all = 0.0
        for prompt in PROMPTS:
            per_max, per_mean, agree, T, T0 = teacher_forced_delta(
                model, cfg, tok, prompt, a.max_new, device, use_ac)
            pre = float(per_max[:T0].max())          # prefill 구간(캐시 무관 — 두 경로 동일해야 함)
            dec = float(per_max[T0:].max()) if T > T0 else 0.0
            # 위치 의존성: decode 구간 전반부 vs 후반부 (RoPE 오프셋이면 뒤로 갈수록 커진다)
            half = T0 + max(1, (T - T0) // 2)
            d1 = float(per_max[T0:half].max()) if half > T0 else 0.0
            d2 = float(per_max[half:].max()) if T > half else 0.0
            nag = int((~agree).sum())
            m_all = max(m_all, dec, pre)
            flag = "OK " if max(pre, dec) < a.tol else "!! "
            print(f"    {flag}{prompt[:22]:<24} prefill {pre:.2e}  decode {dec:.2e}  "
                  f"(전반 {d1:.2e} / 후반 {d2:.2e})  argmax불일치 {nag}/{T}")
            if a.report_greedy:
                idx, gap = greedy_tie_margin(model, cfg, tok, prompt, a.max_new, device, use_ac)
                if idx is None:
                    print(f"        그리디: 일치")
                else:
                    verdict = ("타이브레이크(간격 < 로짓오차)" if gap is not None and gap < dec
                               else "간격이 로짓오차보다 크다 — 조사 필요")
                    print(f"        그리디: 스텝 {idx} 에서 분기, 그 지점 top-2 간격 {gap:.4f}  → {verdict}")
        worst[tag] = m_all
        rows.append((tag, arch, cfg, m_all))
        del model

    banner("요약")
    print(f"    {'모델':<16}{'arch':>7}{'cla':>5}{'g':>4}{'s34':>7}{'max|dlogit|':>14}  판정")
    print("    " + "-" * 62)
    for tag, arch, cfg, m in rows:
        print(f"    {tag:<16}{arch:>7}{cfg.cla_group:>5}"
              f"{(cfg.mlp_group if cfg.tie_mlp else 1):>4}"
              f"{str(bool(getattr(cfg, 'sparse34', False))):>7}{m:>14.3e}  "
              f"{'통과' if m < a.tol else '초과'}")
    print("    ★cla=1 인 모델(dense)만 통과하고 cla=2 가 초과하면 CLA 캐시 경로다.")
    if missing:
        print(f"\n    [건너뜀] {', '.join(missing)}")
        if not rows:
            return 2

    bad = [t for t, m in worst.items() if m >= a.tol]
    if not bad:
        print(f"\n  ✅ 전 모델 max|Δlogit| < {a.tol:g}"
              + ("  — **캐시 구현 정확성 확인.** 속도 측정으로 진행 가능."
                 if not use_ac else
                 "  — bf16 에서도 통과. 정밀도 여유가 충분하다는 뜻."))
        return 0

    print(f"\n  ❌ 허용오차 초과: {', '.join(bad)}")
    if use_ac:
        print("     단 이것은 **bf16** 이다. 하드 게이트는 `--device cpu`(fp32) 로 판정한다.")
    else:
        print("     fp32 에서 초과 = **실제 버그**다. 위의 prefill/decode/전반·후반 분해로 위치를 좁혀라.")
        print("       prefill 부터 큼        → 캐시와 무관한 문제(모델 로드·양자화)")
        print("       decode 만 크고 증가    → RoPE 오프셋(transformer.forward 의 rope 슬라이스)")
        print("       decode 만 크고 평탄    → SDPA 마스크(modules.Attention 의 tril)")
        print("       tied 만 큼             → CLA owner 키 캐시(kv_bank 키가 owner 인지)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
