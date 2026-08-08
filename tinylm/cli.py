#!/usr/bin/env python3
"""TinyLM 단일 진입점.

  python -m tinylm all --data ko-en --tokens 300M --steps 2289 --lr 1e-3 --compile
  python -m tinylm lrfind --method both
  python -m tinylm generate --arch tied --prompt "안녕하세요"

(호환) 저장소 루트의 run100m.py 도 이 main() 을 호출한다.
"""
from __future__ import annotations

import argparse

from . import paths  # noqa: F401  (HF 리다이렉트 먼저)
from .data import DATASETS
from .config import PRESETS


def _tok(s):
    s = str(s)
    return int(float(s.rstrip("MmBb")) * (1e9 if s[-1] in "Bb" else 1e6))


def _preset(a):
    return "tiny" if a.tiny else a.preset


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["prepare", "train", "eval", "compare", "all",
                                   "lrfind", "generate", "kdcache"])
    p.add_argument("--arch", choices=["dense", "tied"], default="tied")
    # choices 를 PRESETS 에서 자동 유도 — 종전 하드코딩은 `m100d` 를 빠뜨리고 있었다.
    p.add_argument("--preset", choices=list(PRESETS), default="m100",
                   help="m100R1a/m100R1c = REVIEW1 잠정 보존 후보(2026-07-31 승격). "
                        "아키텍처만 고정하고 KD 등 학습 조합은 명령줄이 정한다")
    p.add_argument("--data", default="ko-en", choices=list(DATASETS) + ["synthetic"])
    p.add_argument("--tokens", default="300M")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--micro-bs", type=int, default=8)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--tiny", action="store_true", help="tiny 프리셋(파이프라인 확인용)")
    p.add_argument("--no-ckpt", action="store_true", help="gradient checkpointing 끄기")
    p.add_argument("--compile", action="store_true", help="torch.compile 사용")
    p.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"],
                   default="default", help="reduce-overhead=CUDA그래프(런치 오버헤드↓)")
    # --- v6: 효율/실험 ---
    p.add_argument("--sched", choices=["cosine", "wsd", "stable", "decay"], default="cosine",
                   help="stable=plateau 생성(감쇠X), decay=plateau에서 cooldown 분기")
    p.add_argument("--decay-from", default=None, help="decay 분기 시 불러올 plateau 체크포인트 경로")
    p.add_argument("--anneal-end", type=float, default=0.60,
                   help="(P026) 삼진 어닐 완료 지점(진행률 0~1). 기본 0.60=종전 동작. "
                        "cooldown-QAT 정렬은 wsd 의 1-decay_frac 과 같은 값으로 준다(예: 0.80)")
    p.add_argument("--decay-frac", type=float, default=0.2,
                   help="(P026) wsd 스케줄에서 마지막 LR 감쇠 구간 비율(기본 0.2)")
    p.add_argument("--anneal-shape", choices=["linear", "step"], default="linear",
                   help="(P035) 삼진 어닐의 형태. linear=종전 램프(기본, 무변). "
                        "step=--anneal-start 까지 FP(anneal 0), 그 지점에서 1.0 으로 급전이. "
                        "논문이 상정한 'FP 학습 후 별도 QAT' 를 인위적으로 재현해 기전 유무를 본다")
    p.add_argument("--anneal-start", type=float, default=None,
                   help="(P035) 어닐 시작(step 이면 전이) 지점(진행률 0~1). "
                        "미지정이면 종전대로 warm/steps+0.05 를 쓴다(무변)")
    # ── P036 : Arenas(annealing residual synapse). 학습 플래그, 기본 off ──
    p.add_argument("--arenas", action="store_true",
                   help="(P036) Y=X*Ta + lambda_t*X*W. latent W 를 입력 gradient 경로에 넣는다")
    p.add_argument("--arena-lambda", type=float, default=0.1,
                   help="(P036) lambda_0. 학습 시작 시점의 residual 계수")
    p.add_argument("--arena-end", type=float, default=0.9,
                   help="(P036) 이 진행률에서 lambda_t=0 (이후 순수 삼진 → 추론 오버헤드 0)")
    # ── P031 : 추론 시 middle 반복(깊이 외삽). eval/generate 전용, 학습 무영향 ──
    p.add_argument("--infer-repeat", type=float, default=1.0,
                   help="(P031) middle 블록 통과 배수. 1.0=학습된 그대로 / 0.5=축소 / 1.5=확장")
    p.add_argument("--repeat-where", choices=["front", "back", "even"], default="front",
                   help="(P031) 분수 R 에서 어디를 더/덜 돌지. 결과가 이것에 의존한다")
    p.add_argument("--repeat-kv-reuse", action="store_true",
                   help="(P031) 반복 통과에서 KV 를 재계산하지 않고 첫 통과 것을 재사용(대조 조건)")
    p.add_argument("--seed", type=int, default=1337,
                   help="시드(기본 1337=종전 동작). 가중치 초기화 + train 크롭 순서에 반영. "
                        "val 크롭은 항상 고정(99)이라 런 간 비교가 유지된다. 재현 노이즈 측정용")
    p.add_argument("--snapshot-at", default=None, help="토큰 마크에서 명명 스냅샷 저장(콤마, 예: 100M,300M,600M)")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay(0=끔, 예: 0.999)")
    p.add_argument("--early-stop", type=int, default=0, help="val 개선 없이 N회 eval시 종료(0=끔)")
    p.add_argument("--init-from", action="store_true", help="tied를 dense.pt로 부모초기화")
    p.add_argument("--init-from-tag", default=None,
                   help="부모초기화를 base_dense.pt 대신 base_{TAG}.pt 로(태그된 dense에서 초기화). "
                        "토큰스윕 클린판처럼 정본 dense 를 덮지 않고 별도 태그 dense 를 쓸 때.")
    p.add_argument("--kd", action="store_true", help="dense.pt를 교사로 KD")
    p.add_argument("--kd-best", action="store_true", help="KD 교사를 dense_best.pt로(더 강한 교사)")
    p.add_argument("--kd-cache", action="store_true", help="오프라인 KD(캐시 top-k, 교사 forward 없음)")
    p.add_argument("--kd-topk", type=int, default=16, help="오프라인 KD top-k")
    p.add_argument("--kd-every", type=int, default=1,
                   help="온라인 KD skip-forward: K스텝마다 교사 forward 1회(교사 연산 1/K). 1=매 스텝")
    p.add_argument("--kd-dynamic", action="store_true",
                   help="동적 KD: 교사 forward 간격을 1→kd-every 로 선형 증가(초반 촘촘·후반 성김)")
    p.add_argument("--kd-teacher-tag", default=None,
                   help="KD 교사를 dense 대신 임의 태그 체크포인트로(압축 교사 distill, P018)")
    p.add_argument("--ema-start", type=float, default=0.0, help="EMA를 steps의 이 비율 이후부터 누적(0=처음부터)")
    p.add_argument("--kd-alpha", type=float, default=0.5)
    p.add_argument("--kd-temp", type=float, default=2.0)
    p.add_argument("--lora-rank", type=int, default=0, help="공유 MLP 층별 LoRA rank(0=끔)")
    p.add_argument("--lora-bits", type=int, default=2, choices=[2, 16])
    p.add_argument("--lora-decay", type=float, default=0.0,
                   help="(P008) LoRA 출력 스케일 s(t) 를 1->0 으로 어닐. 진행률 이 지점에서 0. "
                        "0=끔(고정 LoRA, 종전 동작). 0 이 되면 배포 메모리 대가가 사라진다")
    p.add_argument("--tag", default=None, help="체크포인트/로그 파일명(실험 조건 구분용)")
    p.add_argument("--vs", default=None, help="compare에서 tied vs tied 비교할 상대 태그")
    p.add_argument("--mlp-group", type=int, default=None, help="MLP 타잉 g 오버라이드(프리셋값 대체, g-스윕용)")
    p.add_argument("--opt-dtype", choices=["fp32", "fp32c", "bf16"], default="fp32",
                   help="(P022B 단계2) AdamW **상태**(exp_avg/exp_avg_sq) 정밀도. "
                        "fp32=종전 torch fused AdamW(기본, 비트 동일) / "
                        "fp32c=우리 구현·fp32 상태(자기검증) / bf16=상태만 bf16(약 254MB 절감). "
                        "master weight·gradient·산술은 어느 모드에서도 fp32 다")
    p.add_argument("--micro-group", type=int, default=None,
                   help="(P051) 삼진 alpha 그룹 크기 오버라이드(프리셋 기본 128). 저장 bpw 의 "
                        "scale 항이 16/g 이므로 g 를 키우면 packed 가 준다. 미지정=프리셋값=비트동일")
    p.add_argument("--kd-teacher-infer", action="store_true",
                   help="(P042) KD 교사에 freeze_quant()+drop_latent() 를 적용한다. 교사는 "
                        "가중치가 고정인데 매 스텝 refresh_quant 를 돌고 fp32 latent 도 든다. "
                        "**기본 off = 종전 동작(비트 동일)**")
    p.add_argument("--emb-rank", type=int, default=None,
                   help="(P046) factorized embedding 병목 E 오버라이드(프리셋 256). "
                        "E 를 줄이면 임베딩이 선형으로 줄지만 **로짓 랭크가 E 로 제한**된다")
    p.add_argument("--mlp-film", action="store_true", help="공유 MLP에 층별 FiLM(거의 공짜 조건화)")
    p.add_argument("--center-weights", action="store_true", help="(실험) g128 그룹 latent weight centering")
    p.add_argument("--ternary-kernel", action="store_true", help="(실험) 커스텀 삼진 커널 경로(레퍼런스)")
    p.add_argument("--ternary-kernel-triton", action="store_true", help="커널 Triton forward(검증 후에만)")
    p.add_argument("--sparse34", action="store_true", help="(P016) 3:4 희소 삼진 1.25bpw(각 4-블록 |w|최소 1개 0강제)")
    p.add_argument("--pool-tokens", default=None,
                   help="데이터 풀(캐시) 크기를 학습길이·이름과 분리 지정(예: 600M). "
                        "미지정이면 --tokens 사용. 토큰스윕 클린판: 모든 예산을 같은 풀에서 샘플.")
    p.add_argument("--doc-filter", action="store_true",
                   help="(P037 단계2) SEO 스팸 문서를 제외하고 캐시를 만든다. 서명 = 대형 AND "
                        "줄바꿈 ~0%% AND 줄 고유율 100%% (결과 018). **별도 디렉터리에 쓴다**")
    p.add_argument("--doc-min-chars", type=int, default=50_000,
                   help="(P037) 스팸 판정 최소 길이(자). 이보다 짧으면 절대 버리지 않는다")
    p.add_argument("--exact-cache", action="store_true",
                   help="상위호환 캐시를 고르지 않고 정확히 {data}_{요청토큰} 캐시만 사용/생성.")
    p.add_argument("--force-dense", action="store_true", help="all 실행 시 dense 재학습 강제(기본은 재사용)")
    # lrfind
    p.add_argument("--method", choices=["range", "grid", "both"], default="range")
    p.add_argument("--lrs", default="3e-4,6e-4,1e-3,2e-3", help="grid 스윕 LR 목록(콤마)")
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--lr-max", type=float, default=1e-1)
    p.add_argument("--lrfind-steps", type=int, default=150)
    # generate
    p.add_argument("--prompt", default="")
    p.add_argument("--max-new", type=int, default=100)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--ckpt-path", default=None)
    # P030 단계1: 캐시/eos. 기본은 켜짐 — 끄는 쪽이 대조군이다.
    p.add_argument("--no-cache", action="store_true",
                   help="KV 캐시 off(정확성 대조용). 느리다 — 속도 측정에 쓰지 말 것")
    p.add_argument("--no-eos-stop", action="store_true",
                   help="<eos> 에서 멈추지 않는다(문서 경계를 넘어 생성)")
    p.add_argument("--check-cache", action="store_true",
                   help="캐시 유/무 그리디 출력 일치 검증만 하고 종료")
    a = p.parse_args()

    import sys as _sys                           # 실행 인자 로그(배치파일에서 어떤 조건인지 추적)
    print("[cmd] python " + " ".join(_sys.argv))

    n_tok = _tok(a.tokens)
    preset, ckpt = _preset(a), not a.no_ckpt
    tokstr = f"{n_tok//1_000_000}M" if n_tok >= 10**6 else str(n_tok)
    base = f"{preset}_{a.data}_{tokstr}"        # 스케일별 이름 프리픽스
    # ★부모 dense 는 프리셋을 넘어 찾는다(2026-08-01). `m100R1a/c` 는 `m100` 에서 한 필드만
    #   바꾼 파생이라 **부모가 하나뿐**인데, 종전에는 `m100R1a_..._dense.pt` 를 찾다 즉사했다
    #   (P038·P036 단계2 실패). 쓰기 경로는 그대로 `{preset}_...` 라 네임스페이스는 유지된다.
    dense_ck = paths.resolve_ckpt(preset, a.data, tokstr, "dense")
    dense_best_ck = paths.resolve_ckpt(preset, a.data, tokstr, "dense_best")

    pool_tok = _tok(a.pool_tokens) if a.pool_tokens else None

    if a.cmd == "prepare":
        from .data import prepare
        prepare(a.data, pool_tok if pool_tok else n_tok, exact=a.exact_cache,
                doc_filter=a.doc_filter, doc_min_chars=a.doc_min_chars)

    elif a.cmd == "kdcache":
        from .train.kd_cache import build_kd_cache
        teacher = str(dense_best_ck if a.kd_best else dense_ck)
        build_kd_cache(base, teacher, a.data, n_tok, a.steps, a.micro_bs, a.seq, a.accum, a.kd_topk, a.kd_temp)

    elif a.cmd == "train":
        from .train import train
        # KD 교사 경로: 압축 교사(--kd-teacher-tag) > dense_best > dense
        if a.kd_teacher_tag:
            kd_teacher = str(paths.resolve_ckpt(preset, a.data, tokstr, a.kd_teacher_tag))
        else:
            kd_teacher = str(dense_best_ck if a.kd_best else dense_ck)
        # 부모초기화 소스: --init-from-tag 주면 base_{TAG}.pt, 아니면 base_dense.pt
        init_src = (str(paths.resolve_ckpt(preset, a.data, tokstr, a.init_from_tag)) if a.init_from_tag
                    else (str(dense_ck) if a.init_from else None))
        train(preset, a.arch, a.data, n_tok, a.steps, a.micro_bs, a.seq, a.accum,
              a.lr, a.eval_every, a.resume, ckpt, a.compile,
              sched=a.sched, ema=a.ema, early_stop=a.early_stop,
              init_from=init_src,
              kd=(kd_teacher if a.kd else False),
              kd_alpha=a.kd_alpha, kd_temp=a.kd_temp,
              lora_rank=a.lora_rank, lora_bits=a.lora_bits, mlp_film=a.mlp_film,
              tag=a.tag, tokstr=tokstr, compile_mode=a.compile_mode, mlp_group=a.mlp_group,
              micro_group=a.micro_group, opt_dtype=a.opt_dtype,
              ema_start=a.ema_start, center_weights=a.center_weights, decay_from=a.decay_from,
              snapshots=([_tok(x) for x in a.snapshot_at.split(',')] if a.snapshot_at else None),
              use_ternary_kernel=a.ternary_kernel, ternary_kernel_triton=a.ternary_kernel_triton,
              kd_cache=a.kd_cache, kd_topk=a.kd_topk,
              kd_every=a.kd_every, kd_dynamic=a.kd_dynamic, sparse34=a.sparse34,
              pool_tokens=pool_tok, exact_cache=a.exact_cache,
              anneal_end=a.anneal_end, decay_frac=a.decay_frac, seed=a.seed,
              anneal_shape=a.anneal_shape, anneal_start=a.anneal_start,
              arenas=a.arenas, arena_lambda=a.arena_lambda, arena_end=a.arena_end,
              doc_filter=a.doc_filter, doc_min_chars=a.doc_min_chars,
              lora_decay=a.lora_decay, emb_rank=a.emb_rank,
              kd_teacher_infer=a.kd_teacher_infer)

    elif a.cmd == "all":
        from .train import train
        from .eval import compare
        import json as _json
        dlog = paths.RUNS / "logs" / f"{base}_dense.json"
        # dense 재사용: 같은 (preset,data,steps) 의 dense 로그·체크포인트가 있으면 재학습 생략.
        # (seq·lr 은 로그에 있으면 함께 대조. --force-dense 로 강제 재학습.)
        reuse = False
        if not a.force_dense and dense_ck.exists() and dlog.exists():
            try:
                dj = _json.loads(dlog.read_text())
                reuse = (dj.get("preset") == preset and dj.get("data") == a.data
                         and dj.get("steps") == a.steps
                         and dj.get("seq", a.seq) == a.seq
                         and abs(dj.get("lr", a.lr) - a.lr) < 1e-12)
            except Exception:
                reuse = False
        for arch in ("dense", "tied"):
            if arch == "dense" and reuse:
                dj = _json.loads(dlog.read_text())
                print("\n" + "#" * 68 + "\n#  dense (재사용)\n" + "#" * 68)
                print(f"[dense] 기존 학습 재사용: val {dj['final']['val_loss']:.4f} "
                      f"(preset={dj.get('preset')} data={dj.get('data')} steps={dj.get('steps')}). "
                      f"재학습하려면 --force-dense")
                continue
            print("\n" + "#" * 68 + f"\n#  {arch}\n" + "#" * 68)
            is_tied = arch == "tied"
            train(preset, arch, a.data, n_tok, a.steps, a.micro_bs, a.seq, a.accum,
                  a.lr, a.eval_every, a.resume, ckpt, a.compile,
                  sched=a.sched, ema=a.ema, early_stop=a.early_stop,
                  init_from=(str(dense_ck) if (is_tied and a.init_from) else None),
                  kd=((str(dense_best_ck) if a.kd_best else str(dense_ck)) if (is_tied and a.kd) else False),
                  kd_alpha=a.kd_alpha, kd_temp=a.kd_temp,
                  lora_rank=(a.lora_rank if is_tied else 0), lora_bits=a.lora_bits,
                  mlp_film=(a.mlp_film if is_tied else False), tokstr=tokstr,
                  compile_mode=a.compile_mode, mlp_group=(a.mlp_group if is_tied else None),
                  ema_start=a.ema_start, center_weights=(a.center_weights if is_tied else False),
                  decay_from=a.decay_from, pool_tokens=pool_tok, exact_cache=a.exact_cache)
        print(); compare(base)

    elif a.cmd == "eval":
        import torch
        from .data import prepare, Loader
        from .eval import evaluate
        from .infer import load_model
        meta = prepare(a.data, n_tok)
        ckp = a.ckpt_path or str(paths.resolve_ckpt(preset, a.data, tokstr,
                                                    a.tag if a.tag else a.arch))
        model, cfg, device = load_model(a.arch, ckp)
        # ★P031 — 체크포인트의 cfg 위에 **추론 전용** 설정만 덮어쓴다. 가중치는 그대로다.
        if a.infer_repeat != 1.0 or a.repeat_kv_reuse:
            cfg.infer_repeat = a.infer_repeat
            cfg.repeat_where = a.repeat_where
            cfg.repeat_kv_reuse = a.repeat_kv_reuse
            sch = model.visit_schedule()
            print(f"[P031] infer_repeat={a.infer_repeat} where={a.repeat_where} "
                  f"kv_reuse={a.repeat_kv_reuse} -> 층 통과 {len(sch)}회"
                  f"(기준 {cfg.n_layers}회), middle {len(sch) - cfg.n_prelude - cfg.n_coda}회")
            print("[P031] 메모리는 R 과 무관하게 동일하다 — 늘어나는 것은 연산과 지연뿐이다.")
        print(model.report())
        va = Loader("val", a.micro_bs, a.seq, device, meta["dir"], seed=99)
        print(evaluate(model, va, 100, device))

    elif a.cmd == "compare":
        from .eval import compare
        compare(base, a.tag, a.vs)

    elif a.cmd == "lrfind":
        from .train import lr_find
        grid_lrs = tuple(float(x) for x in a.lrs.split(","))
        lr_find(method=a.method, preset=preset, arch=a.arch, data=a.data,
                n_tokens=a.tokens, micro_bs=a.micro_bs, seq=a.seq, accum=a.accum,
                ckpt=ckpt, range_steps=a.lrfind_steps, lr_min=a.lr_min, lr_max=a.lr_max,
                grid_lrs=grid_lrs)

    elif a.cmd == "generate":
        from .infer import generate
        if not a.prompt:
            p.error("generate 에는 --prompt 가 필요합니다")
        gckp = a.ckpt_path or str(paths.RUNS / "ckpt" / (f"{base}_{a.tag}.pt" if a.tag else f"{base}_{a.arch}.pt"))
        if a.check_cache:
            from tokenizers import Tokenizer
            from .data import tokenizer_path
            from .infer import load_model, check_cache_equivalence
            m, cfg_, dev = load_model(a.arch, gckp)
            tk = Tokenizer.from_file(str(tokenizer_path(a.data)))
            ok = check_cache_equivalence(m, cfg_, tk, a.prompt, max_new=a.max_new, device=dev)
            raise SystemExit(0 if ok else 1)
        generate(a.prompt, arch=a.arch, data=a.data, max_new=a.max_new,
                 temperature=a.temp, top_k=a.top_k, ckpt_path=gckp,
                 use_cache=not a.no_cache, stop_at_eos=not a.no_eos_stop)


if __name__ == "__main__":
    main()
