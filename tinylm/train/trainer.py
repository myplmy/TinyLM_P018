"""학습 루프: 삼진 어닐 + LR 스케줄(cosine/wsd) + NaN 가드 + EMA + 베스트 체크포인트
+ train-val 모니터 + (선택) 조기 종료 / 부모초기화 / KD.

v6 효율/실험: WSD, EMA, best.pt, 조기종료, dense 부모초기화, dense 교사 KD, 층별 LoRA.
"""
from __future__ import annotations

import json
import math
import time

import torch
import torch.nn.functional as F

from .. import paths
from ..config import build_config
from ..model import TiedMLPTransformer
from ..data import prepare, Loader
from ..eval import evaluate
from .init_utils import init_from_dense, load_dense

CKPT = paths.RUNS / "ckpt"
LOGS = paths.RUNS / "logs"


def _lr_factor(s, warm, steps, sched, decay_frac=0.2):
    # (P026) decay_frac 은 이제 호출자(train)가 --decay-frac 으로 넘긴다. cooldown-QAT 정렬 실험용.
    """cosine / wsd(긴 plateau+감쇠) / stable(warmup+평탄, plateau 생성용) /
    decay(워밍업 없이 peak→0.1 cooldown, plateau에서 분기)."""
    if sched == "decay":                          # cooldown-only (decay-branch)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / max(steps, 1)))
    if s < warm:
        return (s + 1) / warm
    p = (s - warm) / max(steps - warm, 1)
    if sched == "stable":                         # plateau: 감쇠 없이 평탄
        return 1.0
    if sched == "wsd":
        if p < 1.0 - decay_frac:
            return 1.0
        q = (p - (1.0 - decay_frac)) / decay_frac
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * q))
    return 0.1 + 0.45 * (1 + math.cos(math.pi * p))


def train(preset, arch, data, n_tokens, steps, micro_bs, seq, accum, lr, eval_every,
          resume=False, ckpt=True, compile_=False, *,
          sched="cosine", ema=0.0, early_stop=0, init_from=None,
          kd=False, kd_alpha=0.5, kd_temp=2.0, lora_rank=0, lora_bits=2, mlp_film=False,
          tag=None, tokstr=None, compile_mode="default", mlp_group=None, micro_group=None,
          opt_dtype="fp32", ema_start=0.0,
          center_weights=False, decay_from=None, snapshots=None,
          use_ternary_kernel=False, ternary_kernel_triton=False,
          kd_cache=False, kd_topk=16, kd_every=1, kd_dynamic=False, sparse34=False,
          pool_tokens=None, exact_cache=False, anneal_end=0.60, decay_frac=0.2, seed=1337,
          anneal_shape="linear", anneal_start=None,
          arenas=False, arena_lambda=0.1, arena_end=0.9,
          doc_filter=False, doc_min_chars=50_000, lora_decay=0.0, emb_rank=None,
          kd_teacher_infer=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 시드: 기본 1337 = 종전 하드코딩값(무변). --seed 로 재현 노이즈 σ 실측에 쓴다.
    #   ★val 로더 시드는 아래에서 99 로 **고정**한다 — val crop 이 런마다 바뀌면 비교 자체가 무효다.
    torch.manual_seed(seed)
    if device == "cuda":                        # 저비용 성능 스위치
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True   # 고정 shape → cuDNN 오토튜너
        try:
            torch.set_float32_matmul_precision("high")   # fp32 matmul을 TF32로
        except Exception:
            pass
    CKPT.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)

    # 데이터 풀(캐시)은 pool_tokens 로 학습길이·이름과 분리 가능. 미지정이면 기존처럼 n_tokens.
    #   토큰스윕(P007 클린): 모든 예산을 동일 600M 풀에서 샘플 → pool_tokens=600M, exact_cache=True.
    # ★2026-08-06 — `doc_filter` 를 학습 경로에도 전달한다. 종전에는 `prepare` 서브커맨드만
    #   이 인자를 받아, **필터 캐시를 만들 수는 있는데 그것으로 학습할 방법이 없었다**
    #   (결과 023 §2 의 "분리해서 쓰는 것과 분리한 것을 읽는 것은 별개 작업" 과 같은 계열).
    meta = prepare(data, int(pool_tokens) if pool_tokens else n_tokens, exact=exact_cache,
                   doc_filter=doc_filter, doc_min_chars=doc_min_chars)
    cfg = build_config(preset, arch, seq, ckpt)
    if mlp_group and arch == "tied":            # g 스윕용 오버라이드(P003)
        assert cfg.n_middle % mlp_group == 0, f"n_middle {cfg.n_middle} % g {mlp_group} != 0"
        cfg.mlp_group = mlp_group
    if micro_group is not None:                 # (P051) 삼진 alpha 그룹 크기 오버라이드
        # ★`__post_init__` 은 프리셋 값으로 이미 돌았으므로 **여기서 같은 불변식을 다시 검사**한다.
        #   빠뜨리면 TLinear 생성 시점의 assert 로 죽는데, 그때는 어느 층인지가 안 보인다.
        assert micro_group > 0 and cfg.dim % micro_group == 0, \
            f"dim {cfg.dim} % micro_group {micro_group} != 0"
        assert cfg.ffn_dim % micro_group == 0, \
            f"ffn_dim {cfg.ffn_dim} % micro_group {micro_group} != 0"
        if getattr(cfg, "sparse34", False):
            assert micro_group % 4 == 0, "sparse34 는 micro_group 이 4의 배수여야 함(3:4 블록)"
        old_g = cfg.micro_group
        cfg.micro_group = micro_group
        # 저장 bpw 의 scale 항 = SCALE_BITS(16) / g. 코드 항(log2 3 또는 1.25)은 안 바뀐다.
        print(f"[micro_group] alpha 그룹 {old_g} -> {micro_group} — "
              f"scale 항 {16/old_g:.4f} -> {16/micro_group:.4f} bpw "
              f"(코드 항은 불변). ★KD 교사는 자기 cfg 로 로드되므로 g{old_g} 그대로다")
    if emb_rank is not None:                    # (P046) 임베딩 병목 E 오버라이드
        assert emb_rank > 0, "--emb-rank 는 양수여야 한다(0=비활성은 지원하지 않는다)"
        # ★로짓 랭크가 E 로 제한된다. 임베딩·lm_head 가 선형으로 줄고 품질 영향은 미지다.
        cfg.emb_rank = emb_rank
        print(f"[emb] emb_rank 오버라이드 {emb_rank} (프리셋 기본 256) — "
              f"임베딩 파라미터 {(32768*emb_rank + emb_rank*cfg.dim)/1e6:.2f}M")
    cfg.mlp_lora_rank, cfg.mlp_lora_bits = lora_rank, lora_bits
    cfg.mlp_film = mlp_film
    cfg.center_weights = center_weights
    cfg.use_ternary_kernel = use_ternary_kernel
    cfg.ternary_kernel_triton = ternary_kernel_triton
    # ★프리셋이 sparse34=True 로 정의될 수 있다(m100R1a). `--sparse34` 는 **켤 수만** 있게 한다 —
    #   그냥 대입하면 프리셋 값을 조용히 덮어써서 3:4 없이 학습된다(감지 어려운 사고).
    cfg.sparse34 = bool(sparse34) or bool(getattr(cfg, "sparse34", False))
    if sparse34:
        assert cfg.micro_group % 4 == 0, "sparse34 는 micro_group 이 4의 배수여야 함"
        if use_ternary_kernel:
            raise SystemExit("[sparse34] 커스텀 삼진 커널 경로는 3:4 미구현 — "
                             "--sparse34 는 표준(F.linear) 경로에서만 사용하세요(커널 병용 금지).")
        print("[sparse34] 3:4 희소 삼진(1.25bpw) 활성 — 각 4-블록 |w|최소 1개 0강제")
    model = TiedMLPTransformer(cfg).to(device)

    if init_from:
        init_from_dense(model, init_from, device)

    teacher = None
    if kd and not kd_cache:
        teacher, _ = load_dense(kd if isinstance(kd, str) else CKPT / "dense.pt", device)
        teacher.eval(); teacher.set_anneal(1.0)
        for p in teacher.parameters():
            p.requires_grad_(False)
        # ★★P042(2026-08-07) — **교사는 학습하지 않는데 학습하는 것처럼 돌고 있었다.**
        #   `forward()` 는 매 호출 `refresh_quant()` 로 삼진 가중치를 다시 만든다. 학생은
        #   latent 가 스텝마다 바뀌므로 **반드시 그래야 하지만**, 교사는 가중치가 고정이다.
        #   추론용으로 이미 구현이 끝난 두 함수를 붙인다:
        #     `freeze_quant()`  — 삼진 가중치를 **한 번만** 계산하고 재사용
        #     `drop_latent()`   — fp32 latent 해제(상주 2벌 → 1벌). dense 교사 기준 약 472.5MB
        #   ⚠️ **기본 on 이 아니다.** `--kd-teacher-infer` 로 켠다 — 기본 off = 비트 동일.
        #     결과가 달라지면 그건 구현 버그이고, P042 단계0 이 그것만 검사한다.
        if kd_teacher_infer:
            teacher.freeze_quant()
            teacher.drop_latent()
            print(f"[kd] ★교사 추론 모드(P042): freeze_quant + drop_latent 적용 — "
                  f"매 스텝 refresh_quant 를 건너뛰고 latent 를 해제한다")
            print(f"[kd] ⚠️ 로짓은 **비트 동일해야 한다**. 다르면 구현 버그다(게이트 G0)")
        if compile_:
            teacher = torch.compile(teacher)   # KD 가속: 교사 forward도 컴파일(eager→컴파일)
        print(f"[kd] 교사 로드 완료 (alpha={kd_alpha}, T={kd_temp})")

    if decay_from:                              # WSD decay-branch: plateau에서 분기
        from .init_utils import _strip
        st = torch.load(decay_from, map_location=device)
        model.load_state_dict(_strip(st["model"]))
        print(f"[decay] plateau 로드 -> cooldown {steps}스텝  <- {decay_from}")

    if compile_ and (use_ternary_kernel or ternary_kernel_triton):
        # 커널 경로는 torch.compile 과 근본적으로 상성이 나쁘다:
        #   - Triton 커스텀 커널은 dynamo 의 identify_mutated_tensors 가 커널 IR 을 파싱하다
        #     CompilationError(IndexError: Function argument index out of range) 로 크래시.
        #   - anneal>=1.0 데이터 의존 분기 → 그래프 브레이크 + quant_anneal 값 가드 재컴파일 폭주.
        # 따라서 두 옵션 동시 사용은 조용히 느려지거나 크래시하므로 **여기서 학습을 중단**한다.
        raise SystemExit(
            "[중단] --ternary-kernel[-triton] 은 --compile 과 함께 쓸 수 없습니다.\n"
            "  이유: dynamo 가 커스텀 Triton 커널 IR 파싱 중 크래시(IndexError)하거나, anneal 데이터의존\n"
            "        분기로 재컴파일 폭주가 발생합니다(커널은 별도 최적화 경로라 compile 대상이 아님).\n"
            "  조치: 커널 벤치는 --compile 을 빼고 실행하세요. 예)\n"
            "        python run100m.py train ... --ternary-kernel --ternary-kernel-triton   (--compile 제거)")
    if compile_:
        if compile_mode == "reduce-overhead":
            print("[compile] 주의: reduce-overhead(CUDA그래프)는 임베딩 타잉과 충돌해 "
                  "backward에서 크래시할 수 있습니다. 크래시 시 --compile-mode default 로 재실행하세요.")
        print(f'[compile] mode={compile_mode} — 첫 스텝은 수 분 걸릴 수 있습니다')
        model = torch.compile(model, mode=compile_mode)
    tokstr = tokstr or (f"{int(n_tokens)//1_000_000}M" if n_tokens >= 10**6 else str(int(n_tokens)))
    _base = f"{preset}_{data}_{tokstr}"
    name = f"{_base}_{tag}" if tag else f"{_base}_{arch}"   # 스케일별 이름(클로버·오염 방지)
    label = tag or arch                                    # 로그 표시용(예: t_kd_g8 / tied)
    print(model.report())
    # 배포 메모리 정확값을 결과 json 에 기록 → compare 가 "전체×단일bpw" 근사를 쓰지 않게 한다.
    #   (sparse34 는 삼진분에만 1.25bpw 라서 근사가 과소·비율 과대였다. 결과 008 §2-(6))
    _mem = model.mem_breakdown()
    _memall = model.mem_report_all()   # (2026-07-31) packed(B)/packed_container(C)/runtime 병기
    eff = micro_bs * accum * seq
    print(f"[{label}] device={device}  {steps}step x {eff/1e3:.0f}K tok = "
          f"{steps*eff/1e6:.0f}M 토큰  (sched={sched} ema={ema} lora_r={lora_rank}"
          f"{f' seed={seed}' if seed != 1337 else ''})\n")

    # ★★P022B 단계2(2026-08-07) — 옵티마이저 상태 정밀도.
    #   DeepSeek-V3 §3.3.3 은 AdamW 1·2차 모멘트를 **BF16** 으로 들고 master weight·gradient 는
    #   FP32 로 유지해 *"관측 가능한 성능 저하 없음"* 을 1T 토큰 규모로 보고했다.
    #   우리 학습 파라미터 약 63.5M × 2벌 × 4B = **약 508MB** → bf16 이면 **약 254MB 회수**.
    #   ⚠️ **기본 `fp32` = 종전 `torch.optim.AdamW(fused=True)` = 비트 동일.**
    #     `fp32c` 는 **같은 수식의 우리 구현**(자기검증용) — 구현 위험과 dtype 위험을 분리한다.
    if opt_dtype == "fp32":
        opt = torch.optim.AdamW(model.param_groups(lr), betas=(0.9, 0.95), eps=1e-8,
                                fused=(device == "cuda"))   # ①: optimizer update 단일 커널
    else:
        from .adamw_bf16 import AdamWLowPrec
        _sd = torch.bfloat16 if opt_dtype == "bf16" else torch.float32
        opt = AdamWLowPrec(model.param_groups(lr), betas=(0.9, 0.95), eps=1e-8, state_dtype=_sd)
        _n = sum(p.numel() for g in opt.param_groups for p in g["params"])
        print(f"[opt] ★AdamWLowPrec state_dtype={opt_dtype} (P022B 단계2) — 학습 파라미터 "
              f"{_n/1e6:.2f}M, 상태 2벌 예상 {_n*2*_sd.itemsize/1e6:.1f}MB "
              f"(fp32 대비 {_n*2*(4-_sd.itemsize)/1e6:+.1f}MB)")
        if opt_dtype == "bf16":
            print(f"[opt] ⚠️ master weight·gradient 는 fp32 유지, **산술도 fp32**. "
                  f"상태만 bf16 으로 저장한다(결정론적 반올림, stochastic 아님)")
        else:
            print(f"[opt] ⚠️ fp32c 는 **자기검증 모드**다 — 융합 커널이 아니라 파이썬 루프이고, "
                  f"`fp32` 와 인쇄 4자리가 맞아야 bf16 결과를 dtype 탓으로 귀속할 수 있다")
        if resume:
            print(f"[opt] ⚠️ --resume 은 권장하지 않는다 — load_state_dict 가 상태를 파라미터 "
                  f"dtype 으로 캐스팅한다(오버라이드로 되돌리지만 조건이 흐려진다)")
    base_lrs = [g["lr"] for g in opt.param_groups]
    warm = 0 if sched == "decay" else max(5, min(steps // 10, 100))
    # (P026) cooldown-QAT 스케줄 정렬 표시. anneal_end=완전삼진 도달, decay_start=LR 감쇠 시작.
    assert 0.0 < anneal_end <= 1.0, f"--anneal-end 는 (0,1] 이어야 함: {anneal_end}"
    # (P035) 어닐 시작점. 미지정이면 종전 하드코딩식 그대로 → 기본 동작 무변.
    a0 = (warm / steps + 0.05) if anneal_start is None else float(anneal_start)
    assert anneal_shape in ("linear", "step"), f"--anneal-shape: {anneal_shape}"
    assert 0.0 <= a0 < 1.0, f"--anneal-start 는 [0,1) 이어야 함: {a0}"
    if anneal_shape == "linear":
        assert a0 < anneal_end, (f"linear 어닐은 anneal_start({a0:.3f}) < anneal_end({anneal_end}) "
                                 f"여야 램프가 성립한다")
    if anneal_end != 0.60 or decay_frac != 0.2 or anneal_shape != "linear" or anneal_start is not None:
        _dstart = (1.0 - decay_frac) if sched == "wsd" else (0.0 if sched != "stable" else 1.0)
        # step 어닐의 '완전삼진 도달'은 anneal_end 가 아니라 전이점(a0)이다 — 정렬 판정도 그쪽이다.
        _full = a0 if anneal_shape == "step" else anneal_end
        print(f"[sched] shape={anneal_shape}  anneal_start={a0:.2f}(step~{int(a0*steps)})  "
              f"anneal_end={anneal_end:.2f}(step~{int(anneal_end*steps)})  "
              f"완전삼진={_full:.2f}(step~{int(_full*steps)})  "
              f"decay_frac={decay_frac:.2f}  LR감쇠시작={_dstart:.2f}(step~{int(_dstart*steps)})"
              f"  {'정렬됨' if sched == 'wsd' and abs(_full - _dstart) < 1e-6 else '미정렬'}")
    # 토큰 마크별 명명 스냅샷: {마크토큰: 라벨}. decay-branch 소스로 재사용.
    snap_steps = {}
    for tok in (snapshots or []):
        stp = max(1, round(int(tok) / eff))
        lbl = f"{int(tok)//1_000_000}M" if int(tok) >= 10**6 else str(int(tok))
        snap_steps[stp] = lbl

    start = 0
    ck = CKPT / f"{name}.pt"
    ck_best = CKPT / f"{name}_best.pt"
    if resume and ck.exists():
        st = torch.load(ck, map_location=device)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); start = st["step"]
        print(f"[{arch}] step {start}에서 재개")

    kd_reader = None
    if kd_cache:
        from .kd_cache import KdCacheReader
        kd_reader = KdCacheReader(_base, kd_topk, micro_bs, seq)
        kd_reader.seek_step(start, accum)
        print(f"[kd-cache] 오프라인 KD 캐시 사용 (top{kd_topk}, 교사 forward 없음)")

    params = [p for p in model.parameters() if p.requires_grad]
    shadow = None                               # P1: 후반부(ema_start 이후)에만 지연 생성·누적
    ema_start_step = int(ema_start * steps)

    def _swap_in_ema():
        backup = [p.detach().clone() for p in params]
        for p, sh in zip(params, shadow):
            p.data.copy_(sh)
        return backup

    def _swap_out(backup):
        for p, b in zip(params, backup):
            p.data.copy_(b)

    # reduce-overhead(CUDA그래프)는 micro-step마다 그래프 출력 버퍼를 재사용하므로,
    # grad accum에서 각 forward 전에 step 경계를 표시해 backward가 덮인 버퍼를 참조하지 않게 한다.
    _cudagraph_step = (getattr(torch.compiler, "cudagraph_mark_step_begin", None)
                       if (compile_ and compile_mode == "reduce-overhead") else None)

    bpt = meta.get("bytes_per_token")           # bits-per-byte용(토크나이저 무관 지표)
    # Loader 는 torch RNG 와 **별도** np rng 를 쓴다 → --seed 가 데이터 순서에도 반영되도록 파생시킨다.
    #   train: 시드에 따라 크롭 순서가 바뀐다(σ 측정에서 원하는 변동).
    #   val  : **항상 99 고정.** 여기를 흔들면 서로 다른 크롭에서 val loss 를 재게 되어 런 비교가 깨진다.
    #   (mod 2^31 = np.random.default_rng 이 음수 시드를 거부하므로. seed=1337 이면 정확히 1234)
    tr = Loader("train", micro_bs, seq, device, meta["dir"],
                seed=(1234 + (seed - 1337)) % (2 ** 31))
    va = Loader("val", micro_bs, seq, device, meta["dir"], seed=99)
    hist, t0, gmax, gpeak, n_skip = [], time.time(), 0.0, 0.0, 0
    # ★VRAM 자동 계측(P021B 교훈): 사람이 nvidia-smi 를 눈으로 보게 하면 반드시 빠뜨린다.
    #   여기서 피크를 리셋하고 종료 시 json 에 기록한다. compile/모델 로드 뒤라 학습 피크만 잡힌다.
    #   주의: nvidia-smi 표시값 ≈ reserved + CUDA 컨텍스트(~0.4~0.8GB) 이므로 reserved 가 하한이다.
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best_val, best_step, since_improve = float("inf"), 0, 0
    n_kd_fwd = 0                                 # 실제 교사 forward를 수행한 스텝 수(가속 측정용)

    def _do_eval(step, train_loss, train_ce=None, kd_this=False):
        """★`train_loss` 는 KD 스텝에서 **혼합손실**(α·CE+(1-α)·KL·T²)이라 val 과 단위가 다르다.
        그래서 `train_ce`(항상 순수 CE)를 따로 받아 **val-CE 로 비교**한다.
        이 구분이 없던 시절 REVIEW1 로그에서 마지막 eval 만 `val-train +1.69` 로 튀어
        과적합처럼 보였다 — 실제로는 그 스텝이 KD 스텝이라 train 이 혼합손실이었을 뿐이다.
        (주기 eval 은 s≡3 mod 4 = 비KD 스텝, 최종 eval 은 s=2288≡0 = KD 스텝에 걸린다.)
        """
        nonlocal best_val, best_step, since_improve
        m = evaluate(model, va, 50, device, bytes_per_token=bpt); m["ema"] = False   # 주 지표 = raw
        ce = train_ce if train_ce is not None else train_loss
        line = (f"    >> val_loss {m['val_loss']:.4f}  ppl {m['ppl']:.2f}  "
                f"(train_ce {ce:.3f}, val-CE {m['val_loss']-ce:+.3f}"
                + (f", 혼합손실 {train_loss:.3f}[KD스텝]" if kd_this else "") + ")")
        if shadow is not None:                                  # EMA는 부가 표시
            backup = _swap_in_ema()
            me = evaluate(model, va, 50, device)
            _swap_out(backup)
            m["val_ema"] = me["val_loss"]; line += f"  [ema {me['val_loss']:.4f}]"
        if "bpb" in m: line += f"  bpb {m['bpb']:.3f}"
        m.update(step=step, train_loss=train_loss, train_ce=ce, kd_step=bool(kd_this),
                 gap=m["val_loss"] - ce)   # gap 은 CE 기준(혼합손실과 섞지 않는다)
        hist.append(m); print(line)
        blob = {"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step, "cfg": cfg.__dict__}
        if shadow is not None:
            blob["ema"] = [sh.detach().cpu() for sh in shadow]
        torch.save(blob, ck)
        if m["val_loss"] < best_val - 1e-4:                     # best 판정 = raw
            best_val, best_step, since_improve = m["val_loss"], step, 0
            torch.save(blob, ck_best)
            print(f"       best 갱신 {best_val:.4f} -> {ck_best.name}")
        else:
            since_improve += 1
        return m

    for s in range(start, steps):
        if sched == "decay":                    # 이미 학습된 plateau라 완전 삼진 유지
            anneal = 1.0
        elif anneal_shape == "step":
            # (P035) 계단 어닐: a0 까지 full-precision(anneal 0), 그 지점에서 1.0 으로 급전이.
            #   논문(arXiv:2509.22935)이 상정한 "FP 학습 → 별도 QAT" 를 **인위적으로 재현**한다.
            #   P026(결과 015)은 끝점만 옮겨 정렬 효과를 못 봤는데, 우리 선형 램프에는
            #   제거할 중복이 애초에 없었을 수 있다. 그 가설을 검정하려면 중복을 만들어야 한다.
            anneal = 1.0 if (s / steps) >= a0 else 0.0
        else:
            # (P026) anneal_end = 완전삼진 도달 지점(진행률). 기본 0.60 = 종전 하드코딩값.
            #   cooldown-QAT 가설: 이 지점을 LR 감쇠 시작(wsd면 1-decay_frac)과 정렬하면
            #   "FP 학습 후 별도 QAT" 의 중복 업데이트가 사라져 같은 val 을 더 적은 steps 에 도달.
            #   a0 는 위(스케줄 진단 블록)에서 한 번만 계산한다 — --anneal-start 미지정이면 종전값.
            anneal = min(1.0, max(0.0, (s / steps - a0) / max(anneal_end - a0, 1e-6)))
        model.set_anneal(anneal)
        if arenas:
            # ★P036 Arenas — λ_t 를 λ_0 에서 0 으로 선형 감쇠시키고 arena_end 이후 0 으로 고정한다.
            #   논문의 "annealing" 이 가리키는 것이 이 λ 스케줄이다(우리 quant_anneal 과 별개).
            #   끝에서 정확히 0 이 되어야 **배포 시 순수 삼진**이 되고 추론 오버헤드가 0 이다.
            model.set_arena(arena_lambda * max(0.0, 1.0 - (s / steps) / max(arena_end, 1e-6)))
        if lora_decay and lora_rank > 0:
            # ★P008 점진적 타잉 — LoRA 출력 스케일 s(t) 를 1 → 0 으로 선형 감쇠시키고
            #   진행률 `lora_decay` 지점 이후 **정확히 0** 으로 고정한다.
            #   s=0 이 되어야 배포 시 LoRA 가 사라지고 **메모리 대가가 0** 이 된다.
            #   (고정 LoRA 는 추론에 남아 감축비를 1.82× → 1.71× 로 깎는다)
            model.set_lora_scale(max(0.0, 1.0 - (s / steps) / max(lora_decay, 1e-6)))
        f = _lr_factor(s, warm, steps, sched, decay_frac)
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * f

        # Skip-Forward / Dynamic KD: 이 스텝에서 교사 forward를 수행할지 결정(P017).
        #   kd_every=1 → 매 스텝(기존). kd_every=K → K스텝마다 1회(교사 연산 1/K).
        #   kd_dynamic → 간격을 1→K 로 선형 증가(초반 촘촘한 KD, 후반 성김).
        kd_this = teacher is not None
        if teacher is not None and kd_every > 1:
            cur_every = (max(1, round(1 + (kd_every - 1) * s / steps))
                         if kd_dynamic else kd_every)
            kd_this = (s % cur_every == 0)
        if kd_this and teacher is not None:
            n_kd_fwd += 1

        model.train()
        tot, tot_ce = 0.0, 0.0        # tot=실제 최적화 손실(KD면 혼합), tot_ce=항상 순수 CE
        for _ in range(accum):
            x, y = tr()
            if _cudagraph_step is not None:
                _cudagraph_step()
            with torch.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
                logits = model(x)
                ce = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
                if kd_reader is not None:               # 오프라인 KD(캐시 top-k)
                    from .kd_cache import kd_cache_loss
                    tv, ti = kd_reader.next(device)
                    kl = kd_cache_loss(logits, tv, ti, cfg.vocab_size, kd_temp)
                    loss = ((1 - kd_alpha) * ce + kd_alpha * kl) / accum
                elif teacher is not None and kd_this:   # 온라인 KD(교사 forward, skip-forward 반영)
                    with torch.no_grad():
                        tlog = teacher(x)
                    T = kd_temp
                    kl = F.kl_div(F.log_softmax(logits.reshape(-1, cfg.vocab_size) / T, -1),
                                  F.softmax(tlog.reshape(-1, cfg.vocab_size) / T, -1),
                                  reduction="batchmean") * (T * T)
                    loss = ((1 - kd_alpha) * ce + kd_alpha * kl) / accum
                else:
                    loss = ce / accum
            loss.backward()
            tot += loss.item()
            tot_ce += ce.item() / accum      # KD 여부와 무관하게 순수 CE 를 따로 누적
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if not torch.isfinite(gn):
            opt.zero_grad(set_to_none=True); model.clear_quant()
            n_skip += 1
            print(f"  [skip] step {s}: non-finite grad ({n_skip}회째)  "
                  f"anneal {model.cfg.quant_anneal:.2f}  lr {opt.param_groups[1]['lr']:.2e}")
            if n_skip > 20:
                print("  !! non-finite 20회 초과. LR을 낮추고 재시작하세요."); break
            continue

        gpeak = max(gpeak, float(gn))
        if s >= warm:
            gmax = max(gmax, float(gn))
        opt.step(); opt.zero_grad(set_to_none=True)
        if ema > 0 and s >= ema_start_step:     # P1: 감쇠 구간의 좋은 가중치만 평균
            with torch.no_grad():
                if shadow is None:
                    shadow = [p.detach().clone() for p in params]
                else:
                    for sh, p in zip(shadow, params):
                        sh.mul_(ema).add_(p.detach(), alpha=1 - ema)
        model.clear_quant()

        if s % 10 == 0 or s == steps - 1:
            el = time.time() - t0
            # ★비KD 스텝에서는 loss 와 ce 가 **구조적으로 같은 값**이다(loss = ce/accum 을 합산).
            #   그래서 두 열을 항상 찍으면 정보가 0 인 열이 하나 생기고, KD k4 런에서는 loss 열이
            #   4스텝마다 혼합손실 ↔ CE 로 진동해 과적합처럼 보였다(결과 012 §4 의 'loss 2↔4 진동').
            #   → **런 간 비교 가능한 ce 를 항상 앞에** 두고, 최적화 손실은 **다를 때만** 표기한다.
            _mix = (f"  loss {tot:.4f}[KD혼합]"
                    if (kd_this and teacher is not None) or kd_reader is not None else "")
            print(f"  step {s:>5}/{steps}  ce {tot_ce:.4f}{_mix}  |g| {gn:.2f}  "
                  f"anneal {model.cfg.quant_anneal:.2f}  lr {opt.param_groups[1]['lr']:.2e}  "
                  f"{el/(s-start+1)*1000:.0f} ms/step")
        if (s + 1) % eval_every == 0 or s == steps - 1:
            _do_eval(s + 1, tot, train_ce=tot_ce, kd_this=kd_this)
            if early_stop and since_improve >= early_stop:
                print(f"  [early-stop] {early_stop}회 연속 개선 없음 (best {best_val:.4f} @ {best_step}). 종료.")
                break
        if (s + 1) in snap_steps:               # 토큰 마크 스냅샷(plateau 분기 소스)
            snap = CKPT / f"{name}_snap{snap_steps[s + 1]}.pt"
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": s + 1}, snap)
            print(f"       [snapshot] {snap_steps[s + 1]} 토큰 지점 저장 -> {snap.name}")

    final = evaluate(model, va, 100, device, bytes_per_token=bpt); final["ema"] = False
    if shadow is not None:
        backup = _swap_in_ema(); fe = evaluate(model, va, 100, device); _swap_out(backup)
        final["val_ema"], final["ppl_ema"] = fe["val_loss"], fe["ppl"]
    n_par = sum(p.numel() for p in model.parameters())
    import os as _os
    res = {"arch": arch, "preset": preset, "data": data, "params": n_par, "steps": steps,
           "lr": lr, "seq": seq,
           # ★런 재구성용 조건(이게 없으면 로그만 보고 실험을 재현·중복판정할 수 없다).
           #   docs/EXPERIMENT_BASELINES.md 레지스트리와 exp-preflight 스킬이 이 필드를 읽는다.
           "seed": seed, "micro_bs": micro_bs, "accum": accum, "eff_batch": eff,
           "pool_tokens": int(pool_tokens) if pool_tokens else None,
           "exact_cache": bool(exact_cache),
           "doc_filter": bool(doc_filter),
           # ★L1/L4 (결과 023 §9) — 이 런이 실제로 본 소스별 토큰 비율. "한국어 50%" 를
           #   추정으로 쓰지 않기 위해 런 로그에 박아 둔다. 구 캐시에는 없어 None 이 된다.
           "mix_token_frac": meta.get("mix_token_frac"),
           "mix_exhausted": meta.get("mix_exhausted"),
           "bytes_per_token": meta.get("bytes_per_token"),
           "bytes_per_token_val": meta.get("bytes_per_token_val"),
           "mlp_group": (cfg.mlp_group if getattr(cfg, "tie_mlp", False) else 1),
           "micro_group": int(cfg.micro_group),                    # (P051) 삼진 alpha 그룹 크기
           "opt_dtype": str(opt_dtype),                            # (P022B 단계2) 옵티마이저 상태 정밀도
           "opt_state_mb": (opt.state_bytes() / 1e6                # ★계산값이 아니라 실측(결과 026 교훈)
                            if hasattr(opt, "state_bytes") else None),
           "grad_ckpt": bool(ckpt),
           "kd_teacher": (_os.path.basename(str(kd)) if kd else None),
           "init_from_src": (_os.path.basename(str(init_from)) if init_from else None),
           "tokens": steps * eff, "final": final, "best_val": best_val, "best_step": best_step,
           "history": hist, "grad_max": gmax, "grad_peak_warmup": gpeak, "n_skip": n_skip,
           "sched": sched, "ema": ema, "kd": bool(kd), "init_from": bool(init_from),
           "kd_every": kd_every, "kd_dynamic": bool(kd_dynamic), "kd_fwd_steps": n_kd_fwd,
           "lora_rank": lora_rank, "wall_sec": time.time() - t0,
           "sparse34": bool(sparse34), "bpw": 1.25 if sparse34 else 1.95,
           "anneal_end": anneal_end, "decay_frac": decay_frac,    # (P026) 스케줄 정렬 기록
           "anneal_shape": anneal_shape, "anneal_start": a0,      # (P035) 어닐 형태·시작점
           "lora_decay": float(lora_decay),                       # (P008) LoRA 스케일 어닐
           "emb_rank": int(cfg.emb_rank),                          # (P046) 임베딩 병목 E
           "kd_teacher_infer": bool(kd_teacher_infer),            # (P042) 교사 추론 모드
           "arenas": bool(arenas), "arena_lambda": arena_lambda,  # (P036) Arenas residual
           "arena_end": arena_end,
           # ★저장 메모리 회계 통일(2026-07-31, P034 §5): 정본=B(코드+스케일), 병기=C(+컨테이너).
           #   `deploy_mb` 는 **구 로그 호환 별칭**이며 값은 packed_mb(B) 와 같다.
           #   ⚠️ 2026-07-31 이전 로그의 `deploy_mb` 는 **구 규약(1.95/1.25)** 이라 직접 비교 금지.
           "deploy_mb": _memall["packed_mb"], **_memall,
           # ★학습 VRAM 피크(GB). nvidia-smi ≈ reserved + CUDA 컨텍스트(0.4~0.8GB).
           "vram_alloc_gb": (torch.cuda.max_memory_allocated() / 1024 ** 3) if device == "cuda" else None,
           "vram_reserved_gb": (torch.cuda.max_memory_reserved() / 1024 ** 3) if device == "cuda" else None,
           "tokens_per_microbatch": micro_bs * seq}               # ★M — 속도·VRAM 의 지배 변수
    res["tag"] = name
    (LOGS / f"{name}.json").write_text(json.dumps(res, indent=2))
    kd_note = ""
    if teacher is not None and kd_every > 1:
        kd_note = f", KD forward {n_kd_fwd}/{steps}스텝(≈{n_kd_fwd/max(steps,1)*100:.0f}%)"
    _vram = (f", VRAM {res['vram_reserved_gb']:.2f}GB(reserved)/{res['vram_alloc_gb']:.2f}GB(alloc)"
             if res.get("vram_reserved_gb") else "")
    print(f"\n[{label}] 최종 val_loss {final['val_loss']:.4f}  ppl {final['ppl']:.2f}  "
          f"best {best_val:.4f}@{best_step}  ({n_par/1e6:.1f}M, {(time.time()-t0)/60:.1f}분, "
          f"skip {n_skip}{kd_note}{_vram})")
    if res.get("vram_reserved_gb"):
        print(f"[vram] M(=micro_bs×seq)={micro_bs*seq:,}  peak reserved {res['vram_reserved_gb']:.2f}GB"
              f"  (nvidia-smi 표시값은 여기에 CUDA 컨텍스트 0.4~0.8GB 가 더해진다)")
    return res
