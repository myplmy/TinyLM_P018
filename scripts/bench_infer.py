#!/usr/bin/env python3
"""P030 추론 속도 벤치 — **CPU/GPU tok/s 와 TTFT**. 프로젝트 타깃(저사양 CPU)의 첫 측정.

★왜: 지금까지 측정한 것은 전부 **GPU 학습 속도**(결과 007·P021B·010)다.
   `CLAUDE.md` 첫 줄이 "저사양 CPU·엣지" 인데 **CPU 추론은 0회 측정**했다.
   즉 30.9MB → 11.7MB 로 줄인 것이 실제로 무슨 이득인지 아직 모른다.

★2026-07-31 갱신 (P030 단계1 반영) — 결과 014 는 이 두 결함으로 **무효**였다
   (1) `load_model` 이 삼진화를 고정하지 않아 **매 forward 재양자화**했다(CPU 시간의 ~79%).
       → `freeze_quant()` 로 1회만 계산. sparse34 가 dense 보다 느려 보인 원인이 이것이다.
   (2) **KV 캐시가 없어** 매 토큰 전체 시퀀스를 재계산했다(O(T²)).
       → `use_cache=True` 가 기본. `--no-cache` 로 옛 경로와 대조할 수 있다.
   여전히 남는 한계: 삼진 가중치는 추론 시 **dequant 후 fp/bf16 GEMM** 이라
   저장 MB 는 "저장" 크기다. **실행시 메모리는 P034 가 따로 잰다.**

★2026-07-31 2차 수정 — **표의 MB 가 하드코딩이었다**
   `DEFAULT_MODELS` 에 30.9 / 14.9 / 11.7 이 **리터럴로 박혀** 있었다. 그래서
     · 같은 배치 안에서 이 스크립트는 30.9/14.9/11.7 을, `mem_runtime.py` 는
       27.1/13.1/12.4 를 찍어 **한 로그에 두 개의 진실**이 남았고
     · `--models` 를 주면 mb=0.0 이 되어 비율 절이 통째로 사라졌으며
     · 그 값들이 **2026-07-31 이전 구 규약**이라 회계 통일(안 B)과 어긋났다.
   → 이제 **로드한 모델의 `mem_report_all()` 에서 계산**한다(`CLAUDE.md` 파라미터
     단일 소스 규약). 저장(packed)·상주(runtime)를 **둘 다** 찍는다 — 결과 016 이
     "속도는 저장이 아니라 상주를 따라갈 것"이라고 예측했으므로 둘 다 있어야 읽힌다.

사용법
  python scripts/bench_infer.py                                   # 기본 3모델, CPU+GPU
  python scripts/bench_infer.py --device cpu --threads 1 2 4 8     # 스레드 스윕
  python scripts/bench_infer.py --models mA_g4s34_k4 --max-new 32
"""
from __future__ import annotations
import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ★MB 를 여기 적지 않는다 — 모델을 로드한 뒤 mem_report_all() 로 계산한다.
DEFAULT_MODELS = [
    ("p6d", "dense"),
    ("mC_g8_k4", "tied"),
    ("mA_g4s34_k4", "tied"),
]
PROMPT = "대한민국의 수도 서울은"          # 학습 분포 안. 길이는 --prompt-tokens 로 패딩


def bench_one(model, cfg, tok, prompt, max_new, device, reps, use_cache=True):
    """(tok/s, TTFT_ms) 를 reps 회 재서 중위값. 첫 회는 warmup 으로 버린다."""
    import torch
    from tinylm.infer.generate import sample
    rates, ttfts = [], []
    for r in range(reps + 1):
        # TTFT: 프롬프트 1회 forward
        ids = tok.encode(prompt).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            dev_t = device if isinstance(device, str) else device.type
            with torch.autocast(dev_t, dtype=torch.bfloat16, enabled=(dev_t == "cuda")):
                _ = model(x[:, -cfg.max_seq_len:])
        if device == "cuda":
            torch.cuda.synchronize()
        ttft = (time.perf_counter() - t0) * 1000

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = sample(model, cfg, tok, prompt, max_new=max_new, temperature=0.7,
                   top_k=40, device=device, use_cache=use_cache,
                   stop_at_eos=False)      # ★속도 측정은 항상 max_new 토큰을 다 생성해야 공정
        if device == "cuda":
            torch.cuda.synchronize()
        el = time.perf_counter() - t0
        if r == 0:
            continue                       # warmup 버림
        rates.append(max_new / el)
        ttfts.append(ttft)
    return statistics.median(rates), statistics.median(ttfts)


def main():
    ap = argparse.ArgumentParser(description="P030 추론 속도 벤치")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--device", nargs="*", default=["cuda", "cpu"])
    ap.add_argument("--threads", nargs="*", type=int, default=[0],
                    help="CPU 스레드 수(0=torch 기본). 엣지는 코어가 적으니 1 이 가장 현실적")
    ap.add_argument("--max-new", type=int, default=128,
                    help="KV 캐시 구현 후 기본 128(캐시 이득은 길수록 커진다)")
    ap.add_argument("--no-cache", action="store_true",
                    help="KV 캐시 off. 결과 014 조건 재현용 대조군")
    ap.add_argument("--both-cache", action="store_true",
                    help="캐시 on/off 를 같은 표에 나란히 측정(캐시 이득 배수를 직접 본다)")
    ap.add_argument("--check-cache", action="store_true",
                    help="먼저 캐시 유/무 그리디 출력 일치를 검증한다(불일치면 중단)")
    ap.add_argument("--reps", type=int, default=3, help="반복(중위값). Windows 는 노이즈가 크다")
    ap.add_argument("--data", default="ko-en")
    ap.add_argument("--tokens", default="300M")
    ap.add_argument("--preset", default="m100")
    ap.add_argument("--drop-latent", action="store_true",
                    help="P034 단계2 — latent 해제 상태로 잰다. 상주가 절반이면 속도가 따라오는가")
    ap.add_argument("--int8-store", action="store_true",
                    help="P034 단계3 — int8 저장. **느려질 것으로 예상**(forward 마다 되돌린다)")
    ap.add_argument("--unpack-cache", action="store_true",
                    help="P034 단계3C — int8 언팩을 **유니크 모듈당 1회**로. 타잉 모델만 이득이고 "
                         "**dense 는 안 변해야 한다(대조군)**. --int8-store 와 함께 쓴다")
    ap.add_argument("--infer-repeat", type=float, default=1.0,
                    help="(P030 단계4) middle 통과 배수. **층 수만 바꾸고 파라미터는 고정**한다 — "
                         "결과 016 §10.4 의 '속도는 파라미터가 아니라 층 수를 따라간다' 가설 검정용")
    ap.add_argument("--repeat-where", choices=["front", "back", "even"], default="front")
    a = ap.parse_args()

    import torch
    from tinylm import paths
    from tinylm.infer.generate import load_model
    from tinylm.data import tokenizer_path
    from tokenizers import Tokenizer

    models = ([(t, "dense" if t.startswith(("p6d", "dense", "p12d")) else "tied")
               for t in a.models] if a.models else DEFAULT_MODELS)
    tok = Tokenizer.from_file(str(tokenizer_path(a.data)))
    base = f"{a.preset}_{a.data}_{a.tokens}"

    modes = [True, False] if a.both_cache else [not a.no_cache]
    print("=" * 92)
    print("  P030 추론 속도 벤치 (단계1: KV 캐시 + eos + 삼진 1회계산 반영)")
    print(f"  max_new={a.max_new}  reps={a.reps}(중위값)  프롬프트={PROMPT!r}")
    print(f"  캐시 모드={['on' if m else 'off' for m in modes]}   (off = 결과 014 조건)")
    print("=" * 92)
    print("  ★MB 는 하드코딩이 아니라 로드한 모델에서 계산한다(회계 = 안 B, 2026-07-31 통일).")
    print("    저장 = packed_mb(아직 없는 패킹 포맷의 이론값) / 상주 = runtime_mb(결과 016 실측식).")
    print("=" * 92)
    print(f"\n{'device':>8} {'threads':>8} {'model':>16} {'저장MB':>7} {'상주MB':>8} {'캐시':>5} "
          f"{'tok/s':>9} {'TTFT ms':>9}")
    print("-" * 92)

    rows = []
    for dev in a.device:
        if dev == "cuda" and not torch.cuda.is_available():
            print("  (cuda 없음 — 건너뜀)")
            continue
        thread_list = a.threads if dev == "cpu" else [0]
        for nt in thread_list:
            if dev == "cpu" and nt > 0:
                torch.set_num_threads(nt)
            for tag, arch in models:
                ck = paths.resolve_ckpt(a.preset, a.data, a.tokens, tag)
                if not ck.exists():
                    print(f"{dev:>8} {nt:>8} {tag:>16}  체크포인트 없음 — 건너뜀")
                    continue
                try:
                    model, cfg, _ = load_model(arch=arch, ckpt_path=str(ck), device=dev,
                                               drop_latent=a.drop_latent,
                                               int8_store=a.int8_store,
                                               unpack_cache=a.unpack_cache)
                except Exception as e:
                    print(f"{dev:>8} {nt:>8} {tag:>16}  로드 실패: {type(e).__name__}: {e}")
                    continue
                # ★P030 단계4: 층 수만 바꾼다(파라미터·메모리 고정). 1.0 이면 종전 경로.
                if a.infer_repeat != 1.0:
                    cfg.infer_repeat = a.infer_repeat
                    cfg.repeat_where = a.repeat_where
                    _n = len(model.visit_schedule())
                    print(f"  [P030-4] {tag}: 층 통과 {_n}회(기준 {cfg.n_layers}회) "
                          f"— 파라미터·메모리는 그대로다")
                # ★단일 소스: 표에 찍는 MB 는 여기서 계산된다(리터럴 금지).
                _mr = model.mem_report_all()
                mb, rt = _mr["packed_mb"], _mr["runtime_mb"]
                if a.check_cache:
                    from tinylm.infer.generate import check_cache_equivalence
                    if not check_cache_equivalence(model, cfg, tok, PROMPT, 16, dev):
                        print("  [중단] 캐시 출력이 불일치한다 — 속도를 재는 의미가 없다.")
                        return 2
                for use_c in modes:
                    try:
                        r, t = bench_one(model, cfg, tok, PROMPT, a.max_new, dev, a.reps, use_c)
                    except Exception as e:
                        print(f"{dev:>8} {nt:>8} {tag:>16}  실패: {type(e).__name__}: {e}")
                        continue
                    print(f"{dev:>8} {nt if nt else '기본':>8} {tag:>16} {mb:>7.1f} {rt:>8.1f} "
                          f"{'on' if use_c else 'off':>5} {r:>9.2f} {t:>9.1f}")
                    rows.append((dev, nt, tag, mb, r, t, use_c, rt))
                del model
                if dev == "cuda":
                    torch.cuda.empty_cache()

    print("-" * 92)
    if rows:
        print("\n★핵심 질문: 메모리를 줄이면 CPU 추론이 빨라지는가?")
        print("  ★결과 016 의 예측: 속도가 메모리를 따라간다면 **저장이 아니라 상주**를 따라간다.")
        print("    저장 순위(mA<mC<p6d)와 상주 순위(mC<mA<p6d)가 역전돼 있으므로 둘을 갈라 읽는다.")
        for dev in sorted({r[0] for r in rows}):
            for nt in sorted({r[1] for r in rows if r[0] == dev}):
                sub = [r for r in rows if r[0] == dev and r[1] == nt and r[3] > 0 and r[6]]
                if len(sub) < 2:
                    continue
                sub.sort(key=lambda r: -r[3])      # 저장 큰 것부터 = 기준선이 dense
                base_rate, base_pk, base_rt = sub[0][4], sub[0][3], sub[0][7]
                print(f"\n  [{dev} t{nt if nt else '기본'}] 기준 = {sub[0][2]}"
                      f"(저장 {base_pk:.1f}MB / 상주 {base_rt:.1f}MB) {base_rate:.2f} tok/s")
                print(f"    {'모델':16} {'저장비':>7} {'상주비':>7} {'속도비':>7}   판정")
                for _, _nt, tag, mb, r, _t, _c, rt in sub[1:]:
                    sp, st, rr = base_pk / mb, base_rt / rt, r / base_rate
                    # 어느 축을 따라가는지 기계적으로 판정한다(눈대중 금지)
                    d_pk, d_rt, d_1 = abs(rr - sp), abs(rr - st), abs(rr - 1.0)
                    verdict = ("전이 없음(속도비~1.0)" if d_1 <= min(d_pk, d_rt)
                               else "상주를 따라감" if d_rt < d_pk else "저장을 따라감")
                    print(f"    {tag:16} {sp:6.2f}x {st:6.2f}x {rr:6.2f}x   {verdict}")
        print("\n  읽는 법: 속도비가 1.0 근처면 **전이 없음** — 가중치가 GEMM 직전에 fp32 로")
        print("           dequant 되기 때문이다. 그것이 P034 단계2~4 가 존재하는 이유이고,")
        print("           음성 결과이지 실패가 아니다.")
        if a.both_cache:
            print("\n★캐시 이득(같은 모델, on/off):")
            for dev, nt, tag, mb, r, _t, c, _rt in rows:
                if not c:
                    on = [q for q in rows if q[:4] == (dev, nt, tag, mb) and q[6]]
                    if on:
                        print(f"    [{dev} t{nt}] {tag:16} off {r:6.2f} -> on {on[0][4]:6.2f} "
                              f"tok/s = {on[0][4]/r:5.2f}x")
    print("\n★한계: 삼진은 dequant 후 GEMM(저장≠실행 크기 — 상주는 같은 표의 상주MB 열) /")
    print("        Windows CPU 측정은 노이즈 큼(중위값 사용, 1~2% 차이는 읽지 않는다) /")
    print("        batch 1 단일요청. 서버 처리량은 다른 주제다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
