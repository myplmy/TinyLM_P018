#!/usr/bin/env python3
"""P029 정성 프로브 — 프롬프트 5개로 생성 결과를 눈으로 확인한다.

★왜 배치파일이 아니라 파이썬인가
  .bat 는 **순수 ASCII** 규약이다(한글이 들어가면 cmd 코드페이지에서 깨지고, `chcp` 로 덮으면
  파이썬 출력이 또 깨진다). 한국어 프롬프트는 필수인데 배치에 넣을 수 없으므로 **프롬프트를
  이 파일(UTF-8 파이썬 소스)에 두고 배치는 이 스크립트만 호출**한다.
  덤으로 모델을 **모델당 1회만 로드**한다(배치로 매 프롬프트마다 호출하면 15번 로드했다).

★기대치 (계획서 P029 참조 — 이거 안 읽고 결과 보면 오해한다)
  132M 삼진 파라미터 / 300M 토큰 = 파라미터당 **2.3 토큰**(Chinchilla 권장 ~20의 1/9,
  GPT-2 대비 1/130). SFT 도 없다. **사실·논리·지시따르기를 기대하면 안 된다.**
  볼 것은 ① 조사·어미 타당성 ② 문자체계 일관성 ③ 문체 관성 ④ 국소 문법 ⑤ 반복 붕괴 없음.

사용법
  python scripts/probe_prompts.py                          # 기본: mA + p6d 비교
  python scripts/probe_prompts.py --models mA_g4s34_k4     # 한 모델만
  python scripts/probe_prompts.py --temps 0.7              # 온도 하나만
  python scripts/probe_prompts.py --list                   # 프롬프트만 출력(모델 불필요)
  python scripts/probe_prompts.py --check-cache            # P030 단계1 캐시 정확성 게이트
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# (프롬프트, 무엇을 보는가, max_new) — 전부 **학습 분포 안**(위키/교육 웹텍스트 문체)에서 골랐다.
#   질문이 아니라 "이어쓰기"로 설계했다. 사전학습 모델의 실제 능력이 그것이기 때문.
PROMPTS = [
    ("대한민국의 수도 서울은",
     "한국어 위키 문체 + 최빈 도메인. 가장 쉬운 조건 — 여기서 무너지면 근본 문제", 100),
    ("조선 시대의 과학 기술은 세종대왕 때",
     "긴 한국어 문맥 유지. 조사·어미 연쇄와 시대/주제 관성", 100),
    ("물이 끓는 온도는 섭씨",
     "국소 사실 + 숫자 문맥. 100 이 나오면 좋고, 아니어도 '숫자 자리에 숫자'가 오는지가 관전 포인트", 60),
    ("The Industrial Revolution began in",
     "영문 전환. 코퍼스 절반이 영문인데 문자체계를 섞지 않는지", 100),
    ("인공지능 기술의 발전은 사회에",
     "추상 개념 + 열린 문장. 반복 붕괴가 가장 잘 드러나는 조건", 100),
]

# (태그, arch, 설명)
# ★MB·배수를 설명문에 적지 않는다(2026-07-31). 구 규약 값(11.7/2.64x/30.9)이 회계 통일
#   이후에도 문자열로 살아남아 로그마다 다른 숫자가 찍혔다. 크기는 report()·mem_report_all()
#   이 단일 소스로 찍고, 여기서는 **무엇을 보려는 모델인지**만 말한다.
DEFAULT_MODELS = [
    ("mA_g4s34_k4", "tied", "후보 A: g4 + 3:4 희소 + KD k4 — 결과 012 에서 dense 와 구분 불가"),
    ("p6d", "dense", "기준선: dense — 압축의 정성적 대가를 보기 위한 대조군"),
]


def banner(s, ch="="):
    print("\n" + ch * 78)
    print(f"  {s}")
    print(ch * 78)


def main():
    ap = argparse.ArgumentParser(description="P029 정성 프로브")
    ap.add_argument("--models", nargs="*", help="태그 목록(기본 mA_g4s34_k4 p6d)")
    ap.add_argument("--temps", nargs="*", type=float, default=[0.7, 1.0])
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--data", default="ko-en")
    ap.add_argument("--tokens", default="300M")
    ap.add_argument("--preset", default="m100")
    ap.add_argument("--list", action="store_true", help="프롬프트만 출력하고 종료")
    ap.add_argument("--check-cache", action="store_true",
                    help="P030 단계1 게이트: 캐시 유/무 그리디 출력 일치 검증만 하고 종료. "
                         "불일치면 exit 1 (한글 프롬프트가 필요해 .bat 대신 여기 산다)")
    ap.add_argument("--no-cache", action="store_true", help="KV 캐시 off(옛 경로 대조)")
    ap.add_argument("--device", default=None,
                    help="cpu / cuda. 미지정이면 GPU 가 보일 때 cuda. "
                         "★cpu 는 sample() 의 autocast 가 꺼져 **fp32** 로 돈다 — "
                         "P030 단계1.5-A 가 bf16 정밀도와 실제 캐시 버그를 가르는 수단이 이것이다")
    a = ap.parse_args()

    if a.list:
        banner("P029 프롬프트 5개")
        for i, (p, why, mx) in enumerate(PROMPTS, 1):
            print(f"\n[{i}] {p!r}   (max_new={mx})\n    확인: {why}")
        return 0

    models = ([(t, "dense" if t.startswith(("p6d", "dense", "p12d")) else "tied", "")
               for t in a.models] if a.models else DEFAULT_MODELS)

    banner("P029 정성 프로브 — 기대치를 먼저 읽을 것 (계획서 P029)", "#")
    print("  132M 삼진 / 300M 토큰 = 파라미터당 2.3 토큰 (Chinchilla 권장의 1/9, GPT-2 의 1/130).")
    print("  SFT 없음 → 질문에 답하거나 지시를 따르지 못한다. 사실·연도·수치는 거의 틀린다.")
    print("  볼 것: 조사/어미 타당성 · 문자체계 일관성 · 문체 관성 · 국소 문법 · 반복 붕괴 없음.")

    # ★샘플링 루프는 절대 여기서 다시 구현하지 않는다 — `tinylm.infer.generate.sample()` 이 단일 소스.
    #   (최초판이 루프를 복제했다가 `cfg.seq_len`(존재하지 않는 필드)로 크래시했다.
    #    실제 필드명은 `max_seq_len` 이고, 복제본에는 autocast(bf16)도 빠져 있었다.)
    from tinylm import paths
    from tinylm.infer.generate import load_model, sample
    from tokenizers import Tokenizer
    from tinylm.data import tokenizer_path

    tok = Tokenizer.from_file(str(tokenizer_path(a.data)))
    base = f"{a.preset}_{a.data}_{a.tokens}"

    if a.check_cache:
        from tinylm.infer.generate import check_cache_equivalence
        banner("P030 단계1 게이트 — KV 캐시 정확성 (그리디: 캐시 유/무가 완전히 같아야 한다)")
        print("  ★이 게이트는 '로짓 동등성'이 아니라 '그리디 문자열 일치'를 본다(결과 014 §8).")
        print("    bf16(cuda) + 자기회귀 24스텝이면 near-tie 한 번 뒤집힘이 이후 전부를 바꾼다.")
        print("    --device cpu 로 돌리면 autocast 가 꺼져 fp32 가 되고, 정밀도 요인이 사라진다.")
        print("    RoPE 슬라이스·SDPA 마스크는 dense 와 공유 코드이고 p6d 가 통과했으므로 이미 무죄다.")
        print("    확정 판정은 teacher-forced 로짓 비교(scripts/diag_kvcache.py)가 한다.")
        allok = True
        for tag, arch, _d in models:
            ck = paths.RUNS / "ckpt" / f"{base}_{tag}.pt"
            if not ck.exists():
                print(f"  [건너뜀] 체크포인트 없음: {tag}")
                continue
            model, cfg, device = load_model(arch=arch, ckpt_path=str(ck), device=a.device)
            print(f"\n  --- {tag} ({arch})  device={device} "
                  f"{'fp32(autocast off)' if str(device) != 'cuda' else 'bf16 autocast'} ---")
            for prompt, _why, _mx in PROMPTS:
                allok &= check_cache_equivalence(model, cfg, tok, prompt,
                                                 max_new=24, device=device)
            del model
        _dev = a.device or "cuda(자동)"
        print("\n  " + ("전부 일치 ✅"
                        + (" (fp32) — 정밀도 요인 배제. 단 이것은 '증거'지 '증명'이 아니다. "
                           "확정은 scripts/diag_kvcache.py"
                           if str(a.device) == "cpu" else
                           " (bf16) — 이 조건에서 통과했으면 정밀도 여유가 충분한 것이다.")
                        if allok else
                        f"불일치 ❌ (device={_dev}) — "
                        + ("fp32 에서도 다르다면 **실제 버그**다. dense 가 통과했으므로 "
                           "RoPE·마스크가 아니라 CLA owner 키 캐시 경로를 본다."
                           if str(a.device) == "cpu" else
                           "bf16 이므로 **아직 버그로 단정할 수 없다**. --device cpu 로 다시 돌려라.")))
        return 0 if allok else 1

    for tag, arch, desc in models:
        ck = paths.RUNS / "ckpt" / f"{base}_{tag}.pt"
        banner(f"MODEL {tag}  ({arch})", "#")
        if desc:
            print(f"  {desc}")
        if not ck.exists():
            print(f"  [!] 체크포인트 없음: {ck}")
            print("      아직 학습이 안 끝났거나 태그가 다릅니다. 건너뜁니다.")
            continue
        try:
            model, cfg, device = load_model(arch=arch, ckpt_path=str(ck), device=a.device)
        except Exception as e:
            print(f"  [!] 로드 실패: {e}")
            continue
        npar = sum(p.numel() for p in model.parameters())
        print(f"  로드 완료 — {npar/1e6:.1f}M 파라미터, device={device}, anneal=1.0(완전 삼진)")

        for i, (prompt, why, mx) in enumerate(PROMPTS, 1):
            print(f"\n  {'-'*74}")
            print(f"  [{i}] 프롬프트: {prompt}")
            print(f"      확인 포인트: {why}")
            for T in a.temps:
                try:
                    out = sample(model, cfg, tok, prompt, max_new=mx,
                                 temperature=T, top_k=a.top_k, device=device,
                                 use_cache=not a.no_cache)
                except Exception as e:
                    print(f"\n      >>> temp={T}  [!] 생성 실패: {type(e).__name__}: {e}")
                    continue
                cont = out[len(prompt):] if out.startswith(prompt) else out
                print(f"\n      >>> temp={T}")
                print(f"      {prompt}", end="")
                print(cont.replace("\n", "\n      "))

    banner("체크리스트 — 눈으로 채운다. 점수화하지 않는다(표본 5개는 통계가 아니다)")
    for item in ["한글 프롬프트 → 한글 유지", "영문 프롬프트 → 영문 유지",
                 "조사·어미가 그럴듯함", "위키 문체 관성",
                 "무한 반복 없음(temp 0.7)", "문자 깨짐·제어문자 없음"]:
        print(f"  [ ] {item}")
    print("""
  읽는 법
    전부 비어 있다      → 모델이 아니라 파이프라인을 의심(토크나이저? 체크포인트?)
    채워지지만 사실 틀림 → **정상**. 파라미터당 2.3 토큰이면 이렇게 나온다
    mA 가 눈에 띄게 나쁨 → 압축의 정성적 대가를 처음 본 것. 기록할 값어치는 있으나
                          표본 5개는 '인상'이고 val_loss 는 둘이 같다고 말한다

  샘플링은 확률적이다 — 결론 내기 전에 여러 번 돌려 분산을 보라.
  이 프로브는 아키텍처를 고르지 않는다. 그건 REVIEW1 과 val_loss 가 한다.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
