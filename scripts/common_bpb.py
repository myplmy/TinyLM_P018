#!/usr/bin/env python3
"""P028 단계1 — **공통 원문 bpb**. 데이터셋 간 비교를 처음으로 성립시킨다.

★왜 필요한가 (결과 009 §판정, 결과 011)
   `ko-en` 과 `ko-edu-en` 은 **토크나이저도 val 셋도 다르다.** 그래서 두 모델의 val_loss 나
   bpb 를 나란히 놓는 것은 **무효**다. 결과 009 가 "관측은 열세지만 판정 불가" 로 끝난 이유가
   이것이고, 그 뒤로 데이터 계열 실험 전체가 여기 막혀 있었다.

★해법 — **어느 학습 코퍼스에도 없는 공통 원문**을 두 모델에 똑같이 통과시킨다
   bpb = loss / ln2 / bytes_per_token 이고, **같은 원문·같은 바이트 수**라면
   토크나이저가 달라도 분모가 같아진다. 즉 **bpb 는 토크나이저 무관**이 된다.

   원문 = **SQuAD v2 의 `context`** (`datasets/squad/train-v2.0.json`, 로컬 보유).
   고유 context 19,029개 / 14.0MB / 평균 736자. 영문 위키 산문이고 우리 `ko-en`·`ko-edu-en`
   어디에도 들어 있지 않다. 근거·대안 검토는 `docs/methods/07_corpus_selection.md` §3.

★★반드시 함께 읽을 한계
   1. **영문 전용이다.** 한국어 공통 원문은 여전히 없다. 이 스크립트가 재는 것은
      "영문 산문에 대한 압축률" 이지 모델 전체 품질이 아니다.
   2. 모델마다 **토크나이저가 다르므로 토큰 수가 다르다.** 그래서 손실을 그대로 비교하면
      안 되고 **반드시 bpb 로 환산**해야 한다. 이 스크립트가 그걸 강제한다.
   3. SQuAD context 는 **영문 위키**다. `ko-en` 은 영문 fineweb-edu 를 절반 썼고
      `ko-edu-en` 도 그렇다 — **도메인이 완전히 낯설지는 않다.** 그건 공정하다(둘 다 같은 조건).

사용법
  python scripts/common_bpb.py --models p6d t_kd_g8
  python scripts/common_bpb.py --models p6d --data ko-en --tokens 300M
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SQUAD = ROOT / "datasets" / "squad" / "train-v2.0.json"


def load_squad_contexts(path=SQUAD, limit=None):
    """고유 `context` 만 뽑아 순서를 고정해 돌려준다(문서 중복 제거).

    ★순서를 고정하는 이유: 이 텍스트가 **모든 모델에 동일**해야 비교가 성립한다.
    set 을 그대로 쓰면 파이썬 해시 시드에 따라 순서가 달라질 수 있으므로 정렬한다.
    """
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    seen, out = set(), []
    for art in d["data"]:
        for para in art["paragraphs"]:
            c = para["context"]
            if c not in seen:
                seen.add(c)
                out.append(c)
    out.sort()                       # ★결정성
    return out[:limit] if limit else out


def main():
    ap = argparse.ArgumentParser(description="P028 단계1 공통 원문 bpb")
    ap.add_argument("--models", nargs="+", required=True,
                    help="태그 목록. 서로 **다른 데이터셋으로 학습된 모델**을 넣는 것이 요점이다")
    ap.add_argument("--data", nargs="+", default=None,
                    help="모델별 데이터셋(토크나이저 선택용). 미지정이면 --data-default 를 전부 적용")
    ap.add_argument("--data-default", default="ko-en")
    ap.add_argument("--tokens", default="300M")
    ap.add_argument("--preset", default="m100")
    ap.add_argument("--arch", nargs="+", default=None, help="모델별 dense/tied. 미지정이면 태그로 추정")
    ap.add_argument("--max-docs", type=int, default=4000,
                    help="쓸 context 개수. 기본 4000 ≈ 3MB ≈ val 의 2배. 0 이면 전부")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from tokenizers import Tokenizer
    from tinylm import paths
    from tinylm.data import tokenizer_path
    from tinylm.infer.generate import load_model

    if not SQUAD.exists():
        print(f"[!] SQuAD 원문이 없다: {SQUAD}")
        return 2

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    docs = load_squad_contexts(limit=(a.max_docs or None))
    text = "\n\n".join(docs)
    n_bytes = len(text.encode("utf-8"))

    print("#" * 92)
    print("  P028 단계1 — 공통 원문 bpb (SQuAD v2 context)")
    print("#" * 92)
    print(f"  device={dev}  문서 {len(docs):,}개  **{n_bytes/1e6:.2f} MB**  seq={a.seq}")
    print("  ★같은 원문·같은 바이트에 서로 다른 토크나이저의 모델을 통과시킨다.")
    print("    bpb = loss / ln2 / (bytes/token) 이므로 **토크나이저가 달라도 비교가 성립**한다.")
    print("  ⚠️ 영문 전용이다. 한국어 공통 원문은 아직 없다(07_corpus_selection §3).")

    datas = a.data if a.data else [a.data_default] * len(a.models)
    if len(datas) == 1 and len(a.models) > 1:
        datas = datas * len(a.models)
    assert len(datas) == len(a.models), "--data 개수가 --models 와 다르다"
    archs = a.arch if a.arch else [
        "dense" if t.startswith(("p6d", "dense", "p12d")) else "tied" for t in a.models]

    rows = []
    for tag, data, arch in zip(a.models, datas, archs):
        ck = paths.resolve_ckpt(a.preset, data, a.tokens, tag)
        if not ck.exists():
            print(f"\n  [건너뜀] 체크포인트 없음: {ck.name}")
            continue
        tok = Tokenizer.from_file(str(tokenizer_path(data)))
        ids = tok.encode(text).ids
        n_tok = len(ids)
        bpt = n_bytes / max(n_tok, 1)            # ★이 모델 토크나이저의 bytes/token

        model, cfg, _ = load_model(arch=arch, ckpt_path=str(ck), device=dev)
        was = cfg.quant_anneal
        model.set_anneal(1.0); model.eval(); model.freeze_quant()

        arr = np.array(ids, dtype=np.int64)
        n_crop = (len(arr) - 1) // a.seq
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, n_crop, a.micro_bs):
                chunk = [arr[j*a.seq:(j+1)*a.seq + 1] for j in range(i, min(i + a.micro_bs, n_crop))]
                t = torch.from_numpy(np.stack(chunk)).to(dev)
                x, y = t[:, :-1], t[:, 1:]
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
                    logits = model(x)
                l = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
                tot += float(l) * y.numel(); cnt += y.numel()
        model.clear_quant(); model.train(); model.set_anneal(was)
        loss = tot / cnt
        bpb = loss / math.log(2) / bpt
        rows.append((tag, data, n_tok, bpt, loss, bpb))
        print(f"\n  ── {tag} ({data}, {arch})")
        print(f"     토큰 {n_tok:,}  bytes/token {bpt:.4f}  손실 {loss:.4f}  **bpb {bpb:.4f}**")
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    if len(rows) < 2:
        print("\n  비교할 모델이 2개 미만이다. 데이터셋이 다른 모델을 최소 둘 넣어야 의미가 있다.")
        return 0 if rows else 2

    print("\n" + "=" * 92)
    print("  ★공통 원문 bpb — 이것이 데이터셋 간 비교로 유효한 유일한 수치다")
    print("=" * 92)
    print(f"  {'모델':<18}{'데이터셋':<14}{'토큰수':>10}{'B/tok':>8}{'손실':>9}{'bpb':>9}")
    print("  " + "-" * 72)
    for tag, data, n_tok, bpt, loss, bpb in sorted(rows, key=lambda r: r[5]):
        print(f"  {tag:<18}{data:<14}{n_tok:>10,}{bpt:>8.3f}{loss:>9.4f}{bpb:>9.4f}")

    best = min(rows, key=lambda r: r[5])
    print(f"\n  최저 bpb = {best[0]} ({best[1]}) {best[5]:.4f}")
    print("""
  읽는 법
    · **bpb 만** 비교한다. '손실' 열은 토크나이저마다 토큰 수가 달라 **비교 불가**다.
      (일부러 함께 찍는다 — 둘이 다른 이야기를 한다는 것이 이 실험의 요점이다)
    · bpb 차이가 실무 분해능(0.024 nats 를 bpb 로 환산하면 대략 0.008)보다 작으면
      **서열을 매기지 않는다.**
    · 토큰 수가 크게 다르면 그 자체가 정보다 — 토크나이저가 이 원문을 얼마나 잘 압축하는가.

  한계
    · **영문 전용.** 한국어 쪽 결론은 이 실험으로 나오지 않는다.
    · SQuAD context 는 영문 위키다. 두 모델 다 영문을 절반 학습했으므로 조건은 공정하지만,
      **한국어 능력은 전혀 반영되지 않는다.**
    · 문서를 `\\n\\n` 로 이어붙여 하나의 스트림으로 만든다 — 문서 경계에서 컨텍스트가 섞인다.
      모든 모델에 동일하게 적용되므로 비교에는 영향이 없다.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
