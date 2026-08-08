"""작업폴더 기준 경로 + Hugging Face 캐시 리다이렉트.

이 모듈은 `datasets`/`transformers` 를 import 하기 전에 먼저 로드되어야 한다.
그래야 HF_HOME 등이 적용된 상태로 라이브러리가 캐시 위치를 잡는다.
tinylm.__init__ 이 이 모듈을 최상단에서 import 하므로, 패키지를 import 하는 한 보장된다.

경로는 환경변수로 덮어쓸 수 있다:
  TINYLM_DIR   프로젝트 루트 (기본: 이 패키지의 상위 폴더 = Z:\\TinyLM)
  HF_HOME 등   이미 설정돼 있으면 존중한다(setdefault).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("TINYLM_DIR", Path(__file__).resolve().parents[1]))

HF_DIR     = Path(os.environ.get("TINYLM_HF", ROOT / "HF"))  # 모델·데이터셋 로컬 캐시
DATA_CACHE = ROOT / "data_cache"   # 토큰화된 바이너리(크기별 재사용)
RUNS       = ROOT / "runs"         # 체크포인트·로그

# HF 캐시를 '강제로' 작업폴더 하위(HF/)로 돌린다.
# 사용자 환경의 외부 HF_HOME 이 다른 곳을 가리켜도 무시하고 여기로 받는다.
# (원래 위치를 쓰고 싶으면 실행 전 TINYLM_HF=<경로> 로 지정.)
os.environ["HF_HOME"] = str(HF_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_DIR / "hub")
os.environ["HF_DATASETS_CACHE"] = str(HF_DIR / "datasets")
os.environ.pop("TRANSFORMERS_CACHE", None)   # 옛 변수가 외부를 가리키면 제거
if os.name == "posix":   # expandable_segments 는 Linux 전용(Windows는 미지원 경고만 남김)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

for _d in (HF_DIR, HF_DIR / "hub", DATA_CACHE, RUNS, RUNS / "ckpt", RUNS / "logs"):
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 체크포인트 이름 해석 (2026-08-01 신설)
# ─────────────────────────────────────────────────────────────────────────────
def resolve_ckpt(preset, data, tokens, tag, must_exist=False):
    """`runs/ckpt/{preset}_{data}_{tokens}_{tag}.pt` 를 찾되, **프리셋을 넘어 탐색**한다.

    ★왜 필요한가 (P038·P036 단계2 실패, 2026-08-01)
      `m100R1a`/`m100R1c` 승격은 체크포인트 이름을 `{preset}_...` 로 갈라 **네임스페이스를
      분리**했다. 그건 의도한 설계다. 그런데 부모 dense·KD 교사·후처리 도구가 전부
      **현재 프리셋 이름으로만** 경로를 만들어서, `--preset m100R1a --init-from` 이
      존재하지 않는 `m100R1a_ko-en_300M_dense.pt` 를 찾다 즉사했다.
      dense 부모는 하나뿐이고 이름은 `m100_...` 다.

    탐색 순서
      1. 요청한 프리셋 그대로
      2. `PRESET_PARENT` 가 가리키는 부모 프리셋
      3. `*_{data}_{tokens}_{tag}.pt` 전역 검색 — 후보가 **정확히 하나**일 때만 채택
         (여러 개면 어느 것을 쓸지 사람이 정해야 한다. 조용히 고르지 않는다.)
    """
    from .config import PRESET_PARENT
    ck = RUNS / "ckpt"
    tried = []
    for pr in [preset, PRESET_PARENT.get(preset)]:
        if not pr:
            continue
        cand = ck / f"{pr}_{data}_{tokens}_{tag}.pt"
        tried.append(cand.name)
        if cand.exists():
            if pr != preset:
                print(f"[ckpt] {preset} 용 '{tag}' 가 없어 부모 프리셋 것을 쓴다 -> {cand.name}")
            return cand
    hits = sorted(ck.glob(f"*_{data}_{tokens}_{tag}.pt"))
    if len(hits) == 1:
        print(f"[ckpt] 프리셋이 달라 전역 검색으로 찾았다 -> {hits[0].name}")
        return hits[0]
    if len(hits) > 1:
        raise FileNotFoundError(
            f"'{tag}' 후보가 여러 개다 — 어느 것을 쓸지 명시해야 한다: "
            + ", ".join(h.name for h in hits))
    if must_exist:
        raise FileNotFoundError(
            f"체크포인트를 찾지 못했다: {tried} (전역 검색도 0건).\n"
            f"  --preset 을 확인하거나, 그 태그를 먼저 학습할 것.")
    return ck / f"{preset}_{data}_{tokens}_{tag}.pt"       # 쓰기용 기본 경로
