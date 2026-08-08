"""TinyLM — 저사양 CPU·엣지용 초경량 LLM 아키텍처 (메모리 최적화).

패키지 import 시 paths 모듈이 먼저 로드되어 HF 캐시를 작업폴더로 리다이렉트한다
(datasets/transformers import 전에 환경변수를 설정해야 하므로 순서가 중요).
"""
from . import paths  # noqa: F401  (import 부작용으로 HF_HOME 등 설정)

__all__ = ["paths"]
