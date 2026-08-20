# 202608091012 MobileLLM architecture weight parity test 분석

## 판정

이 로그는 현재까지 가장 강한 **dense MobileLLM-125M 1차 parity 증거**다. 공식 구현의 무작위 초기 가중치를 TinyLM에 명시적으로 복사했을 때, 선택된 eager CPU·FP32·길이 8 경로에서 architecture, weight, 모든 층 hidden state와 최종 logits가 정확히 일치했다.

단, 이것은 전체 R1 파이프라인이나 tied g6의 검증 완료를 뜻하지 않는다.

## 실행 조건

| 항목 | 값 |
|---|---|
| 당시 Python 환경 | `p018` |
| PyTorch | 2.13.0+cpu |
| Transformers | 4.41.2 |
| CUDA | False |
| device | CPU |
| 입력 | batch 2, sequence 8 |
| 공식 소스 | `MobileLLM/utils/modeling_llama.py` |

현재 2026-08-21의 `p018` 환경은 CUDA build로 변경되었으므로, 이 로그의 환경은 과거 실행 조건으로만 읽어야 한다.

## 확인된 결과

### 구조와 파라미터

- 576 hidden, 1536 FFN, 30 layers, 9 Q heads, 3 KV heads, vocab 32000.
- context 2048, RoPE theta 10000, RMSNorm epsilon 1e-5, initializer std 0.02.
- 공식과 TinyLM 모두 124,635,456 고유 파라미터.
- 양쪽 모두 272개 canonical parameter tensor가 존재하고 shape가 일치.
- `share_embedding=True`: TinyLM `lm_head.weight`와 `emb.weight`가 동일 Parameter/storage.
- dense MLP 30개와 attention 30개가 모두 독립.

### weight와 수치

- 공식에서 TinyLM으로 272개 tensor를 복사.
- 모든 tensor의 global max absolute/relative difference가 0.
- embedding, layer 0-29 hidden state, final hidden state, final logits가 모두 `max_abs=0`, `max_rel=0`.
- `F.linear(x, emb.weight)`와 `lm_head(x)`가 정확히 일치.

따라서 로그 범위 안에서는 `ARCHITECTURE PARITY`, `WEIGHT PARITY`, `NUMERICAL FORWARD`, `LOGITS PARITY`가 모두 PASS다.

## 증명하지 않은 것

- `parity_reference/mobilellm_125m_state_dict.pt`의 provenance 또는 로드 parity. 현재 스크립트는 해당 파일을 로드하지 않고 같은 프로세스의 공식 모델에서 직접 복사한다.
- tied g6의 5개 MLP registration, gradient 합산, unique parameter 수 58,280,256.
- TinyLM top-level 경로 전체의 일반성. parity 스크립트는 중간값 비교를 위해 TinyLM layer loop를 별도로 전개한다.
- backward/loss/optimizer, gradient checkpointing, save/load/resume.
- 공식 32k tokenizer와 `ko-en` cache, trainer/KD/eval/generate 통합.
- BF16/CUDA, 다른 sequence length와 seed, 메모리·throughput·latency.
- teacher 또는 student 품질과 R1 가설.

## 다음 gate

1. 사용자가 최신 CUDA 환경에서 정리된 수동 검증 스크립트를 실행해 dense/tied g6 top-level forward/backward와 구조를 확인한다.
2. 아주 작은 별도 cache·짧은 trainer smoke로 save/load, online KD, dense/tied tag 분리를 확인한다.
3. 위 gate 통과 후에만 300M teacher/student 본 실험을 시작한다.

