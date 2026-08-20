# 202608090538 Mobile architecture smoke test 분석

## 판정

이 로그는 **초기 프로토타입의 동작 확인 기록**이다. 현재 P018-R1의 MobileLLM-125M 구조 또는 성능 근거로 사용하면 안 된다.

## 로그가 실제로 확인한 것

- 당시 환경: PyTorch 2.10.0, CUDA 사용 가능, NVIDIA GeForce RTX 4070 Ti SUPER.
- 임시 `mobile100` 구조에서 dense와 MLP-sharing 모델이 생성되었다.
- 배치 2, 길이 32의 forward와 causal-LM backward가 완료되었다.
- 공유 모델에서 한 MLP의 gradient가 존재했고, attention 객체는 층별로 독립이었다.
- 같은 프로세스에서 기존 P018 모델도 생성되었다.

## 기록된 수치

| 항목 | Dense | Shared |
|---|---:|---:|
| 구조 | 768d / 2048 FFN / 20L / vocab 32768 | 동일, MLP g4 |
| 고유 MLP | 20 | 5 |
| 파라미터 | 149,060,352 | 78,281,472 |
| 무작위 초기화 loss | 10.569433 | 10.535809 |

두 loss는 서로 다른 무작위 모델의 smoke 값이므로 품질 비교에 사용할 수 없다.

## 현재 R1과 맞지 않는 점

- 공식 MobileLLM-125M은 576d / 1536 FFN / 30L / 9Q / 3KV / vocab 32000이다.
- 현재 R1 기본 tied 조건은 30층을 6층씩 묶는 g6이다.
- 공식 입력/출력 embedding 공유 의미론과 weight parity를 검사하지 않았다.
- tokenizer, data cache, checkpoint, trainer, KD, 평가, 메모리·속도 계측을 검사하지 않았다.

## 보존 가치와 후속 조치

보존할 가치는 “dense/tied 생성, attention 비공유, 공유 MLP gradient, parameter registration, 기존 P018 회귀”라는 검사 항목이다. 이 항목은 현재 구조에 맞춰 `scripts/validation/validate_mobile_architecture.py`로 이전되었다. 원 로그는 역사적 provenance로 그대로 보존한다.

