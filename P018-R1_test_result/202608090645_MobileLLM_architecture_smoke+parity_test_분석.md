# 202608090645 MobileLLM architecture smoke+parity test 분석

## 판정

이 로그는 **MobileLLM-125M shape와 tied g6 구조의 유효한 초기 smoke 기록**이지만, 당시 embedding/output projection 의미론이 잘못되어 dense reference parity 근거로는 폐기한다. 202608091012 weight parity 로그가 이를 대체한다.

## 로그가 실제로 확인한 것

- 당시 환경: PyTorch 2.10.0, CUDA 사용 가능, NVIDIA GeForce RTX 4070 Ti SUPER.
- 576d / 1536 FFN / 30L / 9Q / 3KV / head dim 64 / vocab 32000 구조가 생성되었다.
- tied g6에서 다섯 MLP가 다음처럼 등록되었다.
  - L0-L5
  - L6-L11
  - L12-L17
  - L18-L23
  - L24-L29
- attention은 dense와 tied 모두 30개 독립 객체였다.
- 양 모델에서 배치 2, 길이 32의 forward/backward가 완료되었고, tied MLP gradient가 존재했다.
- 기존 P018 모델 생성 회귀검사가 통과했다.

## 기록된 수치와 해석

| 항목 | Dense | Tied g6 |
|---|---:|---:|
| 로그 파라미터 | 143,067,456 | 76,712,256 |
| 고유 MLP | 30 | 5 |
| 무작위 초기화 loss | 10.501503 | 10.527371 |

파라미터 수가 현재 정본인 124,635,456 / 58,280,256보다 각각 18,432,000 많다. 이는 `(32000 × 576)` 크기의 output projection을 embedding과 별도로 보유했기 때문이다.

## 치명적 한계

로그는 `tie_word_embeddings=False`를 “embedding과 LM head가 별도”로 해석했다. 그러나 공식 MobileLLM 125M 실행 설정은 `share_embedding=True`이며, 최신 parity 로그는 입력 embedding Parameter를 출력 projection에 그대로 사용함을 확인했다. 따라서 이 로그의 다음 문장은 현재 기준으로 틀렸다.

- `input embedding and LM-head are separate parameters`
- `Difference: MLP sharing only`

후자는 embedding 오류를 dense와 tied 양쪽에 동일하게 넣었다는 제한된 의미에서만 성립하며, 공식 모델과의 parity를 뜻하지 않는다.

## 현재 사용할 수 있는 범위

- g6 그룹 배치와 attention 비공유의 역사적 smoke 근거
- forward/backward 및 공유 gradient 검사 항목의 설계 참고

trainer, tokenizer/cache, checkpoint round trip, KD, 실제 top-level 최신 코드, 메모리·속도 및 품질 근거로는 사용할 수 없다.

