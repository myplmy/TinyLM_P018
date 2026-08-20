# P018-R1 MobileLLM-g6 교사 비교 및 KD 정적 통합 설계

- 상태: 정적 통합 완료, 사용자 실행 gate 대기
- 기준일: 2026-08-21
- 주 개입: MobileLLM-125M의 MLP-only g6 parameter sharing

## 1. 연구 질문

MobileLLM-125M dense teacher와 같은 base architecture의 MLP-only tied g6 teacher를 같은 데이터·토큰·optimizer 조건에서 각각 처음부터 학습할 때:

1. teacher quality 차이는 얼마인가.
2. 고유 parameter와 실제 teacher/KD memory는 얼마나 줄어드는가.
3. 동일 student에 전달했을 때 student quality 차이는 얼마인가.

고정 동등성 한계는 사용하지 않는다. 차이의 방향·크기·불확실성을 그대로 보고한다.

## 2. Arms

### Stage A - Teacher comparison

| 코드명 | architecture | 초기화 | teacher |
|---|---|---|---|
| `DT` | MobileLLM-125M dense | seed별 random | 없음 |
| `TT_g6` | MobileLLM-125M, MLP-only g6 | seed별 random | 없음 |
| `DT_to_TT_g6` | optional: dense checkpoint를 g6로 이식 | Dense Teacher | 없음 |

`DT_to_TT_g6`는 주 실험이 아니다. 독립학습 비교가 끝난 뒤 별도 가설과 tag로만 수행한다.

### Stage B - Transfer comparison

현재 권장 student는 `mobile125 tied g6`로 고정한다.

| 코드명 | student | teacher | 초기화 |
|---|---|---|---|
| `S_noKD` | Mobile tied g6 | 없음 | seed별 동일 student init |
| `S_DKD` | Mobile tied g6 | paired Dense Teacher | `S_noKD`와 동일 init |
| `S_TKD` | Mobile tied g6 | paired Tied Teacher | `S_noKD`와 동일 init |

`TT_g6`와 `S_noKD`의 architecture·data·schedule·seed가 완전히 같다면 같은 run은 두 역할을 동시에 수행한다. 별도의 작은 student를 선택하면 이 재사용 규칙은 폐기한다.

## 3. 고정 구조

| 항목 | 값 |
|---|---|
| vocab/tokenizer | 공식 MobileLLM SentencePiece 32000 |
| hidden / FFN | 576 / 1536 |
| layers | 30 |
| Q / KV heads | 9 / 3 |
| activation | SwiGLU/SiLU |
| norm | RMSNorm eps 1e-5 |
| position | RoPE theta 10000, max 2048 |
| embedding/output | 같은 Parameter 공유 |
| attention | 30층 모두 독립 |
| dense MLP | 30개 독립 |
| tied MLP | g6, 5개 고유: 0-5/6-11/12-17/18-23/24-29 |
| quantization | 없음, full precision weights |

Dense와 tied의 optimization LR을 같게 유지한다. 기존 P018 tied optimizer의 `lr/sqrt(g)`는 적용하지 않는다.

## 4. 데이터와 토큰 규약

- dataset key: `ko-en`.
- source 설정: Wikipedia `20231101.ko` 0.5 + FineWeb-Edu `sample-10BT` 0.5.
- tokenizer cache: `ko-en__mobilellm_32k_300000000`.
- encode: `add_special_tokens=False`, 문서 끝에 EOS `2` 한 번.
- cache: exact 300M 우선.
- training consumed tokens: 모든 arm에서 동일한 정확한 수. 실패/OOM run의 소비량을 사후 맞추지 않는다.
- actual corpus composition: `mix_token_frac`, `mix_exhausted`를 보고한다. 설정 weight를 실제 token fraction이라고 쓰지 않는다.

300M은 R1 controlled budget이다. 원 MobileLLM 논문의 1T-token 학습 재현이 아니다.

## 5. Paired seed 설계

제안 seed: `1337, 2024, 3407`.

각 seed `s`에 대해:

- `DT_s`와 `TT_g6_s`는 같은 cache와 data-order seed를 사용한다.
- `S_noKD_s`, `S_DKD_s`, `S_TKD_s`는 같은 student initial state와 data-order seed를 사용한다.
- KD teacher는 같은 seed의 teacher를 연결한다.
- validation artifact와 crop 순서는 모든 arm/seed에서 동일하다.
- run tag, stdout/stderr, JSON, checkpoint는 arm/seed별로 격리한다.

## 6. 통계 보고

Teacher primary contrast:

- `TT_g6 - DT`의 seed별 paired NLL/bpb/perplexity 차이.

Student primary contrasts:

- `S_TKD - S_DKD`.
- `S_DKD - S_noKD`.
- `S_TKD - S_noKD`.

각 contrast에 raw values, mean, standard deviation, 95% confidence interval, standardized effect size를 보고한다. seed가 3개뿐이므로 CI가 넓을 수 있음을 명시하고, p-value 또는 checkpoint-level crop 수를 seed 수처럼 취급하지 않는다. `paired_eval --raw-differences`의 per-crop CI는 해당 checkpoint 쌍의 평가 변동이며 architecture-level seed uncertainty와 별도다.

“동등”, “동일”, “품질 보존”은 고정 margin이 없으므로 결론 용어로 사용하지 않는다. 대신 예를 들어 “평균 NLL 차이 +0.018, 95% CI [...]”처럼 쓴다.

## 7. Memory 및 runtime endpoint

다음을 각각 분리한다.

- logical parameter count.
- unique parameter count.
- checkpoint file bytes 및 dtype.
- teacher-only inference peak allocated/reserved/process memory.
- end-to-end KD peak allocated/reserved/process memory.
- teacher forward latency/throughput.
- KD total step latency 및 wall time.
- OOM/spill 여부.

측정 전 warm-up, `torch.cuda.synchronize()`, peak reset 시점, sequence/micro-batch/precision, compile, 다른 GPU process 상태를 고정한다. parameter sharing은 FLOPs를 줄이지 않으므로 speedup을 사전 가정하지 않는다.

## 8. 실행 gate

### G0 - 정적 검사

- Python AST/compile 통과.
- `mobile125 dense/tied` config가 30L, g1/g6, 공식 tokenizer를 가리킴.
- 기존 P018 config snapshot 불변.
- import에서 model/checkpoint를 생성하지 않음.

### G1 - 사용자 구조 smoke

- dense unique parameter 124,635,456.
- tied g6 unique parameter 58,280,256.
- embedding/output 동일 Parameter.
- attention unique 30, tied MLP unique 5.
- tiny forward/backward 및 tied shared-gradient finite.

### G2 - 사용자 mini data/trainer smoke

- Mobile tokenizer 전용 작은 cache 생성.
- dense와 tied 각각 매우 짧은 학습, eval, checkpoint save/load.
- 기존 `tiny_bpe` cache가 재사용/변경되지 않음.
- JSON에 tokenizer hash, unique/logical parameter, allocated/reserved가 기록됨.

### G3 - 사용자 mini KD smoke

- 동일 vocab/tokenizer teacher/student 확인.
- Dense Teacher -> tied student와 Tied Teacher -> tied student가 각각 짧게 완주.
- CE/KL finite, teacher gradient 없음, tag/output 분리.

### G4 - 300M cache audit

- 정확히 300M 또는 source 고갈 시 실제 token 수 확인.
- tokenizer hash 일치.
- actual source token fraction 및 exhaustion 확인.
- val 구성 고정 및 모든 arm 재사용.

### G5 - 본 run

G0-G4 통과 후 Stage A의 3 seed를 먼저 완료·평가한다. Stage B는 teacher checkpoint와 student protocol이 잠긴 뒤 시작한다. Optional compression/g10/g15는 Stage A/B 1차 결과 이후에만 연다.

## 9. 실패 규칙

- cache/tokenizer/vocab/hash가 다르면 run 시작 금지.
- NaN/Inf, checkpoint load mismatch, unexpected parameter count, wrong group map은 결과가 아니라 구현 실패다.
- OOM 뒤 micro-batch를 한 arm에만 바꾸지 않는다. 모든 비교 arm의 조건을 다시 잠근다.
- 조기 실패 run과 완주 run의 consumed tokens를 사후 equalize하지 않는다.
- 과거 P018 `mC_wsd` 결과는 동기와 설계 참고이며 R1 evidence가 아니다.

## 10. 원 논문과의 관계

공식 architecture/tokenizer에는 맞추지만 pretraining corpus는 논문/공식 저장소에서 재현할 수 없었다. 원 논문의 2e-3 cosine, 1T tokens는 참고 조건이고, 프로젝트 300M `ko-en` WSD primary와 섞지 않는다. 필요하면 주 실험 후 별도 sensitivity arm으로 설계한다.

