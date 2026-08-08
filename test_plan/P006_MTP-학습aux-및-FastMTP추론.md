# 실험계획 P006 — MTP(학습용 aux 헤드) + FastMTP(추론 가속) [별개 2건]

> 용어 정리: **학습용 MTP**(DeepSeek-V3식 aux head)는 *수렴* 이득, **FastMTP**(Tencent,
> arXiv 첨부)는 *추론* 가속(speculative decoding, 2.03× 무손실). 서로 다른 목적 — 별개로 다룬다.

## Part A. 학습용 aux MTP 헤드 (수렴 가속)
### 목적
다음 토큰만이 아니라 미래 t+2..t+k 를 경량 헤드로 동시 예측하는 aux loss로 표현력·수렴 가속.

### 구현(안)
- 공유 factorized 헤드 위에 k−1개 경량 예측 경로(위치별 소형 프로젝션). 파라미터 최소.
- loss = CE(t+1) + λ·Σ_{j=2..k} CE(t+j),  k=2~4, λ 스윕.

### 판정
- 동일 토큰·시드에서 **val 수렴(loss-vs-step)**을 베이스라인(NTP only)과 비교. 이득이 노이즈 초과인지.
- 배포시 aux 헤드는 떼거나(추론 NTP) FastMTP(Part B)로 재활용.

### 리스크
- 소형·undertrained 모델에서 MTP 학습 이득은 불확실. tied·factorized 헤드와의 결합 설계 필요.

## Part B. FastMTP (추론 speculative decoding)
### 목적
자기회귀 병목 완화 → 배포 추론 2× 가속(무손실). CPU/엣지 지연 개선에 직접 기여.

### 선결
- Part A로 MTP 헤드 확보 → position-shared 단일 헤드를 self-distill 데이터로 정렬(추론 패턴 일치).
- language-aware dynamic vocab compression으로 draft 오버헤드 절감.

### 판정
- 7종 벤치에서 **acceptance rate · speedup**, 출력 무손실 확인. 표준 NTP 대비 배속.

### 비고
- FastMTP는 배포(추론) 최적화라 학습 격차·메모리 실험과 별개 트랙. CPU 삼진 커널과 결합 시 시너지.
- 구현 난이도: Part A 중간, Part B 높음(추론 프레임워크 통합).
