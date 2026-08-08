# 실험계획 P015 — 오프라인 KD 캐싱 (교사 forward 제거)  [-done(004)]

## 목적
KD 학습시간 ~2×의 원인(매 스텝 dense 교사 풀 forward)을 제거. 교사를 **1회만** 데이터에 돌려
top-k soft target을 저장 → 이후 모든 KD 학생이 캐시 재사용(교사 forward 0).

## 정합성(핵심)
- 학습 loader(seed 1234)를 **동일 순서로 replay** → 크롭이 정확히 일치. 캐시는 lockstep 소비.
- **캐시는 (data, steps, micro_bs, seq, accum, seed)에 종속** — 하나라도 다르면 무효(재생성).
- 근사: 온라인=전체 vocab KL, 캐시=**top-k KL** → 온라인과 미세 차이(비교 시 유의).

## 사용
```cmd
:: 1) 캐시 생성(교사 1회 forward, ~교사 학습시간)
python run100m.py kdcache --data ko-en --tokens 300M --steps 2289 --micro-bs 8 --seq 1024 --accum 16 --kd-topk 32
:: 2) 학생 KD (교사 forward 없음)
python run100m.py train --arch tied ... --init-from --kd-cache --kd-topk 32 --tag t_kd_offline
```

## 검증(먼저 tiny로)
- 온라인 KD vs 캐시 KD의 학습 loss가 **가까운지**(top-k 근사라 정확히 같진 않음).
  → 결과 004: 가깝지 않았다. 부적합 판정.
- 통과 시에만 m100 캐시 생성 후 실사용.

## 판정
- 학생 val: 온라인 KD(t_kdinit/t_kd_g8)와 노이즈(±0.03) 내면 채택. 속도: 학생 스텝이 비-KD 수준으로 회복.
- 손익: 캐시 1회 비용(~교사 forward 1.6h) + 디스크(top16 ~29GB / top32 ~58GB) vs 매트릭스 전 학생의 교사비용 상각.

## 비고
- top-k(32 권장)와 vocab(32768) 트레이드오프: k 클수록 근사 정확·디스크↑.
- 오프라인 KD 학생은 온라인 KD 학생과 **직접 비교 시 근사차 유의**(별도 라인).

## 실측 결론(004) — 현재 부적합
- top-k KD는 **교사 집중도(ppl<~5)** 에서만 유효. 우리 교사 ppl 45 → top-32 질량 ~20% → under-distill.
- 지금은 미사용 권고. 정확 대안=E=256 중간표현 캐싱(~154GB). KD 2× 비용 감수 권장.
