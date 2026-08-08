# 실험계획 P005 — muP 하이퍼파라미터 전이 + Muon 옵티마이저

## 목적
(스케일링 대비) 규모마다 LR 재탐색을 없애고(muP), 같은 loss까지 step·compute를 줄인다(Muon).
1.7B 확장 시 자체 dense 교사 학습 비용이 가장 큰 부담이므로 여기서 큰 이득 기대.

## 배경/근거
- Muon(Essential AI, arXiv:2505.02222): **대배치(critical batch 초과)에서 데이터 효율 유지**로
  AdamW 대비 compute-time Pareto 확장. Linear 가중치에 스펙트럴(Newton-Schulz) 업데이트,
  vector(embed/norm/bias)엔 AdamW 병용. muP와 결합해 HP 전이.
- **주의: 이점이 대배치에 집중** → 현재 소배치(131K)·단일 GPU엔 이득 제한적. 스케일/대배치 단계용.

## 선결·리스크
- **Muon × ternary STE 미검증**: 우리는 full-precision `self.weight`를 STE로 최적화. Muon이
  orthogonalize하는 gradient와 삼진의 상호작용을 반드시 ablation(fp16 vs ternary).
- muP: init·LR 스케일 규칙(폭에 따른 1/√fan_in 등) 구현 필요. coordinate check로 검증.

## 실험
### A. muP 전이 검증
1. 좁은 폭(예: d=256, tiny 계열)에서 LR 스윕 → 최적 LR* 확인.
2. muP 스케일 규칙 적용 후 넓은 폭(d=768 m100)에서 **예측 LR**과 실제 최적 LR 비교(전이 오차).
3. coordinate check: 폭을 바꿔도 활성/업데이트 크기가 일정한지.

### B. Muon vs AdamW (+ ternary 상호작용)
| 조건 | 옵티마이저 | 양자화 |
|---|---|---|
| A1 | AdamW(기준) | ternary |
| M1 | Muon(Linear)+AdamW(vector) | ternary |
| M2 | Muon+AdamW | fp16(삼진 off, 상호작용 분리용) |
- 동일 토큰·유효배치에서 **loss-vs-step / loss-vs-compute** 곡선. 대배치 이점 확인 위해 유효배치도 스윕.

## 판정
- muP: 전이 LR 오차 작으면(예 ±20%) 채택 → 스케일 실험서 LR 탐색 생략.
- Muon: 같은 loss까지 step/compute 30~50% 절감 재현 여부. **단 ternary(M1)에서 재현돼야 실전 채택**.

## 비고
- 단일 GPU라 "대배치→더 많은 디바이스로 시간 단축"의 절반만 검증 가능. 절감의 방향성 확인 위주.
- 구현: Muon(NS 반복 5스텝 orthogonalize) + param 그룹 라우팅. AdamW(fused)와 병행 가능.

## 부속(C) — 축소 스크리닝 워크플로우
- muP 전이가 되면, 아키텍처 후보를 **작은 프록시(작은 d/저토큰)에서 랭킹 → HP 전이 → 승자만 풀스케일**.
  반복 1회를 100분→10~20분으로. (별도 실험 아님 — P005의 muP가 enabler, P007이 스케일 곡선 제공.)
