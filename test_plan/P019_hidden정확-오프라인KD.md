# 실험계획 P019 — hidden(E=256) 정확 오프라인 KD (P015 top-k 실패의 대안)

## 배경
P015(004): top-k 로짓 캐시는 고엔트로피 교사(ppl~45)에서 질량 손실로 **부적합**. 그러나 우리 head는
**factorized(E=256)**: `logits = h_E @ emb.weight^T` (emb: 32768×256 고정). 따라서 교사의 **pre-head
hidden h_E(256차)만 캐시하면 교사 로짓을 손실 없이 정확 복원** → top-k 문제 원천 제거, 교사 body forward 제거.

- 정확성 하한 = head 랭크 = **E=256**(dense 교사도 emb_rank=256). 그 이하는 근사(SVD/PQ).
- full-logit(V=32768) 캐시 대비 **128×(=V/E) 작고 정확**. emb.weight 는 한 번만 저장(16.8MB).
- 복원 비용 = 스텝당 `h_E @ emb^T`(256×32768) 소형 matmul — body forward보다 훨씬 쌈.

## 캐시 크기 (300M 토큰) 와 10GB 이하 목표
| 표현 | 크기 | 정확성 | 비고 |
|---|---|---|---|
| fp16 | 143 GB | 정확 | 과대 |
| int8(퍼채널) | 71.5 GB | ≈정확 | KD soft target엔 충분 |
| int4(팩) | 35.8 GB | 약손실 | |
| **PQ m=32(8bit/subvec)** | **8.94 GB** ✅ | 튜너블 근사 | **전 토큰 유지**(추천 1안) |
| PQ m=16 | 4.47 GB ✅ | 더 거침 | |
| int4 + 토큰 1/4 추출 | 8.94 GB ✅ | 근사+부분 | skip-forward 결합(추천 2안) |
| int8 + 토큰 1/8 추출 | 8.94 GB ✅ | ≈정확+부분 | |

**zstd/lzma 실측(모의 hidden, rank-32 구조):** fp16 1.08×, **int8 1.16×, int4 1.40×**. → **압축만으론
10GB 불가**. 목표 달성은 **PQ(곱양자화)** 또는 **양자화+토큰추출** 조합으로. (실제 hidden에서 zstd 재측정 권장.)

## 축소 방안 (다각도)
1. **정밀도**: fp16→int8(2×)→int4/NF4(4×). KD는 온도 완화된 soft target이라 양자화 내성 큼.
2. **PQ/VQ**: 256차를 m개 서브벡터로 쪼개 각 256엔트리 코드북(8bit) → m B/token. m=32→8.94GB, m=16→4.47GB.
   전 토큰 유지(부분추출 없이 10GB 이하 달성하는 유일 경로).
3. **SVD 랭크절단**: 실효랭크 r<256 이면 r차만 캐시(교사 hidden이 저랭크면 유효). r/256 배 축소, 근사.
4. **토큰 부분추출**: KD를 1/K 토큰에만(P017 skip-forward 와 결합). /K 축소.
5. **캐시 토큰 수 축소**: 100M만 캐시(quality-efficiency 상충 감수).
6. **zstd on-disk**: int4 위 +1.4× 보조(주력 아님).

## 구현 개요
- `kdcache` 명령 확장: 교사 forward 1회 패스에서 **h_E**(emb_up 투영 직후 256차) 를 스트림 저장.
  packer: (a) int8 퍼채널, (b) PQ(코드북 학습은 첫 수천 토큰 샘플로 k-means). emb.weight 별도 저장.
- 학습 시 reader: h_E 복원→`logits_t = h_E @ emb^T`→ 기존 온라인 KD와 **동일 KL**(top-k 아님, full soft).
  교사 forward 없음. skip-forward 옵션과 직교.
- 정확성 게이트: **k_online vs k_offline_hidden** val_loss 가 int8에서 일치(≈0.00x), PQ에서 격차 측정.

## 실험 매트릭스 (tiny 스모크 → m100)
| 태그 | 캐시 | 확인 |
|---|---|---|
| kh_online | (교사 forward) | 기준 |
| kh_int8 | h_E int8 71GB급(스모크는 축소) | val 일치(정확성) |
| kh_pq32 | PQ m=32 8.9GB | 격차 vs online(근사 손실 허용치) |
| kh_pq16 | PQ m=16 4.5GB | 더 공격적 축소의 손익 |

판정: kh_int8 이 online과 val 일치(정확성 입증) → kh_pq* 로 크기-품질 곡선. 목표 = **≤10GB에서
online 대비 격차 무시할 수준**이면 KD 교사비용을 **1회로 상각**(모든 KD 학생 재사용).

## tiny 스모크 (사용자 대리)
```
:: 교사(dense tiny) 준비 후
python run100m.py train --arch dense --tiny --data synthetic --tokens 2M --steps 40 --micro-bs 4 --seq 128 --accum 2 --tag dense
python run100m.py kdcache --tiny --data synthetic --tokens 2M --steps 40 --micro-bs 4 --seq 128 --accum 2   :: (h_E 캐시 빌드 — 구현 후)
python run100m.py train --arch tied --tiny --data synthetic --tokens 2M --steps 40 --micro-bs 4 --seq 128 --accum 2 --kd --kd-cache --tag kh_int8
```
(주: 현재 `kdcache` 는 top-k 로짓용. P019는 **h_E 캐시 모드 추가 구현 필요** — 계획 단계.)

## 비판/리스크
- 디스크 공간(≥9GB)·교사 1회 패스 비용은 남음. 단 I/O는 int8 9GB를 100분 스트리밍 = ~1.5MB/s로 무시 가능.
- PQ 코드북 학습·복원 오버헤드(작음). 정확성은 int8까지만 보장, PQ부터는 실측 필요.
- 교사가 바뀌면 캐시 무효 → 교사 확정 후 1회 빌드가 전제.
