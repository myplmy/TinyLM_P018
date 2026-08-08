# 실험계획 P012 (A) — 한국어 데이터 효율 (curated 교체)

## 목적
"더 많은 토큰"이 아니라 "더 좋은 데이터"로 목표 loss까지 **토큰(=학습시간)을 줄인다**.
현재 en=FineWeb-Edu(curated)지만 **ko=원시 위키** → 한국어를 curated로 올려 대칭 구성.

## 권장 데이터 (웹 조사)
- **Korean-webtext-edu (Elice, HF, ~190B 토큰)** — FineWeb-Edu 파이프라인의 한국어판(교육 가치 필터). **1순위**.
- FineWeb2 한국어 서브셋(HuggingFaceFW/fineweb-2), AI-Hub(NIA), 모두의 말뭉치(국립국어원).
- 참고 모델 파이프라인: Kanana, KORMo, Mi:dm(저→고 품질 커리큘럼 + 필터).

> **합성 데이터 오해 정정**: 사용자가 "합성 직접 생성은 비효율"이라 본 것은 **맞습니다**(강한 생성모델+연산 필요).
> 다만 **이미 curated/필터된 공개 한국어 코퍼스(위 Korean-webtext-edu 등)를 쓰는 것**이 효율적 경로입니다.

## 방법
- `DATASETS`에 `ko-edu-en` 믹스 추가: ko=Korean-webtext-edu 0.5 + en=FineWeb-Edu 0.5.
- 토크나이저는 새 믹스로 재학습(또는 기존 공유 — 비교는 같은 토크나이저로).

## 조건
- 동일 토큰(300M)에서 (a) 현행 ko-en(위키+fineweb) vs (b) ko-edu-en 의 dense val 비교.
- 저토큰(100M)에서 동일 목표 loss 도달에 필요한 토큰수 비교(데이터 효율 곡선).

## 판정
- 같은 토큰에서 val↓ 또는 같은 val까지 토큰↓. 절감폭이 크면 이후 전 실험의 기본 데이터로 채택 → **절대 학습시간 직접 절감**.

## 추가 실험 — URL/메타데이터 prepending (저비용 토큰 절감)
근거: *Beyond URLs: Metadata Diversity and Position for Efficient LLM Pretraining* (arXiv:2511.21613) —
문서 앞에 출처 URL/메타데이터를 prepend 하면 **30~40% 토큰 절감**(데이터 필터 위에 추가 이득) 보고.
FineWeb-Edu 계열은 URL·score 등 메타를 이미 보유(우리 `ko-edu-en` 의 eliceai 셋은 `score` 필드 있음).

- **방법**: `prepare._stream` 에서 각 문서 앞에 `"[url] {source}\n"` 또는 `"[score={n}]\n"` 를 삽입하는
  옵션(`--meta-prepend`). 학습 시 해당 접두는 loss 마스킹 없이 그대로(논문은 마스킹 불필요 보고).
- **조건**: 같은 데이터·같은 토큰에서 (a) prepend 없음 vs (b) URL prepend 의 목표 val 도달 토큰수 비교.
- **판정**: 같은 val에 토큰 30%↓면 채택 → 절대 학습시간 직접 절감(연산바운드에서 유효한 몇 안 되는 레버).
- **리스크**: 한국어 웹 메타(eliceai)는 URL 원본이 없을 수 있음(HAERAE-WEBTEXT 유래) → score/도메인
  태그로 대체. 효과는 영어(FineWeb-Edu, URL 보유) 대비 다를 수 있어 en/ko 분리 측정.

## 비고
- 새 데이터는 스트리밍(HF)로 캐시. 스케일별 이름으로 기존 ko-en과 공존.
- 데이터 효과는 ms/step 불변(연산바운드) → **토큰당 품질(bpb)** 및 **목표loss까지 토큰수**로 측정.
