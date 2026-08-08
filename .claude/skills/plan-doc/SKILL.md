---
name: plan-doc
description: 기획/설계 plan 문서(P##/S##/CPR)의 신설·패치·suffix 이관을 처리한다. "P## 신설", "S## 플랜 작성", "CPR 라운드 N", "_absorbed로 이관", "_superseded 처리", "문서 정합성 검증", "§4.4 미결 섹션" 같은 지시에 사용한다. 사용자가 "플랜 문서"라고 명시하지 않더라도 기획 문서·cross-plan review·접미사 전환·단일 소스 참조 같은 맥락이 감지되면 이 스킬을 호출한다. 문서 골격 템플릿, 접미사 규칙 6종을 번들로 제공한다. 하위 플랜에 수치·enum·수식이 silent 중복되는 실수를 감지·교정한다. 문서 디렉토리와 단일 소스 맵은 `.claude/project.json`의 `planDir`·`singleSource`로 설정한다.
---

# plan-doc

`planDir`(기획 문서 디렉토리) 관행을 강제하며 plan 문서를 생성·패치한다. 프로젝트 중립이며, 문서 위치·단일 소스는 [`.claude/project.json`](../../project.json)에서 읽는다 (`planDir`, `planDoneDir`, `singleSource`).

## 목적

세 종류의 작업을 자동화한다:
1. **P##/S## 플랜 신설** — 표준 섹션 구조(Context → Goals → Non-goals → Steps → Verification → Risks → Follow-up)로 초안 작성
2. **Cross-Plan Review(CPR) 작성** — `CPR_<scope>_<title>_<suffix>.md` 네이밍, §4.4 미결 섹션 `(a) 현재 (b) 옵션 (c) 권장` 3층 구조
3. **Suffix 이관** — `_planned` → `_done` / `_absorbed` / `_superseded` / `_canceled` / `_revised_done` 전환 시 헤더 배너·경로 수정 자동화

부수 기능: 하위 플랜이 상위 단일 소스의 수치·enum·수식을 silent 중복할 때 경고하고 참조 링크로 대체 제안.

## 언제 이 스킬을 사용하나

### 반드시 사용
- "S82 신설해줘, X 기능" 같은 신규 플랜 요청
- "CPR 라운드3 작성" / "Cross-plan review 작성"
- "P## 를 _absorbed로 이관" / "_superseded 처리" / "done 폴더로 이동"
- "P-doc 간 정합성 확인" / "silent 중복 체크" / "단일 소스 참조로 치환"
- "§4.4 미결 섹션 작성"

### 사용하지 않음
- plan 디렉토리와 무관한 일반 문서 작업 (README, 코드 주석)
- 코드 구현 자체 (구현 플랜 작성은 이 스킬, 구현은 직접 처리)

## 워크플로우

### (1) 신규 플랜 작성 요청
1. **문서 타입 판별**:
   - `P##` — 시스템/기능 플랜 (구현 지향)
   - `S##` — 제안/사이드카 (승인 시 P## 격상 가능)
   - Umbrella/총괄 — 여러 서브플랜의 상위 문서. `plan_umbrella.md` 템플릿.
2. **번호 자동 제안**: `ls <planDir>/`로 기존 번호 스캔 → 다음 번호.
3. **템플릿 선택**: `assets/templates/plan_basic.md`(기본), `plan_umbrella.md`(총괄), CPR이면 `cpr.md`.
4. **Context 섹션 채우기**: 관련 CPR 링크, 선행 플랜 링크, 기준 코드 경로를 **상단에 명기**.
5. **단일 소스 검증**: `.claude/project.json`의 `singleSource` 참조. 등재된 수치·enum·수식은 **silent 중복 금지** → "<문서> §<위치> 참조" 형태로 작성.
6. **파일명**: `P##_Descriptive_Name_planned.md` (신설은 무조건 `_planned`).

### (2) Suffix 이관 요청
1. **의도 확인**:
   - 병합·구현 완료 → `_done.md`
   - 구조 변경 후 재완료 → `_revised_done.md`
   - 철회 (재시도 없음) → `_canceled.md`
   - **더 큰 계획으로 교체** (방향 변경) → `_superseded.md`
   - **타 문서 부록으로 흡수** (방향 유지, 단순 이관) → `_absorbed.md`
2. `references/suffix_rules.md`에서 각 suffix의 의미·선택 기준 확인.
3. 파일 이동: `<planDir>/X_planned.md` → `<planDoneDir>/X_<new_suffix>.md`.
4. **헤더 배너 추가**: "**이 문서는 YYYY-MM-DD에 ABC로 absorbed/superseded 되었습니다. 결정 근거 아카이브 목적으로 보존됩니다.**" + 대체 문서 링크.
5. 상대 경로 조정 (하위 디렉토리로 이동 시 `./` → `../`).
6. 다른 문서에서의 역참조 grep → 링크 업데이트 필요 여부 보고 (자동 수정 금지, 사용자 확인 후).

### (3) CPR 작성
1. 파일명: `CPR_<scope>_<title>_<suffix>.md`. 예:
   - `CPR_<scope>_initial_done.md` (1라운드)
   - `CPR_<scope>_round2_done.md` (2라운드)
   - `CPR_<scope>_followup_planned.md` (진행 중 후속)
2. `assets/templates/cpr.md` 사용. 필수 섹션:
   - §0 Context / §1 결정 확정 항목 (표) / §2 추가 분석 / §3 오인 기재 / §4 패치 대상 요약
   - §4.4 미결(Unresolved) — **모든 미결에 (a) 현재 상태 (b) 옵션 분석 (c) 권장안**
   - §5 사용자 확인 필요 항목 (다음 턴 AskUser 대상)
3. 2라운드 이상이면 **단일 파일 누적 금지** — 새 라운드는 새 파일.

### (4) Silent 중복 감지
`.claude/project.json`의 `singleSource`에 등재된 수치·enum·수식이 하위 문서에서 **literal로 재정의**되면 경고한다. 각 항목은 `{ item, source, refForm }` 형식:

```jsonc
"singleSource": [
  { "item": "<식별자/수식 이름>", "source": "<문서 §위치>", "refForm": "<참조 표기 예>" }
]
```

감지 패턴: 하위 문서에서 `Literal["A", "B", ...]` enum 블록, 구체 수치, 구체 수식이 단일 소스와 중복 재등장. 이 경우 "<source> 참조. 본 문서 내 재정의 금지." 형태로 치환 제안.

`singleSource`가 비어 있으면 이 검증은 skip (경고 없음).

## 섹션 순서 (Plan 문서 기본 구조)

모든 P##/S## 플랜은 다음 순서를 고정:

1. **§0 Context** — 왜 이 변경이 필요한가. 선행 플랜·CPR 링크·기준 코드 경로.
2. **§1 목표 (Goals)** — 번호 달린 목표 리스트
3. **§2 비목표 (Non-goals)** — 명시적 범위 밖 항목
4. **§3 (옵션) 결정 요약·파라미터·수식** — 단일 소스 참조 우선. 신규 정의 시에만.
5. **§N 단계 (Steps)** — 구현 단계별 세부
6. **§검증 (Verification)** — 테스트 항목·검증 기준
7. **§리스크 (Risks)** — "리스크 \| 영향 \| 완화" 표 형식
8. **§후속 (Follow-up)** — 의존·후속 작업

## Suffix 규칙 요약

| Suffix | 의미 | 위치 |
|-|-|-|
| `_planned` | 진행 중/예정 | `planDir` 루트 |
| `_done` | 구현·병합 완료 | `planDoneDir` |
| `_revised_done` | 수정·재완료 | `planDoneDir` |
| `_canceled` | 철회 (재시도 없음) | `planDoneDir` |
| `_superseded` | **더 큰 계획으로 교체** (방향 변경) | `planDoneDir` |
| `_absorbed` | **타 문서 부록으로 흡수** (방향 유지) | `planDoneDir` |

상세는 `references/suffix_rules.md` 참조.

## 규약 유지 체크리스트

플랜 작성 완료 전 체크:
- [ ] 섹션 순서가 Context → Goals → Non-goals → Steps → Verification → Risks → Follow-up?
- [ ] 리스크 섹션이 "리스크 \| 영향 \| 완화" 표 형식?
- [ ] 관련 CPR·선행 플랜·기준 코드 경로가 상단에 있나?
- [ ] 단일 소스 수치·enum·수식이 silent 중복되지 않나? (`project.json`의 `singleSource` 대조)
- [ ] TBD 미결 항목이 있다면 (a) 현재 (b) 옵션 (c) 권장안이 모두 있나?
- [ ] CPR 문서면 §4.4 미결 섹션이 3층 구조인가?
- [ ] 파일명 suffix가 실제 상태와 일치하나?

## 번들 리소스

- `references/suffix_rules.md` — 6종 suffix 의미·선택 기준·이관 절차
- `references/single_source_map.md` — 단일 소스 맵 작성 가이드 (프로젝트별로 `project.json`에 채움)
- `references/document_types.md` — P##/S##/CPR/Umbrella 구분 및 작명 가이드
- `assets/templates/plan_basic.md` — P##/S## 기본 템플릿
- `assets/templates/plan_umbrella.md` — 총괄(Umbrella) 플랜 템플릿
- `assets/templates/cpr.md` — Cross-Plan Review 템플릿

## 작성 원칙

- **프로젝트 문서 언어 규약 준수**: 이 프로젝트는 "한글 본문 · 영문 식별자"(섹션 제목·본문은 한글, 변수·파일명·enum은 영문)를 따른다. 규약은 CLAUDE.md가 정본.
- **추측성 세부 규칙 금지**: 사용자가 요청하지 않은 구체 수치·규칙을 임의 추가하지 않는다. 미결은 §4.4로 모아 권장안과 함께 AskUser에 태운다.
- **즉시 inline 반영**: 사용자 결정이 나오면 다음 질문 전에 문서에 정착 → 재확인 → 다음.
- **§4.4 미결 규약**: "어느 것?" 단답형 질문 금지. (a) 현재 (b) 옵션 (c) 권장안 3단 동반.
- **다단계 follow-up**: 미결이 많을 때 세대별 follow-up으로 진행. 매 단계마다 이전 답변 inline 정착.
