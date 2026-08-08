# Plan 파일 Suffix 규칙

`planDir` 폴더의 suffix는 **문서의 현재 상태**를 표시한다. 6종. (`planDir`·`planDoneDir` 경로는 `.claude/project.json`에서 읽는다.)

## 전체 목록

| Suffix | 의미 | 위치 | 이관 트리거 |
|-|-|-|-|
| `_planned` | 진행 중 / 구현 대기 | `planDir` 루트 | 신설 시 기본값 |
| `_done` | 구현·병합 완료 | `planDoneDir` | PR 머지 완료 + 모든 Steps 완료 |
| `_revised_done` | 수정·재완료 | `planDoneDir` | 기존 `_done` 문서의 구조적 수정 후 재완료 |
| `_canceled` | 철회 (재시도 없음) | `planDoneDir` | 사용자가 계획 자체 취소 선언 |
| `_superseded` | **더 큰 계획으로 교체** (방향 변경) | `planDoneDir` | 같은 문제를 다른 구조의 신규 플랜이 대체 |
| `_absorbed` | **타 문서 부록으로 흡수** (방향 유지) | `planDoneDir` | 내용이 다른 플랜의 부록·섹션으로 단순 이관 |

## `_superseded` vs `_absorbed` 구분

혼동이 잦다. 핵심 차이:

- **`_superseded`**: *방향성 자체가 바뀜*. 원 문서는 참고용, 신규 문서가 주도.
  - 예: 일체형 설계(P07) → 5분할 설계(P07a~P07e)
- **`_absorbed`**: *방향성 유지, 위치만 이동*. 원 문서 내용이 타 문서 부록에 그대로 들어감.
  - 예: 별도 제안(S08) → 관련 플랜(P07) §부록으로 흡수

## 이관 절차

### `_planned` → `_done` (가장 일반)
1. PR 머지 확인
2. 모든 Steps 체크박스 완료 확인
3. 파일 이동: `git mv <planDir>/X_planned.md <planDoneDir>/X_done.md`
4. 상대 경로 조정: `](./` → `](../`
5. 문서 상단에 구현 완료 마커 (선택): `> **Status**: Merged in PR #NN on YYYY-MM-DD.`
6. 다른 문서의 역참조 grep → 링크 업데이트

### `_planned` → `_canceled`
1. 철회 사유 기록 섹션 `## Cancellation notice` 상단 추가
2. 파일 이동: `<planDoneDir>/X_canceled.md`
3. 대체 플랜이 있다면 링크

### `_planned` → `_superseded`
1. 상단 배너:
   ```
   > **Status (YYYY-MM-DD)**: 이 문서는 [신규 플랜 링크]로 대체되었습니다.
   > 방향성 변경에 따른 설계 교체로, 본 문서는 결정 근거 아카이브 목적으로 보존됩니다.
   ```
2. 파일 이동: `<planDoneDir>/X_superseded.md`
3. 신규 플랜 §0 Context에 "기반 문서: X_superseded.md" 링크 명기

### `_planned` → `_absorbed`
1. 상단 배너:
   ```
   > **Status (YYYY-MM-DD)**: 이 문서는 [흡수 대상 문서 §N]에 absorbed 되었습니다.
   > 방향성은 유지되며, 구현·상세는 흡수처에서 관리됩니다. 본 문서는 원안 아카이브입니다.
   ```
2. 파일 이동: `<planDoneDir>/X_absorbed.md`
3. 흡수처 문서에 "이 섹션은 X_absorbed.md에서 흡수됨" 각주 (선택)

### `_done` → `_revised_done`
1. 구조적 수정이 아니라면 **내부 섹션 업데이트**로 족함. suffix 변경 불필요.
2. 근본적 재작업일 때만 새 파일 `X_revised_done.md` 생성하고 원본 `_done`은 유지 (둘 다 공존 허용).

## 엣지 케이스

- **`_planned`가 `planDoneDir`에 있는 경우**: 다른 방식으로 흡수·보류된 레거시. 신규 작업에서는 이 패턴을 **반복하지 않는다**.
- **여러 suffix 혼용 금지**: 한 문서는 단일 suffix만. "_planned_done" 같은 합성 suffix 금지.
