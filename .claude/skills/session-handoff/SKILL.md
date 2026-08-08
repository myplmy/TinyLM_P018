---
name: session-handoff
description: 세션 종료 시 다음 Claude 세션이 맥락을 그대로 이어받을 수 있도록 핸드오프 메모리 파일을 생성한다. "세션 이전 메모리 만들어", "임시 기억 파일", "다음 세션에서 이어서", "핸드오프 정리", "세션 종료 전 핸드오프", "다른 세션이 참고할 메모" 같은 지시에 반드시 사용한다. 사용자가 세션 종료·마무리·다음 세션 언급을 암시하면 명시 지시가 없어도 이 스킬을 호출한다. 7개 표준 섹션(요약/완료사항/우선순위/성향/학습/착수가이드/치트시트)을 고정 템플릿으로 제공하고, 영구 규범 문서(CLAUDE.md)와 중복되는 내용은 자동으로 링크 참조로 축소한다. 저장 위치는 `.claude/project.json`의 `handoffDir`로 설정한다.
---

# session-handoff

세션 연속성을 유지하기 위한 **임시 메모리 파일**을 생성한다. 다음 세션이 이 파일을 읽는 즉시 이전 작업 맥락을 완전히 흡수하도록 설계. 프로젝트 중립이며 저장 위치 등은 [`.claude/project.json`](../../project.json)에서 읽는다.

## 왜 이 스킬이 필요한가

세션 간 결정 사항·진행 상황·사용자 성향은 축적되는데, 세션이 compact 또는 종료되면 맥락이 끊겨 다음 세션이 **같은 실수를 반복**하거나 **사용자 성향을 모른 채 응답**한다. 이 스킬은 그 간극을 메운다.

> **영구 memory와의 역할 구분**: 이 핸드오프 파일은 **진행 중 상태의 세션 스냅샷**(휘발성, 이어받은 뒤 폐기)이다. Claude Code의 파일기반 memory(`~/.claude/projects/.../memory/`)나 CLAUDE.md는 **영구 사실·규범**을 담는다. 영구 가치가 있는 사실은 핸드오프가 아니라 memory/CLAUDE.md로 승격한다.

## 언제 이 스킬을 사용하나

### 반드시 사용
- "세션 이전 메모리 만들어" / "핸드오프 파일 생성"
- "다음 세션이 이어받을 수 있게 정리"
- "오늘 작업 핸드오프에 기록"
- "세션 종료 전 메모리 작성"
- 사용자가 명시적으로 "이 세션을 통해서 claude가 학습한 내용" 요청
- PR 머지 직후 대형 마일스톤 완료 시 (사용자 요청 없어도 핸드오프 권장)

### 사용하지 않음
- 영구 규범 갱신 (CLAUDE.md는 정식 PR 대상, 영구 사실은 memory)
- 기획 문서 작성 (그건 `plan-doc` 스킬)

## 저장 위치 (동적 — worktree 자동 감지 + 설정)

```bash
WORKTREE_ROOT=$(bash .claude/skills/session-handoff/scripts/detect_worktree_root.sh)
# HANDOFF_DIR = .claude/project.json 의 handoffDir (예: ".claude/handoff")
# → $WORKTREE_ROOT/$HANDOFF_DIR/session_YYYYMMDD_<topic>.md
```

`detect_worktree_root.sh`는 `git rev-parse --show-toplevel`으로 현재 체크아웃된 working tree 루트를 반환한다. worktree에서 실행하면 worktree 경로, 메인 레포에서 실행하면 메인 레포 경로를 반환하므로 항상 올바른 위치에 파일이 생성된다.

- `HANDOFF_DIR`: `.claude/project.json`의 `handoffDir` 값. 미설정 시 `.claude/handoff` 기본.
- `YYYYMMDD`: 세션 종료일 기준 (today)
- `<topic>`: 세션의 주요 산출물·마일스톤을 영문 snake_case로 요약
  - 예: `session_20260711_feature_x_ready.md`
  - 예: `session_20260711_docs_patch.md`

## 표준 섹션 (7개 고정)

핸드오프 파일은 다음 7개 섹션을 순서대로 포함:

### § 1. 세션 요약
- 세션 시작일 ~ 종료일
- 세션 최대 성과 1~3개
- 머지된 PR 목록 (번호·제목·머지일·내용 한 줄 표)

### § 2. 현재까지 구현된 사항
- 코드/문서 분리 (병합 완료 범위 + 최근 PR 반영 문서)
- CLAUDE.md 갱신된 섹션 이름만 나열 (중복 금지)

### § 3. 향후 수행 우선순위
- **1순위**: 다음 세션이 **바로 시작 가능한** 작업. 상태·핵심 사실·PR 구조·테스트 전략 포함.
- **2순위 이하**: 표 형식. 순위 \| 작업 \| 의존
- **보류** 항목 명시

### § 4. 사용자 성향 / 세션 운영 스타일
- **이번 세션에서 확립·강화된 것만** 기재 (영구 규범은 CLAUDE.md 링크로)
- 꾸준히 요구한 사항, 세션 운영 스타일 특이점

### § 5. 이번 세션에서 학습·기억한 사항
- **프로젝트 구조 핵심** (이번 세션에서 처음 확정된 것)
- **프로젝트 핵심 수치** (단일 소스 포인터)
- **기술 스택 유의사항** · **세션 도구 팁**

### § 6. 다음 세션 착수 가이드
사용자가 "핸드오프 확인" 지시 시 이 섹션을 따라 행동하도록 **명령형 단계**로 작성:

```markdown
1. CLAUDE.md 읽기 (특히 X, Y, Z 섹션)
2. 본 파일 읽기 (session_YYYYMMDD_<topic>.md)
3. 관련 기획 문서 전체 재독
4. 코드 진입점 재확인: `<module>:NN-MM`
5. 새 feature branch 생성 커맨드
6. 첫 커밋 착수
```

주의사항도 함께 (실수 방지용).

### § 7. 참조 경로 치트시트
자주 참조할 절대 경로 목록: 기획 문서 root, 세션 문서, 프로젝트 규범(CLAUDE.md), 핵심 모듈 경로, 테스트 경로, 이전 세션 노트.

## 작성 원칙

### 원칙 1: 영구 규범 중복 회피
임시 메모리는 **진행 중 상태 스냅샷**. CLAUDE.md는 **영구 규범**. 둘이 중복되면 CLAUDE.md가 우선이므로, 임시 메모리에는 "CLAUDE.md 'X 섹션' 참조"만 남기고 실제 규범 문구는 복사하지 않음.

- ❌ (규범 본문을 그대로 복사)
- ✅ "PR base 관련: CLAUDE.md 'User collaboration style' 섹션 참조"

### 원칙 2: 이번 세션 고유 정보에 집중
임시 메모리의 가치는 **이번 세션에서 처음 확정된 것** 또는 **강화된 패턴**. 영구 사실은 CLAUDE.md/memory가 담당.

### 원칙 3: 다음 세션 착수 가이드가 가장 중요
§6은 핸드오프 파일의 핵심. 여기에 적힌 순서를 따라가면 다음 세션이 **바로 실제 작업에 착수** 가능해야 함. 모호한 지시 금지.

- ❌ "다음 세션에서 X 구현 착수"
- ✅ "1. CLAUDE.md 읽기 → 2. 이 파일 읽기 → 3. 기획 문서 읽기 → 4. git checkout -b ... → 5. Commit 1 착수..."

### 원칙 4: 저장 위치·버전관리는 프로젝트 정책
핸드오프 파일을 git 추적할지 여부는 프로젝트 정책이다. 추적한다면 명시 경로 `git add`로 staging하여 다음 PR 사이클에 포함하고, 효력이 끝난 메모는 다음 PR에서 정리한다. 추적하지 않는다면(`.gitignore`) 로컬 전용으로 둔다. `handoffDir` 위치와 추적 정책은 CLAUDE.md에 명기할 것.

## 작성 워크플로우

### (1) 세션 정보 수집
- **worktree 루트 확인**: `bash .claude/skills/session-handoff/scripts/detect_worktree_root.sh`
- 오늘 날짜: `date +%Y%m%d` 또는 세션 context의 today
- 머지된 PR 목록: `gh pr list --state merged --limit 5 --json number,title,mergedAt`
- 현재 branch: `git branch --show-current`
- 세션 중 수정된 파일: `git log <base>..HEAD --name-only` (merged 이후)

### (2) 토픽 결정
세션의 가장 큰 산출물·마일스톤을 영문 snake_case로 요약. 예: 기능 구현 준비만 하고 착수는 다음 세션이면 → `<feature>_ready`. PR을 3개 머지했으면 → `three_prs_merged`.

### (3) 파일 생성
`assets/templates/session_handoff.md` 템플릿을 복사하고 채워 넣음.

### (4) 기존 핸드오프 파일 확인 (선택)
폴더 내 기존 `session_*.md`가 있으면 마지막 파일을 읽어 **체인 연장** 형태로 작성. 이전 세션 내용을 복사하지 말고 "이전 세션: session_YYYYMMDD_prev.md" 링크만.

### (5) 사용자에게 생성 경로 보고
```
생성됨: <WORKTREE_ROOT>/<handoffDir>/session_YYYYMMDD_<topic>.md
```

### (6) compact 수행 여부 확인 + 프롬프트 제공
핸드오프 메모 작성·보고 직후, **`AskUserQuestion`**으로 compaction 수행 여부를 확인한다 (자유 텍스트 문의 금지).

- 옵션: "compact 수행" (세션 종료 맥락이면 첫 옵션 + Recommended) / "미수행".
- "compact 수행" 선택 시, 아래 §"compact 프롬프트 구성" 기준으로 작성한 compact 프롬프트를 코드 블록으로 제공한다. 사용자는 이 프롬프트를 `/compact` 인자로 사용한다.
- compaction 실행 자체는 사용자가 `/compact`로 트리거한다 — 스킬은 프롬프트 제공까지만.

## compact 프롬프트 구성

compact 프롬프트는 방금 작성한 핸드오프 메모를 **단일 소스**로 가리키고 핵심만 요약한다. 메모 전문을 복제하지 않는다 (토큰 낭비·drift 방지). 필수 요소:

1. **핸드오프 메모 경로** — `@<handoffDir>/session_YYYYMMDD_<topic>.md`를 단일 소스로 명시. "§3 우선순위·§6 착수 가이드 보존 필수" 지시 포함.
2. **현재 상태** — 본 세션 코드/문서 변경 1~2줄 요약.
3. **다음 착수 예정** — 다음 세션이 시작할 작업 (메모 §3 1순위와 일치).
4. **유지할 사용자 성향** — CLAUDE.md §"User collaboration style" 참조로 (규범 본문 복사 금지). 이번 세션 강화 성향만 간단히.
5. **미결 검토 항목** — 승인 대기·후속 검토 항목.

## 번들 리소스

- `scripts/detect_worktree_root.sh` — 현재 git worktree 루트 절대경로 반환
- `assets/templates/session_handoff.md` — 7개 섹션 템플릿
- `references/what_to_include.md` — 섹션별 수집 기준 (무엇을 쓰고 무엇은 생략)
- `references/examples.md` — 섹션별 작성 예시 (플레이스홀더 기반)

## 체크리스트

핸드오프 파일 완성 전 체크:
- [ ] 파일명이 `session_YYYYMMDD_<topic>.md` 형식인가
- [ ] 저장 위치가 `handoffDir`(project.json)인가
- [ ] 7개 섹션이 모두 있나 (요약/완료/우선순위/성향/학습/착수가이드/치트시트)
- [ ] CLAUDE.md와 중복된 규범이 복사되지 않았나 (링크로만 처리)
- [ ] §6 "다음 세션 착수 가이드"가 명령형·구체적 단계로 작성됐나
- [ ] §4 "사용자 성향"이 이번 세션 고유 내용인가
- [ ] 핸드오프 메모 작성·보고 후 compact 수행 여부를 `AskUserQuestion`으로 확인했는가 (워크플로우 (6))
- [ ] "compact 수행" 시 핸드오프 메모를 단일 소스로 한 compact 프롬프트를 제공했는가
