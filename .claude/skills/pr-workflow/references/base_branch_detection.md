# Base Branch 결정 — `detect_base.sh` 사용 가이드

## 정책

PR base는 **현 worktree/branch가 분기한 대상 branch**다. 항상 `main`이라고 가정하지 않는다:
- 사용자가 default working branch에서 worktree를 분기해 작업하는 경우, PR/머지 목적지는 그 분기 base다.
- `main`으로의 직접 머지는 별도 절차 (사용자 직접 또는 명시 요청 시 `--base main` override).

이 결정을 자동화하기 위해 `scripts/detect_base.sh`를 사용한다.

## 우선순위 + 신뢰도

| 순위 | 출처 | confidence | 출력 |
|---|---|---|---|
| 1 | `--base <branch>` 명시 인자 | HIGH | branch 그대로 |
| 2 | 메타 파일 `$GIT_DIR/pr-base` 첫 줄 | HIGH | branch 그대로 |
| 3 | 환경변수 `PR_BASE_BRANCH` (caller가 `project.json`의 `baseBranch`를 주입) | HIGH | branch 그대로 |
| 4 | 휴리스틱 — 첫 commit을 포함하는 local branch (단일 후보 또는 우선순위 1개 매치) | MEDIUM | branch |
| 5 | 휴리스틱 다중 매치 (복수 우선순위 또는 비매치) | LOW | `AMBIGUOUS` + stderr 후보 |
| 6 | 감지 실패 (reflog 정보 없음 등) | NONE | `AMBIGUOUS` |

휴리스틱의 우선순위 후보 기본값은 `main master develop`이며, env `PR_BASE_PRIORITY`(공백 구분)로 override 가능.

종료 코드: `0` = HIGH/MEDIUM (사용 가능), `2` = LOW/NONE (사용자 확인 필요), `1` = 오류.

## 사용 패턴

### Pattern A — 신규 worktree 진입 직후 (메타 1회 기록)

```bash
RAW=$(bash .claude/skills/pr-workflow/scripts/detect_base.sh --candidates 2>/tmp/db.err)
EC=$?
CONF=$(grep -oE 'confidence=[A-Z]+' /tmp/db.err | head -1 | cut -d= -f2)

if [ $EC -eq 0 ]; then
    BASE="$RAW"                       # HIGH/MEDIUM — 그대로 사용
fi
if [ $EC -eq 2 ]; then
    # claude: AskUserQuestion 으로 후보 제시 → 사용자 선택 = $CHOSEN
    bash .claude/skills/pr-workflow/scripts/detect_base.sh --write "$CHOSEN"
fi
```

### Pattern B — 이후 호출 (메타 파일 자동 사용)

```bash
BASE=$(bash .claude/skills/pr-workflow/scripts/detect_base.sh)
# 메타 파일 있으면 즉시 반환 (HIGH)
```

### Pattern C — 명시 override (예: main 으로 직접 머지)

```bash
BASE=$(bash .claude/skills/pr-workflow/scripts/detect_base.sh --base main)
```

## 휴리스틱 알고리즘 (요약)

```
1. $GIT_DIR/logs/HEAD 첫 줄에서 첫 commit hash 추출 (worktree 신설 시점 HEAD)
2. git branch --contains <hash> 로 그 commit 을 포함하는 local branch 수집 (현재 branch 제외)
3. 후보 수:
   - 0개 → NONE
   - 1개 → MEDIUM
   - 2개 이상 → 우선순위(PR_BASE_PRIORITY, 기본 main/master/develop) 매치 시도
     - 매치 1개 → MEDIUM
     - 매치 0개 또는 2개 이상 → LOW (사용자 문의)
```

## 한계 (정직하게)

- **시간 경과로 후보 증가**: worktree가 오래되어 다른 branch들이 같은 ancestor commit을 포함하면 후보가 늘어난다.
- **메타 파일 권장**: 1회 사용자 확인 후 `--write`로 영구 기록하면 이후 자동.
- **커밋 이력이 없는 신규 리포**: 휴리스틱이 NONE을 반환한다. 첫 PR 전 `project.json`의 `baseBranch` 또는 `--write`로 base를 지정하는 것이 확실하다.

## 사용자 문의 절차 (claude 책임)

`detect_base.sh`가 LOW/NONE 반환 시:
1. `AskUserQuestion`으로 후보 목록 제시
2. 옵션 라벨: 후보 branch 이름 그대로
3. 옵션 description: 각 branch의 최근 commit 1줄 (`git log -1 --oneline <branch>`)
4. 사용자 선택 후 `bash detect_base.sh --write <chosen>`으로 메타 파일 기록
5. 이후 PR 생성·머지에 그 base 사용

## 관련 파일

- `scripts/detect_base.sh` — 검출 헬퍼
- `scripts/create_pr.sh` — `--base` 미지정 시 detect_base 호출
- `scripts/merge_and_sync.sh` — 머지 후 sync 대상도 PR의 baseRefName 기준
- `SKILL.md` §"Base 결정" 섹션
