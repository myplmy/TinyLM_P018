---
name: pr-workflow
description: PR을 안전하게 생성·머지한다. "PR 생성", "merge 수행", "브랜치 만들고 푸시", "pull request" 같은 지시에 반드시 사용한다. Base branch는 detect_base.sh로 자동 감지(또는 사용자 확인)하고, Head는 현재 branch(HEAD)다. `git add .` / `-A` 를 차단하여 명시 경로 staging만 허용하고, `.claude/project.json`의 `excludedPaths`에 등재된 영구 제외 경로를 자동 필터링한다. Commit 메시지 HEREDOC + Co-Authored-By 라인, PR body "## Summary / ## Test plan" 템플릿을 표준으로 적용한다. gh CLI 경로를 OS별로 자동 해결한다. 사용자가 PR 관련 의도를 보이면(base 언급, 커밋 메시지 초안, push/merge 언급 등) 명시 지시가 없어도 이 스킬을 호출한다.
---

# pr-workflow

PR 생성·머지 표준을 강제한다. **scope 사고(`git add -A`로 제외 경로 포함)** 를 원천 차단한다. 프로젝트 중립이며, 프로젝트 고유 값(제외 경로·base 브랜치·정본 문서)은 [`.claude/project.json`](../../project.json)에서 읽는다.

## 왜 이 스킬이 필요한가

일부 경로(개인 로컬 설정·대용량 데이터·재생성 산출물·비밀키)는 **영구 untracked**로 두어야 한다. 일반적 PR 관행대로 `git add .`를 쓰면 이들이 우발적으로 포함된다. 이 스킬이 그 관행을 내재화한다.

## 언제 이 스킬을 사용하나

### 반드시 사용
- "PR 만들어" / "pull request 생성" / "PR 올려"
- "merge 수행" / "PR 머지해"
- "브랜치 만들고 푸시" / "현재 작업을 PR로"
- 사용자가 `gh pr create` 커맨드를 초안으로 작성한 경우

### 사용하지 않음
- 로컬 커밋만 원할 때 (push 없이) — 일반 git 커밋으로 진행하되 scope 룰은 동일 적용
- PR 조회만 하는 경우 (`gh pr list`, `gh pr view`)

## 핵심 규약 (비협상)

### 1. Base = detect_base.sh 자동 감지, Head = 현재 branch

Base 결정은 `scripts/detect_base.sh`가 자동화 (자세한 가이드: [`references/base_branch_detection.md`](references/base_branch_detection.md)):

```bash
BASE=$(bash .claude/skills/pr-workflow/scripts/detect_base.sh 2>/tmp/db.err)
EC=$?

if [ $EC -eq 2 ] || [ "$BASE" = "AMBIGUOUS" ]; then
    # confidence LOW/NONE — 사용자에게 AskUserQuestion 으로 문의
    # 후보 목록은 /tmp/db.err 의 candidates= 라인 참조
    # 결정 후: bash detect_base.sh --write <chosen>
    ...
fi

HEAD=$(git branch --show-current)
gh pr create --base "$BASE" --head "$HEAD" ...
```

우선순위: `--base` 명시 인자 → `.claude/project.json`의 `baseBranch` → 메타 파일 → env `PR_BASE_BRANCH` → 휴리스틱(첫 commit 포함 local branch). 기본 우선순위 후보는 `main master develop`.

**사용자 문의 절차** (detect_base가 LOW/NONE 반환 시 필수):
1. `AskUserQuestion`으로 후보 목록 제시 (옵션 라벨 = branch 이름)
2. 각 옵션 description에 `git log -1 --oneline <branch>` 결과 포함
3. 사용자 선택 후 `bash detect_base.sh --write <chosen>`으로 메타 파일 영구 기록
4. 이후 이 worktree의 모든 PR은 그 base 사용 (HIGH confidence)

### 2. `git add .` / `-A` 금지

스테이징은 **반드시 명시 경로**로:

```bash
git add path/to/file1.md path/to/file2.py
```

이유: `.claude/project.json`의 `excludedPaths`에 등재된 경로(개인 로컬 설정·대용량 데이터·재생성 산출물·비밀키)가 untracked로 유지되며, 전체 staging은 이들을 우발적으로 포함시킨다.

사용자가 "전체 추가" 요청 시 **거부하고 대안 제시**:
1. `git status --porcelain`으로 변경 목록 확인
2. `excludedPaths` 필터링 (`scripts/safe_stage.sh` 참조)
3. 필터링 후 남은 파일 목록을 사용자에게 제시 → 확인 후 `git add <paths>`

**영구 제외 경로는 `.claude/project.json`의 `excludedPaths`가 단일 소스**다. 기본값: `.claude/settings.local.json`, `.env`. 프로젝트별 추가 경로는 그 배열에 등재한다. 상세: [`references/excluded_paths.md`](references/excluded_paths.md).

### 3. Commit 메시지 포맷

- 제목: 70자 이하, 대문자 시작, 명령형
  - ✅ `Docs: generalize workflow skills + add project.json config`
  - ❌ `update some docs`
- 본문: 1줄 공백 후 변경 요약 (bullet)
- 마지막 줄: `Co-Authored-By: Claude <noreply@anthropic.com>` (환경에 따라 모델명 포함 가능)
- HEREDOC 사용:

```bash
git commit -m "$(cat <<'EOF'
제목 한 줄

- 변경 1
- 변경 2

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

### 4. PR body 포맷

```markdown
## Summary

- 핵심 변경 1~3개 (bullet)
- 기술적 맥락·근거 필요 시 한 줄 추가

## Test plan

- [ ] 테스트 항목 1
- [ ] 테스트 항목 2

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

**Doc Impact (조건부 섹션)**: `.claude/project.json`의 `docImpactTargets`에 정본 문서가 등재돼 있으면 아래 섹션을 추가하고 각 문서 영향을 평가한다. 배열이 비어 있으면 **섹션 자체를 생략**한다.

```markdown
## Doc Impact

- [ ] <doc 1> — 영향 없음 / 갱신: §X.Y
- [ ] <doc 2> — 영향 없음 / 갱신: §Y
```

영향 있으면 동일 PR에 동기 갱신 커밋을 포함하고 갱신 위치를 기재, 없으면 "영향 없음" 명시. 템플릿 변형은 [`references/pr_body_template.md`](references/pr_body_template.md) 참조.

**체크박스 표기 기준** — 작성 시점에 실제 수행·검증을 마친 항목만 `[x]`, 미수행은 `[ ]`로 둔다.

### 5. gh CLI 경로

`gh`가 PATH에 없을 수 있으므로 경로를 자동 해결한다:

```bash
GH="$(command -v gh || echo '/c/Program Files/GitHub CLI/gh.exe')"
"$GH" pr create --base "$BASE" --head "$HEAD" ...
```

PowerShell 환경이면 `gh`로 단축 가능. 스크립트는 두 환경 모두 지원. 상세: [`references/gh_cli_paths.md`](references/gh_cli_paths.md).

## 워크플로우 (전체 흐름)

> **토큰 효율 원칙**: git/gh 호출 출력은 LLM 컨텍스트로 직접 들어간다. 인간 친화 정보(diff stat, fast-forward 메시지, LF→CRLF warning, "Already up to date")는 토큰 손실 — `-q`/`--quiet` 옵션과 stderr 리다이렉션을 적극 활용한다.

### (1) 사전 확인
1. 현재 브랜치 확인: `git branch --show-current` (이것이 PR head)
2. **base 결정**: `bash scripts/detect_base.sh 2>/tmp/db.err`. exit code 2 (LOW/NONE) 시 `AskUserQuestion`으로 문의 → 결정 값 `--write`로 영구 기록.
3. base 최신화: `git fetch -q origin "$BASE"`
4. `git status --short -uno`로 변경 파악 (`-uno`는 untracked 노이즈 차단)
5. gh auth 상태: `"$GH" auth status` (1회/세션)

### (2) 작업 branch 준비
현재 working branch에서 작업 중이 아니면: `git checkout -b <descriptive-branch-name>`. 이미 적절한 branch면 생략.

### (3) 안전 staging (LF/CRLF warning 억제)
`scripts/safe_stage.sh` 실행 또는 수동:
1. `git status --porcelain` → 변경 파일 목록
2. `excludedPaths` 필터링
3. 사용자에게 최종 목록 보여주고 확인
4. `git add <filtered paths> 2>/dev/null` — Windows `core.autocrlf=true`에서 파일별 LF→CRLF warning 폐기 (진짜 에러는 비-zero exit로 신호되므로 안전)

**주의 — 금지 패턴**: `git add .` / `git add -A` — 제외 경로가 `.gitignore` 누락 시 우발 staging. 본 스킬 핵심 룰 §2 위반.

### (4) Commit
위 "Commit 메시지 포맷" 준수. HEREDOC 사용.

### (5) Push (출력 압축)
```bash
git push -u --quiet origin <branch-name>
```

### (6) PR 생성
`scripts/create_pr.sh` (자동 base 감지) 또는 수동:
```bash
HEAD=$(git branch --show-current)
BASE=$(bash .claude/skills/pr-workflow/scripts/detect_base.sh 2>/tmp/db.err) \
  || { echo "base detection failed — see /tmp/db.err"; exit 1; }

"$GH" pr create --base "$BASE" --head "$HEAD" --title "..." --body "$(cat <<'EOF'
## Summary
...
EOF
)"
```

`create_pr.sh`는 detect_base가 LOW/NONE 반환 시 exit 2 + stderr 안내. 그 경우 사용자 문의 후 `--base <chosen>` 명시하여 재호출.

### (7) Merge (사용자 요청 시) — 헬퍼 스크립트 권장

**권장 (1라인 OK 출력)**:
```bash
bash .claude/skills/pr-workflow/scripts/merge_and_sync.sh <NN>
# OK pr=<NN> merged synced base=<branch>
```

이 스크립트는 mergeStateStatus 사전 확인 → PR baseRefName 조회 → 머지 → 로컬 base branch 동기화까지 1회 호출로 처리. 성공 시 1줄, 실패 시 `FAIL pr=<NN> state=<X>` 1줄.

머지 전 확인: `mergeStateStatus == "CLEAN"`, CI 체크가 있다면 통과.

### (8) 보고
사용자에게 PR URL을 보고. 머지된 경우 `(merged)` 추가.

### (9) 이슈 close 코멘트 (해당 PR이 이슈를 해소할 때)

PR이 GitHub 이슈를 해소하면 **머지 후** 그 이슈에 해소 코멘트를 남기고 close한다.

**작성 방법 (비협상)**:
- `gh issue close <N> --comment "..."`의 인라인 문자열로 **긴 코멘트를 한 줄로 밀어넣지 말 것**. GitHub 마크다운은 문단 사이 **빈 줄**이 있어야 렌더되므로, 한 줄 코멘트는 헤딩·리스트가 뭉개진다.
- 반드시 **파일(HEREDOC)로 작성 → `--body-file`**로 전달:
  ```bash
  cat > "$SCRATCH/close_<N>.md" <<'EOF'
  ...아래 구조...
  EOF
  gh issue close <N> --comment "$(cat "$SCRATCH/close_<N>.md")"
  # 또는 already-closed 이면: gh issue comment <N> --body-file "$SCRATCH/close_<N>.md"
  ```

**구조 (§ 헤딩 필수)**: `## ✅ 해소` → `### 구현 PR` → `### 내역` → `### 검증` → `### 잔여/후속`. **PR 번호 링크 + 빈 줄 문단 구분은 생략 불가.**

- **`### 구현 PR` = 이슈를 실제로 해소한 PR을 전부 나열** (한 이슈를 여러 PR로 해소하면 다수 항목):
  ```markdown
  ### 구현 PR
  - #<PR-a> (<요약>)
  - #<PR-b> (<요약>)

  (참고) 부수 PR:
  - #<PR-plan> (plan)
  ```
  구현 PR과 부수 PR(plan/핸드오프)은 `(참고)`로 구분. 단일 PR이면 항목 1개로 축약.
- **`### 내역`** — 번호 리스트로 변경점.
- **`### 검증`** — 회귀 테스트 수치·영향.
- **`### 잔여/후속`** — 남은 항목·분리 이슈 (없으면 "없음").

**PR ↔ 이슈 링크 (필수)**: 이슈만 보고도 어느 PR이 해소했는지 추적 가능해야 한다. `### 구현 PR` 절의 PR 번호 링크가 그 추적 고리이므로 **생략 금지**.

## 에러 복구

### ".gitignore로 무시된 파일이 스테이징 됨"
→ 제외 경로가 `-A`로 끌려들어간 경우. 언스테이징: `git reset HEAD <path>`

### Pre-commit hook 실패
**amend 금지**. 문제 수정 후 **새 커밋** 생성.

### "gh: command not found"
→ 경로 자동 해결: `command -v gh || echo '/c/Program Files/GitHub CLI/gh.exe'`

## 번들 리소스

- `scripts/safe_stage.sh` — `excludedPaths` 필터링 staging 래퍼
- `scripts/detect_base.sh` — base 결정 (project.json → 메타 → env → 휴리스틱, confidence 라벨)
- `scripts/create_pr.sh` — PR 생성 헬퍼 (자동 base 감지 + gh 경로)
- `scripts/merge_and_sync.sh` — PR 머지 + base branch 로컬 동기화 1라인 헬퍼
- `references/pr_body_template.md` — PR body 표준 템플릿 모음
- `references/gh_cli_paths.md` — OS·환경별 gh 경로 해결
- `references/excluded_paths.md` — 제외 경로 설정 방법 + 필터링 규칙
- `references/base_branch_detection.md` — base 자동 감지 정책·휴리스틱·사용자 문의 절차

## Do / Don't 요약

### DO
- Base: `detect_base.sh` 자동 감지. 모호하면 사용자 문의 + 메타 영구 기록.
- Head: 현재 branch (`git branch --show-current`)
- Stage: 명시 경로만
- Commit: HEREDOC + Co-Authored-By
- gh 경로: `command -v gh` 우선, 없으면 OS별 fallback
- 이슈 close: 코멘트를 HEREDOC 파일 → `--body-file`, `## 해소 → ### 구현 PR(#번호 링크) → ### 내역 → ### 검증 → ### 잔여` 구조

### DON'T
- `git add -A` / `git add .`
- `git commit --amend` (pre-commit hook 실패 후)
- `--no-verify` / `--no-gpg-sign` (사용자 명시 요청 없이)
- Force push to base branch
- detect_base의 LOW/NONE 결과를 무시하고 임의 base로 fallback — 반드시 사용자 문의
- 이슈 close 코멘트를 긴 인라인 문자열 한 줄로 (마크다운 안 렌더) · PR 번호 링크 생략
