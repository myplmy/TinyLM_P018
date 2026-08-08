# gh CLI 경로 해결

`gh`가 PATH에 등록돼 있지 않을 수 있다 (특히 Windows + Git Bash). 경로를 자동 해결한다.

## 경로 탐색 순서

1. `command -v gh` — PATH에 있으면 이 경로 사용
2. `/c/Program Files/GitHub CLI/gh.exe` — Windows 기본 설치 경로 (Git Bash)
3. `C:\Program Files\GitHub CLI\gh.exe` — CMD/PowerShell
4. `/usr/local/bin/gh` / `/opt/homebrew/bin/gh` — macOS / Linux

## 세션 내 경로 캐싱 (권장)

```bash
GH="$(command -v gh || echo '/c/Program Files/GitHub CLI/gh.exe')"
"$GH" pr create --base "$BASE" --head "$HEAD" ...
```

스크립트(`create_pr.sh`, `merge_and_sync.sh`)는 이 로직을 내장한다.

## PowerShell

```powershell
gh pr create --base $BASE ...
# 또는 전체 경로
& "C:\Program Files\GitHub CLI\gh.exe" pr create --base $BASE ...
```

## 인증 확인

```bash
"$GH" auth status
```

결과가 `not logged into any GitHub hosts`이면 `"$GH" auth login`.

**주의**: `gh auth login`은 interactive prompt 발생. 세션이 non-interactive면 **사용자에게 위임**하고 "로그인 완료 후 알려달라"고 요청.

## 자주 쓰는 명령

| 작업 | 명령 |
|-|-|
| PR 목록 | `gh pr list --state all --limit 10` |
| PR 상세 | `gh pr view NN --json state,mergeable,mergeStateStatus` |
| PR 생성 | `gh pr create --base "$BASE" --title ... --body ...` |
| PR base 수정 | `gh pr edit NN --base "$BASE"` |
| PR 머지 | `gh pr merge NN --merge --delete-branch` |
| 이슈 목록 | `gh issue list` |

> base 브랜치는 하드코딩하지 말고 `detect_base.sh` 결과(`$BASE`)를 사용한다.
