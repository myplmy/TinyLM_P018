---
name: check-and-verify
description: PR 또는 Issue의 Test plan 체크박스를 읽어 각 항목의 수행 여부를 실제로 검증하고, 통과한 항목의 체크박스를 자동 갱신하는 스킬. "PR 체크박스 검증", "체크박스 자동 체크", "test plan 검증", "PR #N 체크박스 확인하고 체크", "머지 전 체크박스 점검", "이 PR 테스트 항목 돌려봐" 같은 요청에 사용한다. 또한 사용자가 명시하지 않더라도 PR 머지 직전 Test plan 항목을 검증해야 하는 맥락이면 이 스킬을 사용한다. 체크박스를 눈으로만 보고 수동 체크하거나, 검증 없이 전부 체크하는 패턴을 차단한다.
---

# check-and-verify

PR / Issue body의 Test plan 체크박스를 **실제로 검증**하고 통과 항목만 체크한다.

## 왜 이 스킬이 필요한가

PR 머지 전 Test plan 체크박스 검증은 수작업이면 (1) 항목을 빠뜨리거나 (2) 검증 없이 전부 체크하는 실수가 난다. 이 스킬은 체크박스를 추출·분류하고, 자동 실행 가능한 항목을 **사용자 승인 하에** 실제로 돌린 뒤, 통과한 항목만 갱신한다 — 검증과 기록을 한 흐름으로 묶는다.

## 언제 사용하나

### 사용
- PR / Issue의 Test plan 체크박스를 검증·갱신해야 할 때
- PR 머지 직전 테스트 항목 실행 확인이 필요할 때

### 사용하지 않음
- 체크박스가 없는 PR / 단순 문서 PR (검증할 항목 자체가 없음)
- PR body 작성·머지 자체 — 그건 `pr-workflow`

## 승인 모델 (비협상)

이 스킬은 **두 번의 사용자 승인**을 거친다. 자동으로 명령을 돌리거나 PR body를 수정하지 않는다.

1. **실행 전 항목별 승인** — 자동 분류된 항목을 `AskUserQuestion` multiSelect로 제시하고, 사용자가 실행할 항목을 고른다. 고르지 않은 항목은 실행하지 않는다.
2. **PATCH 전 일괄 승인** — 실행 결과 보고 후, 체크박스 갱신 여부를 `AskUserQuestion`으로 1회 확인한다. 승인 시에만 `gh ... edit`으로 body를 갱신한다.

분기 결정은 모두 `AskUserQuestion`으로 한다. 후보가 5개 이상이면 한 차수 4개 이하로 나눈다.

## 워크플로우

### 1. 입력 식별 — PR / Issue 자동 판별

번호를 받으면 PR인지 Issue인지 판별한다. PR을 먼저 시도하고 실패하면 Issue.

```bash
GH="$(command -v gh || echo '/c/Program Files/GitHub CLI/gh.exe')"
N=<번호>
if "$GH" pr view "$N" --json body -q .body > /tmp/cav_body.md 2>/dev/null; then
    KIND=pr
elif "$GH" issue view "$N" --json body -q .body > /tmp/cav_body.md 2>/dev/null; then
    KIND=issue
else
    echo "FAIL: #$N 은 PR 도 Issue 도 아님"; exit 1
fi
```

body는 `/tmp/cav_body.md`에 저장된다. 이 파일이 이후 모든 단계의 입력.

### 2. 체크박스 추출·분류

```bash
python .claude/skills/check-and-verify/scripts/checkbox_tool.py classify /tmp/cav_body.md
```

JSON 배열을 반환한다 — 원소마다 `index, lineno, checked, text, category, command`.

분류 카테고리:

| category | 판정 근거 | 자동 실행 |
|---|---|---|
| `pytest` | 코드 스팬이 `python -m pytest ...` 또는 `pytest ...` | O — 명령 실행, exit 0 = PASS |
| `script` | 코드 스팬이 `python test/...` 또는 `python scripts/...` | O — 명령 실행, exit 0 = PASS |
| `script+env` | 위 명령에 `NAME=value` env prefix가 붙음 | O — env 포함 한 줄로 실행 |
| `file_check` | 존재/생성/저장 키워드 + 경로형 코드 스팬 | O — 경로 존재 확인 |
| `manual` | 위 어디에도 해당 없음 (수동 확인 항목) | X — 사용자 확인 |

분류는 **휴리스틱**이다. 오분류로 보이는 항목(예: 실제로는 수동인데 `script`로 잡힘)은 사용자에게 알리고 manual로 강등한다.

> 프로젝트의 테스트 명령 관행이 다르면(`test/`·`scripts/` 외 디렉토리, 다른 러너) `scripts/checkbox_tool.py`의 `classify_command` 정규식을 그 관행에 맞게 조정한다.

### 3. 분류 결과 보고

이미 `checked: true`인 항목은 **SKIP** (재실행 안 함). 미체크 항목을 카테고리별로 표로 보고한다.

### 4. 실행 항목 승인 (승인 1)

미체크 + 자동 가능(`pytest`/`script`/`script+env`/`file_check`) 항목을 `AskUserQuestion` multiSelect로 제시 — 사용자가 실행할 항목을 고른다. 항목이 5개 이상이면 차수를 나눈다. 각 옵션 라벨에 index + 명령을 넣어 무엇을 돌리는지 명확히 보이게 한다.

### 5. 승인 항목 실행

- **pytest / script / script+env**: 리포 루트에서 `command`를 그대로 실행. exit code 0 = PASS, 그 외 = FAIL. 출력이 길면 마지막 10~20줄만 인용.
- **file_check**: `command`가 경로다.
  - `/`로 끝나는 디렉토리: 비어있지 않은지 확인 — `[ -n "$(ls -A <path> 2>/dev/null)" ]`
  - 그 외: 파일 존재 — `test -f <path>`

명령은 체크박스 텍스트에서 추출된 것이므로, 실행 전 사용자가 승인한 항목만 돌린다 (승인 1). 화이트리스트(pytest·프로젝트 스크립트) 밖 명령은 애초에 자동 분류되지 않으므로 임의 명령은 실행되지 않는다.

### 6. 결과 집계 보고

| index | category | result | 요약 |
|---|---|---|---|
| 0 | pytest | PASS | 42 passed |
| 2 | script | FAIL | exit 1 — ImportError |

`result` ∈ PASS / FAIL / SKIP(이미 체크됨·미선택). FAIL 항목은 원인 한 줄 포함.

### 7. manual 항목 처리

`manual` 카테고리 + 미선택 항목을 사용자에게 보고한다. 사용자가 수동으로 확인한 항목이 있으면 `AskUserQuestion` multiSelect로 "완료된 수동 항목"을 받아 PATCH 대상에 포함한다.

### 8. 체크박스 갱신 승인 (승인 2)

PATCH 대상 = PASS 항목 + 사용자가 완료 확인한 manual 항목의 index 목록. `AskUserQuestion`으로 "이 N개 항목 체크박스를 갱신할까요?" 1회 확인.

### 9. body 갱신·PATCH

승인 시:

```bash
python .claude/skills/check-and-verify/scripts/checkbox_tool.py apply \
    /tmp/cav_body.md "0,2,3" > /tmp/cav_updated.md

# KIND 에 따라 분기
"$GH" pr edit  "$N" --body-file /tmp/cav_updated.md     # KIND=pr
"$GH" issue edit "$N" --body-file /tmp/cav_updated.md   # KIND=issue
```

`gh pr edit` / `gh issue edit`은 현재 리포 컨텍스트에서 동작하므로 owner/repo를 명시할 필요가 없다. `--body-file`은 파일을 UTF-8로 읽으므로 한글이 안전하다.

### 10. 최종 보고

갱신된 체크박스 수, 남은 FAIL·manual 항목을 보고한다. FAIL이 있으면 머지 부적합 신호로 명확히 전달한다.

## 번들 스크립트

`scripts/checkbox_tool.py` — 네트워크 호출 없는 순수 텍스트 변환 (프로젝트 중립).

- `classify <body_file>` — 체크박스 추출·분류 → JSON (stdout)
- `apply <body_file> <idx,idx,...>` — 지정 인덱스를 `[x]`로 바꾼 body 전문 (stdout)

stdout은 UTF-8로 고정되어 있어 리다이렉트 출력 시 한글이 손상되지 않는다. 콘솔 직접 표시 시 mojibake는 정상 — 파일·파이프 출력은 정확하다.

## gh CLI 경로

`gh`가 PATH에 없으면 경로 자동 해결: `command -v gh || echo '/c/Program Files/GitHub CLI/gh.exe'`. 상세는 `pr-workflow` 스킬의 `references/gh_cli_paths.md` 참조.

## 에러 복구

- **체크박스 0개**: classify가 `[]` 반환 → "검증할 Test plan 항목 없음" 보고 후 종료.
- **명령 FAIL**: 해당 항목은 체크하지 않는다. 원인을 보고하고 사용자가 수정하도록 둔다.
- **`gh ... edit` 실패** (권한·네트워크): updated body 파일 경로를 사용자에게 알리고 수동 갱신 안내.
- **오분류 의심**: manual로 강등하고 사용자 확인. 추측 실행 금지.

## 체크리스트

- [ ] PR / Issue 자동 판별했는가
- [ ] 자동 항목을 실행 전 `AskUserQuestion`으로 승인받았는가 (승인 1)
- [ ] 이미 체크된 항목을 재실행하지 않았는가 (SKIP)
- [ ] 결과 표(PASS/FAIL/SKIP)를 보고했는가
- [ ] 체크박스 갱신을 `AskUserQuestion`으로 일괄 승인받았는가 (승인 2)
- [ ] PASS 항목만 `[x]`로 갱신했는가 (FAIL 항목 미체크)
