#!/bin/bash
# merge_and_sync.sh — PR 머지 + 분기 base branch 로컬 동기화 (1 호출)
#
# 정책 (2026-05-04~): 머지 후 sync 대상은 PR 의 baseRefName (분기 base).
# main 이 아닐 수 있음.
#
# 사용:
#   bash merge_and_sync.sh <PR_NUMBER>
#
# 출력:
#   OK pr=NN merged synced base=<branch>
#   FAIL pr=NN state=<X>
#
# Exit code: 0 성공, 비-0 실패
#
# 동작:
#   1. mergeStateStatus == CLEAN 검증
#   2. PR 의 baseRefName 조회 (sync 대상 확정)
#   3. gh pr merge --merge --delete-branch
#   4. base branch 가 다른 worktree 에 checkout 중이 아니면 로컬 fast-forward
#      (worktree 충돌 시 origin 만 fetch, 알림 출력)

# set -e 비활성화: gh pr merge 가 내부적으로 git checkout base 를 시도하며
# worktree 환경에서 fatal 을 내지만 원격 머지는 성공한 경우가 있음.
# 머지 성공 여부는 API(state==MERGED) 로 직접 검증.

PR=$1
if [ -z "$PR" ]; then
  echo "FAIL pr=missing usage='bash merge_and_sync.sh <PR_NUMBER>'"
  exit 1
fi

# gh 경로
if command -v gh >/dev/null 2>&1; then
  GH=gh
elif [ -x "/c/Program Files/GitHub CLI/gh.exe" ]; then
  GH="/c/Program Files/GitHub CLI/gh.exe"
else
  echo "FAIL pr=$PR reason=gh-not-found"
  exit 1
fi

# Pre-flight: state(OPEN/MERGED) + mergeStateStatus + baseRefName 동시 조회
info=$("$GH" pr view "$PR" --json state,mergeStateStatus,baseRefName \
  -q '.state + " " + .mergeStateStatus + " " + .baseRefName' 2>/dev/null || echo "ERROR ERROR ERROR")
pr_state=$(echo "$info" | awk '{print $1}')
merge_state=$(echo "$info" | awk '{print $2}')
base=$(echo "$info" | awk '{print $3}')

if [ -z "$base" ] || [ "$base" = "ERROR" ]; then
  echo "FAIL pr=$PR reason=no-base-branch"
  exit 1
fi

# 이미 머지된 PR: sync 만 수행하고 OK 반환
if [ "$pr_state" = "MERGED" ]; then
  git fetch -q origin "$base" 2>/dev/null || true
  echo "OK pr=$PR already-merged synced base=$base"
  exit 0
fi

# UNKNOWN: GitHub가 mergeability 계산 중 (PR 생성 직후 일시적). 재시도.
if [ "$merge_state" = "UNKNOWN" ]; then
  sleep 3
  info=$("$GH" pr view "$PR" --json state,mergeStateStatus,baseRefName \
    -q '.state + " " + .mergeStateStatus + " " + .baseRefName' 2>/dev/null || echo "ERROR ERROR ERROR")
  pr_state=$(echo "$info" | awk '{print $1}')
  merge_state=$(echo "$info" | awk '{print $2}')
  base=$(echo "$info" | awk '{print $3}')
fi

if [ "$merge_state" != "CLEAN" ]; then
  echo "FAIL pr=$PR state=$merge_state"
  exit 1
fi

# Merge — gh 내부 git 작업(checkout/pull) 실패는 무시, API 로 성공 여부 검증
"$GH" pr merge "$PR" --merge --delete-branch >/dev/null 2>/dev/null || true

# 머지 성공 여부를 API 로 직접 확인
merged_state=$("$GH" pr view "$PR" --json state -q '.state' 2>/dev/null || echo "UNKNOWN")
if [ "$merged_state" != "MERGED" ]; then
  echo "FAIL pr=$PR reason=merge-not-confirmed state=$merged_state"
  exit 1
fi

# Sync — 본 worktree 의 현재 branch 와 base 가 같으면 그냥 pull
# 다르면 base 가 다른 worktree 에 checkout 중일 수 있음.
current=$(git branch --show-current 2>/dev/null || echo "")
if [ "$current" = "$base" ]; then
  git pull -q origin "$base" 2>/dev/null || true
else
  # 원격 트래킹 브랜치만 업데이트 (로컬 ref 미변경 — worktree 충돌 방지)
  git fetch -q origin "$base" 2>/dev/null || true
fi

echo "OK pr=$PR merged synced base=$base"
