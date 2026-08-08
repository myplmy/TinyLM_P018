#!/bin/bash
# detect_worktree_root.sh — 현재 git worktree 루트 절대경로 반환
#
# git worktree / 메인 레포 모두 동일 원리로 동작.
# `git rev-parse --show-toplevel` 은 현재 체크아웃된 working tree 루트를 반환하므로
# worktree 에서 실행하면 worktree 경로, 메인 레포에서 실행하면 메인 레포 경로를 반환.
#
# Output (stdout):  절대경로 (예: /z/myrepo/.claude/worktrees/sleepy-cannon-a73572)
# Exit code: 0=성공, 1=git 레포 아님

TOPLEVEL=$(git rev-parse --show-toplevel 2>/dev/null) || {
  echo "ERROR: not a git repository" >&2
  exit 1
}

echo "$TOPLEVEL"
