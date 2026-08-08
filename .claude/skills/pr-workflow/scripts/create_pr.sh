#!/bin/bash
# create_pr.sh — PR 생성 헬퍼 (base 자동 감지, 프로젝트 중립)
#
# Base 결정:
#  - --base 명시 시 그대로 사용
#  - 미지정 시 detect_base.sh 호출 (project.json → 메타 → env → 휴리스틱)
#  - 감지 결과가 LOW/NONE 이면 caller (claude) 가 사용자에게 문의
#
# 사용법:
#   ./create_pr.sh --title "..." --body-file body.md
#   ./create_pr.sh --title "..." --body-file body.md --base main      # 명시 override
#
# 옵션:
#   --title TEXT         PR 제목 (필수)
#   --body TEXT          PR 본문 inline
#   --body-file PATH     PR 본문 파일 경로
#   --head BRANCH        head branch (기본: 현재 branch)
#   --base BRANCH        base branch (기본: detect_base.sh 결과)
#   --draft              draft PR 생성

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# gh 경로 자동 해결
resolve_gh() {
  if command -v gh >/dev/null 2>&1; then
    echo "gh"; return
  fi
  if [[ -x "/c/Program Files/GitHub CLI/gh.exe" ]]; then
    echo "/c/Program Files/GitHub CLI/gh.exe"; return
  fi
  echo "ERROR: gh CLI not found. Install from https://cli.github.com" >&2
  exit 1
}

GH="$(resolve_gh)"

TITLE=""
BODY=""
BODY_FILE=""
HEAD=""
BASE=""
DRAFT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --title) TITLE="$2"; shift 2 ;;
    --body) BODY="$2"; shift 2 ;;
    --body-file) BODY_FILE="$2"; shift 2 ;;
    --head) HEAD="$2"; shift 2 ;;
    --base) BASE="$2"; shift 2 ;;
    --draft) DRAFT="--draft"; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$TITLE" ]]; then
  echo "ERROR: --title is required" >&2
  exit 1
fi

# Head branch 기본값: 현재
if [[ -z "$HEAD" ]]; then
  HEAD="$(git branch --show-current)"
fi

# Base branch — 명시 없으면 detect_base.sh 호출
if [[ -z "$BASE" ]]; then
  BASE_RAW=$(bash "$SCRIPT_DIR/detect_base.sh" 2>/tmp/cp_db.err) || EC=$? && EC=${EC:-0}
  if [[ $EC -eq 2 || "$BASE_RAW" == "AMBIGUOUS" ]]; then
    echo "ERROR: base branch detection ambiguous. confidence below MEDIUM." >&2
    echo "  stderr from detect_base.sh:" >&2
    cat /tmp/cp_db.err >&2
    echo "" >&2
    echo "  → caller (claude) should ask user, then re-invoke with --base <chosen>" >&2
    echo "    or run: bash $SCRIPT_DIR/detect_base.sh --write <chosen>" >&2
    exit 2
  fi
  BASE="$BASE_RAW"
fi

# Body 수집
if [[ -n "$BODY_FILE" ]]; then
  BODY="$(cat "$BODY_FILE")"
elif [[ -z "$BODY" ]] && [[ ! -t 0 ]]; then
  BODY="$(cat)"
fi

if [[ -z "$BODY" ]]; then
  echo "ERROR: PR body가 비어있습니다. --body, --body-file 또는 stdin 사용." >&2
  exit 1
fi

echo "=== PR 생성 정보 ==="
echo "Base: $BASE"
echo "Head: $HEAD"
echo "Title: $TITLE"
echo ""
echo ">>> PR 생성 실행..."

"$GH" pr create \
  --base "$BASE" \
  --head "$HEAD" \
  --title "$TITLE" \
  --body "$BODY" \
  $DRAFT
