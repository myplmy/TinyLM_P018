#!/bin/bash
# detect_base.sh — 본 worktree 의 PR base branch 결정 (+ confidence level)
#
# 우선순위 + 신뢰도:
#   1. 명시 인자 (--base <branch>)            → confidence=HIGH
#   2. 메타 파일: $GIT_DIR/pr-base             → confidence=HIGH
#   3. 환경변수: PR_BASE_BRANCH                → confidence=HIGH
#      (caller 가 .claude/project.json 의 baseBranch 를 이 env 로 주입)
#   4. 휴리스틱: 첫 commit 포함 local branch (우선순위 매치 단일 후보)
#                                              → confidence=MEDIUM
#   5. 휴리스틱: 여러 후보 모호 / 비매치        → confidence=LOW (출력: AMBIGUOUS)
#   6. 감지 실패 (fallback)                    → confidence=NONE (출력: AMBIGUOUS)
#
# 출력:
#   stdout: <branch> 또는 AMBIGUOUS
#   stderr: confidence=<HIGH|MEDIUM|LOW|NONE> source=<source-tag> [candidates=<list>]
#
# 종료 코드:
#   0 = HIGH or MEDIUM (사용 가능)
#   2 = LOW or NONE (사용자 문의 필요)
#   1 = 오류
#
# Caller 사용 패턴 (SKILL.md §base-detection 참조):
#   base=$(bash detect_base.sh 2>/tmp/db.err) || true
#   conf=$(grep -oE 'confidence=[A-Z]+' /tmp/db.err | cut -d= -f2)
#   if [ "$conf" = "LOW" ] || [ "$conf" = "NONE" ]; then
#       # 사용자에게 AskUserQuestion 으로 문의 (후보 목록 stderr 에 있음)
#       # 결정 후: bash detect_base.sh --write <chosen>
#   fi
#
# Modes:
#   --base <branch>     명시 base (HIGH)
#   --write <branch>    메타 파일에 기록 + stdout 출력 (이후 영구 사용)
#   --verbose           stderr 에 source 정보 추가
#   --candidates        AMBIGUOUS 시 후보 목록을 stderr 에 한 줄씩 출력

set -e

EXPLICIT=""
WRITE_VALUE=""
VERBOSE=""
SHOW_CANDIDATES=""

while [ $# -gt 0 ]; do
  case "$1" in
    --base) EXPLICIT="$2"; shift 2 ;;
    --write) WRITE_VALUE="$2"; shift 2 ;;
    --verbose) VERBOSE=1; shift ;;
    --candidates) SHOW_CANDIDATES=1; VERBOSE=1; shift ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

GIT_DIR=$(git rev-parse --git-dir 2>/dev/null) || {
  echo "FAIL not-a-git-repo" >&2
  exit 1
}
META="$GIT_DIR/pr-base"

emit() {
  # $1=branch, $2=confidence, $3=source-tag, $4=candidates(optional)
  local branch="$1" conf="$2" src="$3" cand="$4"
  echo "confidence=$conf source=$src${cand:+ candidates=$cand}" >&2
  echo "$branch"
  case "$conf" in
    HIGH|MEDIUM) exit 0 ;;
    LOW|NONE)    exit 2 ;;
  esac
}

# --write 모드
if [ -n "$WRITE_VALUE" ]; then
  echo "$WRITE_VALUE" > "$META"
  echo "wrote $META=$WRITE_VALUE" >&2
  emit "$WRITE_VALUE" HIGH meta-write
fi

# 1) 명시 인자
if [ -n "$EXPLICIT" ]; then
  emit "$EXPLICIT" HIGH explicit-arg
fi

# 2) 메타 파일
if [ -f "$META" ]; then
  base=$(head -1 "$META" | tr -d '[:space:]')
  if [ -n "$base" ]; then
    emit "$base" HIGH "meta-file($META)"
  fi
fi

# 3) 환경변수 (project.json 의 baseBranch 를 caller 가 주입)
if [ -n "${PR_BASE_BRANCH:-}" ]; then
  emit "$PR_BASE_BRANCH" HIGH env-PR_BASE_BRANCH
fi

# 4-5) 휴리스틱
HEAD_LOG="$GIT_DIR/logs/HEAD"
first_commit=""
if [ -f "$HEAD_LOG" ]; then
  first_commit=$(head -1 "$HEAD_LOG" 2>/dev/null | awk '{print $2}')
fi

if [ -z "$first_commit" ] || [ "$first_commit" = "0000000000000000000000000000000000000000" ]; then
  emit "AMBIGUOUS" NONE "no-reflog-info"
fi

current=$(git branch --show-current 2>/dev/null || echo "")
containing=$(git branch --contains "$first_commit" --format='%(refname:short)' 2>/dev/null \
             | grep -v "^${current}$" \
             | grep -v "^(HEAD" \
             | sort -u \
             || true)

if [ -z "$containing" ]; then
  emit "AMBIGUOUS" NONE "no-containing-branch"
fi

# 후보 수
count=$(echo "$containing" | wc -l | tr -d ' ')

# 단일 후보 → MEDIUM
if [ "$count" -eq 1 ]; then
  emit "$containing" MEDIUM "heuristic-single"
fi

# 다중 후보 — 우선순위 매치는 MEDIUM (단 1개만 매치 시), 여러 매치는 LOW
# 우선순위 후보: env PR_BASE_PRIORITY (공백 구분) 로 override 가능, 기본 main/master/develop
priority_matches=""
for pri in ${PR_BASE_PRIORITY:-main master develop}; do
  if echo "$containing" | grep -qx "$pri"; then
    priority_matches="$priority_matches $pri"
  fi
done
priority_matches=$(echo "$priority_matches" | xargs)  # trim
pri_count=$(echo "$priority_matches" | wc -w | tr -d ' ')

cand_str=$(echo "$containing" | tr '\n' ',' | sed 's/,$//')

if [ "$pri_count" -eq 1 ]; then
  emit "$priority_matches" MEDIUM "heuristic-priority" "$cand_str"
fi

# 다중 후보 + 우선순위 0개 또는 2개 이상 → LOW (사용자 문의 필요)
if [ -n "$SHOW_CANDIDATES" ]; then
  echo "candidates:" >&2
  echo "$containing" | sed 's/^/  - /' >&2
fi
emit "AMBIGUOUS" LOW "heuristic-multi" "$cand_str"
