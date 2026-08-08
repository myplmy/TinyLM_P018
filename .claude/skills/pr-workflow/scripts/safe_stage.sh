#!/bin/bash
# safe_stage.sh — 안전 staging 도우미 (프로젝트 중립)
#
# 영구 제외 경로(prefix 매칭)를 필터링하고, 남은 파일 목록을 사용자에게
# 제시한 뒤 명시적으로 `git add` 한다.
#
# 제외 경로 소스 (우선순위):
#   1. 내장 기본값 (.claude/settings.local.json, .env)
#   2. env SAFE_STAGE_EXCLUDES — 개행 구분 문자열 (기본값에 추가됨)
#      → caller(claude)가 .claude/project.json 의 excludedPaths 를 읽어 주입한다:
#        SAFE_STAGE_EXCLUDES=$(jq -r '.excludedPaths[]' .claude/project.json 2>/dev/null) \
#          bash safe_stage.sh --add ...
#      (jq 미설치 시 claude 가 값을 직접 개행 구분으로 넘겨도 된다)
#
# 사용법:
#   ./safe_stage.sh                   # 전체 변경 파일을 필터링 후 출력만 (add 안 함)
#   ./safe_stage.sh --add             # 필터링 후 git add 실행
#   ./safe_stage.sh --add path1 path2 # 지정 경로만 필터 검증 후 add
#
# 반환: 0 = 성공, 1 = 스테이징할 파일 없음 또는 전원 제외됨, 2 = 제외 경로만 선택됨

set -euo pipefail

# 영구 제외 경로 (prefix 매칭) — 내장 기본값
EXCLUDE_PATTERNS=(
  ".claude/settings.local.json"
  ".env"
)

# env 로 주입된 프로젝트별 제외 경로를 추가 (개행 구분)
if [[ -n "${SAFE_STAGE_EXCLUDES:-}" ]]; then
  while IFS= read -r line; do
    [[ -n "$line" ]] && EXCLUDE_PATTERNS+=("$line")
  done <<< "$SAFE_STAGE_EXCLUDES"
fi

is_excluded() {
  local path="$1"
  for pattern in "${EXCLUDE_PATTERNS[@]}"; do
    if [[ "$path" == "$pattern"* ]]; then
      return 0
    fi
  done
  return 1
}

# git status 기반 변경 파일 수집 (untracked 포함)
collect_changes() {
  git status --porcelain | awk '{
    # 첫 두 문자는 상태 코드, 나머지는 경로
    status = substr($0, 1, 2)
    path = substr($0, 4)
    # rename 표기 "R  old -> new" 처리: new 쪽만 사용
    if (index(path, " -> ") > 0) {
      split(path, a, " -> ")
      path = a[2]
    }
    # 따옴표 제거 (공백 포함 경로 대응)
    gsub(/^"/, "", path)
    gsub(/"$/, "", path)
    print path
  }'
}

main() {
  local mode="list"
  local explicit_paths=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --add)
        mode="add"
        shift
        ;;
      *)
        explicit_paths+=("$1")
        shift
        ;;
    esac
  done

  local all_paths=()
  if [[ ${#explicit_paths[@]} -gt 0 ]]; then
    all_paths=("${explicit_paths[@]}")
  else
    while IFS= read -r line; do
      [[ -n "$line" ]] && all_paths+=("$line")
    done < <(collect_changes)
  fi

  if [[ ${#all_paths[@]} -eq 0 ]]; then
    echo "No changes detected." >&2
    exit 1
  fi

  local included=()
  local excluded=()
  for path in "${all_paths[@]}"; do
    if is_excluded "$path"; then
      excluded+=("$path")
    else
      included+=("$path")
    fi
  done

  echo "=== 영구 제외 경로 (staging 대상 아님) ==="
  if [[ ${#excluded[@]} -eq 0 ]]; then
    echo "  (none)"
  else
    printf "  %s\n" "${excluded[@]}"
  fi

  echo ""
  echo "=== Staging 대상 (${#included[@]} 파일) ==="
  if [[ ${#included[@]} -eq 0 ]]; then
    echo "  (none — 모두 제외 경로)"
    exit 2
  else
    printf "  %s\n" "${included[@]}"
  fi

  if [[ "$mode" == "add" ]]; then
    echo ""
    echo ">>> git add 실행 중..."
    git add "${included[@]}"
    echo "완료. 현재 staging 상태:"
    git status --short
  else
    echo ""
    echo "(list mode — --add 옵션 없이는 staging하지 않음)"
  fi
}

main "$@"
