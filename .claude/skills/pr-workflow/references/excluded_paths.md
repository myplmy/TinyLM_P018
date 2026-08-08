# 영구 제외 경로 (PR scope 밖)

이 경로들은 **PR에 포함되지 않는다**. staging 단계에서 필터링 필수.

## 단일 소스: `.claude/project.json` 의 `excludedPaths`

제외 경로 목록은 **`.claude/project.json`의 `excludedPaths` 배열이 단일 소스**다. 스크립트·SKILL·본 문서 어디에도 하드코딩하지 않는다.

```jsonc
// .claude/project.json
{
  "excludedPaths": [
    ".claude/settings.local.json",   // 내장 기본값
    ".env",                          // 내장 기본값
    // ↓ 프로젝트별 추가 (예시)
    "data/raw/",                     // 대용량 원본 데이터
    "build/",                        // 재생성 산출물
    "*.db"                           // 로컬 DB 바이너리 (또는 .gitignore 로)
  ]
}
```

내장 기본값(`safe_stage.sh`에 하드코딩): `.claude/settings.local.json`, `.env`. 그 외 프로젝트 고유 경로는 위 배열에 추가한다.

**전형적 제외 대상 유형** (프로젝트가 있으면 추가):
- 개인 로컬 설정 (`.claude/settings.local.json` 등)
- 비밀키·토큰 (`.env`) — 절대 커밋 금지
- 대용량 원본 데이터·참고 자료
- 세션마다 재생성되는 산출물 (리포트·로그)
- 로컬 DB 바이너리

## prefix 매칭 규칙

위 경로는 **prefix 매칭**으로 필터링:
- `.env` → 정확 경로 제외
- `data/raw/` → `data/raw/` 하위 모두 제외
- `data/raw_backup/` → prefix 다름, `data/raw/`로는 제외되지 않음 (별도 등재 필요)

> **`.claude/skills/`는 제외 대상 아님.** 리포에 특화된 워크플로우 스킬이므로 정상 tracked. 개인 환경 설정인 `settings.local.json`만 제외.

## 스크립트에 주입하는 법

`safe_stage.sh`는 env `SAFE_STAGE_EXCLUDES`(개행 구분)로 프로젝트 제외 경로를 받는다:

```bash
# jq 가 있으면
SAFE_STAGE_EXCLUDES=$(jq -r '.excludedPaths[]' .claude/project.json) \
  bash .claude/skills/pr-workflow/scripts/safe_stage.sh --add <paths>

# jq 가 없으면 claude 가 project.json 을 읽어 값을 개행 구분으로 직접 주입
```

## 특수 케이스

### 제외 경로에 의도적 추가가 필요할 때
사용자가 명시적으로 "제외 경로 내부 X 파일 추가"를 요청하면 예외 허용 가능. 단:
1. 추가 사유를 사용자에게 재확인
2. `.gitignore` 수정 여부 판단 (경로가 `.gitignore`에 있으면 add 자체가 실패)
3. PR body에 예외 사유 명기

### 새 제외 경로 발견 시
프로젝트에서 새 임시·개인 디렉토리가 생기면 **`.claude/project.json`의 `excludedPaths`에만** 추가하면 된다 (스크립트 수정 불필요).

## 검증

staging 후 확인:
```bash
git diff --cached --name-only
```
제외 경로가 포함됐으면: `git reset HEAD <path>`
