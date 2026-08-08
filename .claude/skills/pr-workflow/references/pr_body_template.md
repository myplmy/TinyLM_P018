# PR Body 표준 템플릿

## 기본 (코드 변경)

```markdown
## Summary

- 핵심 변경 1
- 핵심 변경 2
- 기술적 맥락 (선택)

## Test plan

- [ ] 관련 단위 테스트 통과
- [ ] E2E·통합 검증 (해당 시)
- [ ] 수동 검증 항목 (선택)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

## 문서 전용

```markdown
## Summary

- 문서 N건 신설·패치
- 주요 변경 요약

## Test plan

- [ ] Doc-only change, no code touched
- [ ] 영구 제외 경로(excludedPaths)가 포함되지 않음
- [ ] 내부 링크 렌더링 확인

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

## Doc Impact 섹션 (조건부)

`.claude/project.json`의 `docImpactTargets`에 정본 문서가 등재된 프로젝트에서만 아래 섹션을 추가한다. **배열이 비어 있으면 이 섹션을 넣지 않는다.**

```markdown
## Doc Impact

- [ ] <정본 문서 1> — 영향 없음 / 갱신: §X.Y
- [ ] <정본 문서 2> — 영향 없음 / 갱신: §Y
```

- `docImpactTargets` 예: 스펙 문서, 용어집, 아키텍처 레지스트리 등 프로젝트가 "단일 소스"로 관리하는 문서.
- 영향 있으면 동일 PR에 동기 갱신 커밋 포함 + 갱신 위치 기재. 없으면 "영향 없음".

## 체크리스트 (모든 템플릿 공통)

- [ ] Summary는 bullet 3개 이내가 이상적
- [ ] Test plan은 실제로 수행한 것을 `[x]`로, 예정이면 `[ ]`로
- [ ] base 브랜치는 `detect_base.sh` 결과 사용 (하드코딩 금지)
- [ ] "🤖 Generated with..." 라인 포함

## 커밋 메시지와의 관계

- **Commit subject**: 한 줄 요약 (70자 이내). PR title로도 사용 가능.
- **PR title**: Commit subject와 같거나, 여러 커밋일 때 상위 개념 요약.
- **PR body**: Commit 본문의 확장판. Test plan 추가 필수.
