---
name: grill-me
description: Interview the user relentlessly about a plan or design until reaching shared understanding, resolving each branch of the decision tree. Use when user wants to stress-test a plan, get grilled on their design, or mentions "grill me".
---

Interview me relentlessly about every aspect of this plan until we reach a shared understanding. Walk down each branch of the design tree, resolving dependencies between decisions one-by-one. For each question, provide your recommended answer with rationale.

If a question can be answered by exploring the codebase, explore the codebase instead.

## 질문 진행 규칙 (FoldTheTeam — P109)

- **모든 질문은 `AskUserQuestion` 툴로 수행한다.** 자유 텍스트 문의 금지. 각 질문에 구조화된 옵션·근거·트레이드오프를 동봉하고, 권장안은 (Recommended) 라벨 + 첫 옵션으로 제시.
- **한 차수(round) = 최대 4질문.** `AskUserQuestion` 4-질문 한계는 Claude 시스템 한계치(5개 이상 InputValidationError). 의존 관계가 있는 결정을 **한 차수씩 순차 해소**한다 — "한 번에 하나" 는 "한 차수씩" 으로 해석.
- **질문이 5개 이상이면 차수 분할.** 묶어서 한 번에 전달하지 말고 여러 차수로 나누어 연속 수행. 각 차수 응답 후 결과를 문서·plan 에 inline 반영 → 다음 차수 진입.
- **차수 번호 명시** (예: "차수 1 Q1~Q4", "차수 2 Q5~Q8") 로 추적성 확보. 단일 차수 안에서는 4-질문 제한 엄수, 차수 수에는 제한 없음.
- 1회 grill 마다 코드·문서를 직접 추적(line-level)하여 근거 기반으로 질문·권장안을 구성한다. 자유 추정 금지.
