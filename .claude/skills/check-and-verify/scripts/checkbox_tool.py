#!/usr/bin/env python3
"""check-and-verify 스킬 번들 — PR/Issue body 체크박스 파싱·분류·갱신.

네트워크 호출 없음. 순수 텍스트 변환만 수행한다 — `gh` 호출(body 조회·PATCH)은
SKILL.md 절차가 담당한다. 이 분리로 스킬은 결정론적이고 재사용 가능한 텍스트
처리 단계를 매 호출 재작성 없이 갖는다.

서브커맨드:
  classify <body_file>
      체크박스를 추출·분류하여 JSON 배열을 stdout 으로 출력.
      각 원소: {index, lineno, checked, text, category, command}
      category ∈ {pytest, script, script+env, file_check, manual}

  apply <body_file> <idx,idx,...>
      지정한 체크박스 인덱스를 `[ ]` → `[x]` 로 바꾼 body 전문을 stdout 출력.
      이미 체크된 항목·범위 밖 인덱스는 무시. 다른 줄은 원형 보존.

분류는 휴리스틱이다 — 오분류 가능성이 있으므로 SKILL.md 는 항상 manual 강등·
사용자 확인 경로를 제공한다.
"""
import json
import re
import sys

# 체크박스 라인: 선행 공백 + 불릿(-,*,+) + [ ] / [x]
#   group1 = 닫는 대괄호 직전까지의 prefix, group2 = 상태문자, group3 = `]` 이후 전체
CHECKBOX_RE = re.compile(r'^(\s*[-*+] \[)([ xX])(\].*)$')
# 인라인 코드 스팬(백틱)
CODE_RE = re.compile(r'`([^`]+)`')
# 명령 앞 환경변수 prefix (NAME=value ...)
ENV_PREFIX_RE = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*=\S+\s+)+')


def classify_command(span):
    """코드 스팬 문자열 → (category, full_command) 또는 (None, None).

    실행 가능한 명령(pytest / 프로젝트 스크립트)만 category 를 부여한다.
    env prefix(LLM_MODE=... 등)가 있으면 'script+env' 로 표기하되 명령 전문은
    prefix 를 포함해 그대로 반환한다 — Bash 가 한 줄로 실행하면 되기 때문이다.
    """
    c = span.strip()
    body = ENV_PREFIX_RE.sub('', c)
    has_env = body != c
    if re.match(r'(python3?\s+-m\s+pytest|pytest)\b', body):
        return ('script+env' if has_env else 'pytest', c)
    if re.match(r'python3?\s+(test|scripts)/', body):
        return ('script+env' if has_env else 'script', c)
    return (None, None)


def classify_item(text):
    """체크박스 텍스트 → {'category', 'command'}."""
    for span in CODE_RE.findall(text):
        cat, cmd = classify_command(span)
        if cat:
            return {'category': cat, 'command': cmd}
    # 파일 존재 확인 휴리스틱: 존재/생성/저장/출력 키워드 + 경로형 코드 스팬
    if re.search(r'(존재|생성|저장|출력|파일)', text):
        for span in CODE_RE.findall(text):
            s = span.strip()
            if '/' in s and ' ' not in s and not s.startswith(('python', 'pytest')):
                return {'category': 'file_check', 'command': s}
    return {'category': 'manual', 'command': None}


def parse_checkboxes(raw):
    """body 텍스트 → 체크박스 dict 리스트."""
    items = []
    idx = 0
    for lineno, line in enumerate(raw.splitlines()):
        m = CHECKBOX_RE.match(line)
        if not m:
            continue
        info = classify_item(m.group(3)[1:].strip())
        items.append({
            'index': idx,
            'lineno': lineno,
            'checked': m.group(2).lower() == 'x',
            'text': m.group(3)[1:].strip(),
            'category': info['category'],
            'command': info['command'],
        })
        idx += 1
    return items


def cmd_classify(body_file):
    with open(body_file, encoding='utf-8') as f:
        items = parse_checkboxes(f.read())
    json.dump(items, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write('\n')


def cmd_apply(body_file, indices_csv):
    targets = {int(x) for x in indices_csv.split(',') if x.strip()}
    with open(body_file, encoding='utf-8') as f:
        raw = f.read()
    trailing_nl = raw.endswith('\n')
    out = []
    idx = 0
    for line in raw.splitlines():
        m = CHECKBOX_RE.match(line)
        if m:
            if idx in targets and m.group(2).lower() != 'x':
                line = m.group(1) + 'x' + m.group(3)
            idx += 1
        out.append(line)
    sys.stdout.write('\n'.join(out) + ('\n' if trailing_nl else ''))


def main():
    # 리다이렉트 출력 시 Windows locale 인코딩(cp949)으로 한글이 손상되지 않도록
    # stdout 을 UTF-8 로 고정한다. 콘솔 표시(mojibake)는 무관 — 파일 출력이 핵심.
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    if len(sys.argv) < 3:
        sys.exit('usage: checkbox_tool.py {classify|apply} <body_file> [indices]')
    sub = sys.argv[1]
    if sub == 'classify':
        cmd_classify(sys.argv[2])
    elif sub == 'apply':
        if len(sys.argv) < 4:
            sys.exit('apply requires <indices> (예: 0,2,3)')
        cmd_apply(sys.argv[2], sys.argv[3])
    else:
        sys.exit(f'unknown subcommand: {sub}')


if __name__ == '__main__':
    main()
