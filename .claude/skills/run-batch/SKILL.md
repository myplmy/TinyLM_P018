---
name: run-batch
description: Windows 실행 배치파일(`run100m_*.bat`, `run_*.bat`)을 작성·수정·검증하는 스킬. "배치파일 만들어", "bat 작성", "실험용 배치", "이 배치 고쳐", "배치 점검" 같은 요청에 사용한다. 또한 사용자가 명시하지 않더라도 여러 학습 명령을 순서대로 실행할 수단을 만들거나 기존 .bat 을 손보려 하면 이 스킬을 적용한다. 순수 ASCII 강제, `<>|%`·chcp 금지, errorlevel 정책(선행 의존만 중단), 태그 위생, 판정 안내문 포함을 검사하고 `scripts/lint_bat.py` 로 기계 검증한다. 인코딩 사고와 "한 런 실패로 배치 전체 중단"을 차단한다.
---

# run-batch

## 왜 이 스킬이 필요한가

`.bat` 은 이 프로젝트에서 GPU 시간을 태우는 유일한 실행 경로인데, 조용히 실패하는 방식이 많다.

- **인코딩**: 한글이 들어가면 cmd 기본 코드페이지에서 깨져 명령이 망가진다. `chcp` 로 덮으면
  파이썬 출력이 또 깨진다. → **순수 ASCII 만.**
- **리다이렉션**: `REM ... -> foo` 처럼 `>` 를 쓰면 cmd 가 리다이렉션으로 파싱할 수 있어
  엉뚱한 파일이 생긴다. `echo` 에서는 `^>` `^<` `^|` 로 escape 해야 한다.
- **`goto ERROR`**: 결과 007에서 [3/4] OOM 이 [4/4]를 죽여 사용자가 수동 재실행해야 했다.
- **태그 누락**: 태그를 안 붙이면 정본 `dense`/`tied` 로그·체크포인트를 **조용히 덮어쓴다**.

## 규칙

### 1. 인코딩·문자 (위반 시 작성 거부)

- **순수 ASCII.** 주석·echo 전부 영문. 한글 설명이 필요하면 계획서(`test_plan/`)에 쓴다.
- `chcp` 금지.
- `<` `>` `|` `%` 금지. 꼭 필요하면 `echo` 안에서 `^<` `^>` `^|` 로 escape.
  주석에서는 그냥 다른 말로 바꿔라(`->` → `to`, `>1.2x` → `over 1.2x`, `<=` → `at most`).
- 줄 끝은 CRLF 여도 되고 LF 여도 cmd 는 실행한다(굳이 변환하지 않는다).

### 2. errorlevel 정책

```
REM 선행 의존이 있는 단계(prepare, 교사/부모 학습)만 중단
python run100m.py prepare --data ko-en --tokens 600M --exact-cache
if errorlevel 1 goto ERROR

REM 독립 런은 경고만 하고 계속
python run100m.py train ... --tag X
if errorlevel 1 echo [WARN] X failed - continuing
```

판단 기준: **이 런이 실패하면 다음 런이 무의미해지는가?**
- 예 → `goto ERROR` (prepare, 그 예산의 dense/교사)
- 아니오 → `echo [WARN] ... - continuing` (스윕의 각 변형, 벤치의 각 조건)

### 3. 태그 위생

- 스윕·벤치의 모든 런에 `--tag` 를 붙인다. 태그 없으면 파일명이 `..._{arch}` 가 되어 **정본을 덮어쓴다.**
- 새 태그는 `exp-preflight` 로 충돌 확인 후 `docs/EXPERIMENT_BASELINES.md` §2.6 에 등재.
- 교사·부모를 태그된 것에서 가져와야 하면 `--kd-teacher-tag` / `--init-from-tag` 를 잊지 않는다.

### 4. 명령줄 표준

- `--micro-bs 8 --accum 16 --seq 1024`(유효배치 131K)를 **명시**한다. `--accum` 기본은 8 이다.
- 풀은 `--pool-tokens {2배} --exact-cache`.
- `--compile` 은 mode=default. **커널(`--ternary-kernel*`)과 병용 금지**(코드가 SystemExit).
- VRAM: dense 는 `--no-ckpt` 가능(13.5GB). **tied+KD+dense교사에는 붙이지 않는다**(OOM 위험).

### 5. 배치 안에 반드시 들어갈 문서

헤더 주석:
- 이 배치가 검증하는 **가설/목적** 한두 줄
- **왜 이 설정인지**(특히 표준을 벗어난 부분과 그 이유)
- **총 예상 비용**(시간, VRAM)
- errorlevel 정책 한 줄

꼬리 `echo` 블록:
- **무엇을 어떤 순서로 읽어야 하는지**
- **판정 규칙**(어떤 값이면 채택/보류/폐기)
- 알려진 함정 경고(예: sparse34 의 compare 메모리를 믿지 말 것)

### 6. 마지막 골격

```
echo done.
pause
exit /b 0

:ERROR
echo [WARN] stopped: <어떤 선행 단계가 실패했는지>
pause
```

`goto ERROR` 를 하나도 쓰지 않았다면 `:ERROR` 블록은 도달하지 않지만 남겨둬도 무해하다.

## 검증 (작성·수정 후 반드시 실행)

```
python scripts/lint_bat.py                 # 전체
python scripts/lint_bat.py run100m_P026.bat
```

린터가 잡는 것: 비ASCII 바이트, `chcp`, escape 안 된 `<>|%`, `--tag` 없는 train 명령,
`--accum` 누락, 모든 학습 런이 `goto ERROR` 인 경우(스윕 위험), `pause`/`exit /b` 누락.

**린터 통과는 "문법이 안전하다"는 뜻이고 "실험이 옳다"는 뜻이 아니다.**
실험설계는 `exp-preflight` 스킬이 따로 본다.

## ★줄끝(CRLF) — 배치를 고친 뒤 반드시

`.gitattributes` 가 `*.bat text eol=crlf` 로 선언하는데 **이 워크플로의 편집 도구는 전부 LF 로 쓴다.**
그래서 배치를 고치면 선언과 작업트리가 어긋나고, git 이 매번 경고하며 재작성한다.
**이 저장소에서 두 번 재발했다.**

```
python scripts/lint_bat.py --fix     # 줄끝만 CRLF 로 교정(내용 불변)
python scripts/lint_bat.py           # E11 로 검출된다
```

`.bat` 을 LF 로 통일하지 않는 이유: cmd.exe 는 **LF-only 배치에서 `goto`/라벨이 드물게 어긋난다.**
우리 배치는 `goto NOTIMPL`·`goto CACHEBAD` 를 실제로 쓴다 — 조용히 틀릴 위험을 지지 않는다.
