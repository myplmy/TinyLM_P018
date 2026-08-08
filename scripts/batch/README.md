# `scripts/batch/` — 실험이 아니라 **기능 모듈**

여기 있는 `.bat` 은 **도구**다. 실험이 아니다. 실험 번호도, 계획서 참조도, `test_result` 상의
자리도 없다. **사용자에게 "실험 X 를 하려면 이 파일을 실행하세요" 라고 안내하지 않는다.**

## 왜 이렇게 바꿨나 (2026-07-31)

이전 이름은 `run_P034_memory.bat` 처럼 **계획번호가 도구에 용접**돼 있었다. 그래서 같은 도구를
다른 계획에 재사용하려면 ① 로그 이름에 거짓 계획번호를 남기거나 ② 파일을 복사하는 수밖에
없었고, **둘 다 실제로 일어났다**. 이제 **도구는 기능만**, **호출자가 정체성**을 갖는다.

또한 이전 파일들은 정상 종료 후 `:BADROOT` 블록으로 **흘러들어가** 성공한 실행 끝에
`[STOP] could not locate the repo root` 를 찍었다(2026-07-31 사용자 보고). 본문 끝에
`exit /b 0` 이 없었기 때문이다. 재작성본은 **모든 경로가 명시적으로 `exit /b`** 한다.

## 실험은 이렇게 만든다

작업폴더 **최상위**에 `run_P0NN_stageM_요약.bat` 을 새로 만들고, 그 안에서 도구를 `call` 한다.
그래야 로그·재현 명령·결과문서가 **하나의 실험 단계**로 묶인다.

```bat
setlocal enabledelayedexpansion
set TL_LOGNAME=P034-stage2
set TL_NOPAUSE=1
set TL_EXTRA=--drop-latent
call scripts\batch\tool_mem_profile.bat
if errorlevel 1 echo [WARN] mem profile reported a problem - continuing
```

## 호출 규약

인자는 **환경변수**로 준다. 이 저장소는 `.bat` 에서 `%` 를 금지하므로(`lint_bat.py` E3)
`%1`·`%*`·`%~dp0` 은 물론 **`for` 반복변수도 쓸 수 없다**. 그래서 목록형 입력은
`TL_DATA1`/`TL_DATA2` 처럼 **번호 슬롯**으로 받는다. 도구 내부는
`setlocal enabledelayedexpansion` + `!TL_X!` 로 읽는다(`!` 는 금지문자가 아니다).

공통 변수

| 변수 | 뜻 |
|---|---|
| `TL_LOGNAME` | `runlog.py --name` 에 들어갈 이름. **실험 단계 이름을 여기 넣는다** |
| `TL_NOPAUSE` | 정의돼 있으면 `pause` 를 건너뛴다. **호출자는 반드시 설정** |
| `TL_MODELS` | 체크포인트 태그 목록(공백 구분) |
| `TL_EXTRA` | 해당 파이썬 스크립트로 그대로 흘려보낼 추가 플래그 |

각 도구의 고유 변수는 파일 상단 `CONTRACT` 블록에 있다.

## 로그에 단계 제목을 남기는 법 (`--note`)

`runlog.py` 는 **감싼 파이썬 명령의 출력만** 기록한다. 배치의 `echo` 는 콘솔에만 나오고
로그파일에는 **안 남는다** — 그래서 콘솔과 파일이 달라 보인다(2026-07-31 사용자 보고).

단계 제목은 이렇게 **양쪽에** 남긴다.

```bat
echo [1/2] baseline
python scripts\runlog.py --name !TL_LOGNAME! --note "[1/2] baseline"
python scripts\runlog.py --name !TL_LOGNAME! -- python scripts\mem_runtime.py --device cpu
```

`--note` 는 명령을 실행하지 않고 텍스트만 쓴다(여러 개 주면 여러 줄). **errorlevel 을 건드리지
않도록 실제 명령 앞에 둔다.** 현재 모든 배치가 `[N/M]`·`[run]`·`[pre]`·`[guard]` 표지에 대해
자동으로 이 줄을 갖고 있다.

## 도구 목록

| 파일 | 기능 | 부르는 스크립트 |
|---|---|---|
| `tool_smoke.bat` | 계측 계약 스모크(tiny·synthetic). **새 스크립트 쓰기 전 필수** | `check_attrs` · `run100m train` · `check_smoke` |
| `tool_kvcache_gate.bat` | **KV 캐시 정확성 정본 게이트**(teacher-forced 로짓, fp32) | `diag_kvcache.py` |
| `tool_mem_profile.bat` | 저장(packed) vs 상주(runtime) 메모리 | `mem_runtime.py` |
| `tool_sparse34_audit.bat` | 3:4 구현·실제 희소율·bpw 회계 감사 | `diag_sparse34.py` |
| `tool_datacache_diag.bat` | 토큰 캐시 진단(누설·bytes/token) | `diag_cache.py` |
| `tool_eval_slices.bat` | train 앞/뒤 vs val 구간별 손실 | `eval_slices.py` |
| `tool_valdocs.bat` | 문서단위 val 손실 분해(+`TL_INSPECT` 로 원문 덤프) | `diag_val_docs.py` |
| `tool_qual_probe.bat` | 정성 생성 프로브 | `probe_prompts.py` |

## 종료코드

`0` 정상 / `1` 게이트 실패(정본 게이트만) / `3` 필요한 스크립트 미구현 / `9` 저장소 루트를 못 찾음.

## 규칙

- 도구에 **실험 조건을 하드코딩하지 않는다.** 기본값은 "가장 흔한 값"이지 "그 실험의 값"이 아니다.
- 도구를 고치면 `python scripts/lint_bat.py --fix` 를 돌린다(CRLF·비ASCII·금지문자).
- 새 도구를 추가하면 이 표와 `CLAUDE.md` 의 배치 표를 함께 갱신한다.
