"""`cli.py` 가 넘기는 키워드 인자가 **대상 함수에 실제로 있는지** 정적으로 검사한다.

## 왜 필요한가 (2026-08-06 사고)

`--lora-decay` 를 `train()` 에 전달하려고 `cli.py` 를 문자열 치환으로 고쳤는데,
그 문자열이 **유일하지 않았다** — `prepare()` 호출부에도 같은 꼬리가 있었고
`replace(..., 1)` 이 **첫 번째**(prepare)를 바꿨다. 결과:

    TypeError: prepare() got an unexpected keyword argument 'lora_decay'

**그리고 기존 그물이 전부 통과시켰다**:

| 검사 | 왜 못 잡았나 |
|---|---|
| `py_compile` | **문법은 정상**이다 |
| `check_attrs.py` | `cfg.X` 오타만 본다. 함수 호출 시그니처는 안 본다 |
| `check_batch_flags.py` | `--lora-decay` 는 **파서에 있다**. 이름만 보는 도구다 |
| 스모크 | **`train` 만** 돈다. 깨진 것은 `prepare` 였다 |

**전혀 관련 없는 실험(P037 단계4)이 대신 죽었다.** 사용자가 그걸로 발견했다.

## 무엇을 검사하나

`tinylm/cli.py` 안의 `f(...)` 호출 중 **대상 함수를 이 저장소에서 찾을 수 있는 것**만 골라,
넘기는 키워드가 그 함수의 파라미터 목록에 있는지 대조한다.
`**kwargs` 를 받는 함수는 건너뛴다.

**torch 없이 돈다**(AST 만 쓴다) — 학습이 GPU 를 점유한 중에도 돌릴 수 있다.

사용:
    python scripts/check_call_kwargs.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# cli.py 가 부르는 주요 진입 함수 → 정의 파일
TARGETS = {
    "prepare": "tinylm/data/prepare.py",
    "train": "tinylm/train/trainer.py",
    "evaluate": "tinylm/eval/evaluate.py",
    "compare": "tinylm/eval/compare.py",
    "build_kd_cache": "tinylm/train/kd_cache.py",
    "generate": "tinylm/infer/generate.py",
    "load_model": "tinylm/infer/generate.py",
    "lr_find": "tinylm/train/lr_finder.py",
}


def params_of(path: Path, fname: str):
    """(파라미터 이름 집합, **kwargs 를 받는가). 못 찾으면 None."""
    if not path.exists():
        return None
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fname:
            a = node.args
            names = {p.arg for p in (a.posonlyargs + a.args + a.kwonlyargs)}
            return names, (a.kwarg is not None)
    return None


def main():
    cli = ROOT / "tinylm/cli.py"
    tree = ast.parse(cli.read_text(encoding="utf-8"))
    cache, errs, checked = {}, [], 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
        if name not in TARGETS:
            continue
        if name not in cache:
            cache[name] = params_of(ROOT / TARGETS[name], name)
        info = cache[name]
        if info is None:
            errs.append((node.lineno, name, None, f"정의를 못 찾음 ({TARGETS[name]})"))
            continue
        names, has_kwargs = info
        if has_kwargs:
            continue
        checked += 1
        for kw in node.keywords:
            if kw.arg is None:          # **d 전개는 정적으로 못 본다
                continue
            if kw.arg not in names:
                errs.append((node.lineno, name, kw.arg,
                             f"{name}() 에 '{kw.arg}' 파라미터가 없다"))

    print("=" * 78)
    print("  cli.py 호출 키워드 정합성 검사 (torch 불필요)")
    print("=" * 78)
    print(f"  검사한 호출 {checked}건 / 대상 함수 {len(cache)}종")
    for ln, fname, kw, msg in errs:
        print(f"  [E] cli.py:{ln}  {msg}")
    print(f"\n총 에러 {len(errs)}건")
    if errs:
        print("★가장 흔한 원인: **문자열 치환이 유일하지 않은 패턴에 걸렸다.**")
        print("  2026-08-06 에 `doc_min_chars=a.doc_min_chars)` 가 prepare 와 train 양쪽에 있어")
        print("  replace(...,1) 이 prepare 를 고쳤고, 무관한 실험(P037 단계4)이 대신 죽었다.")
    print("주의: `**kwargs` 를 받는 함수와 `**d` 전개는 건너뛴다. 정적 검사의 한계다.")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
