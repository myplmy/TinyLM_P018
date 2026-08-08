#!/usr/bin/env python3
"""정적 속성 검사 — `cfg.<attr>` 이 실제 `TMTConfig` 에 있는지 **torch 없이** 확인한다.

★왜 만들었나 (P029 사고)
  `scripts/probe_prompts.py` 초판이 `cfg.seq_len` 을 썼다. 실제 필드는 `max_seq_len` 이다.
  파이썬은 실행 시점까지 이걸 잡아주지 않고, 그 줄은 **모델을 GPU 에 로드한 뒤**에야 실행되므로
  사용자가 배치를 돌린 다음에야 터졌다.

  더 근본 원인은 **샘플링 루프를 복제한 것**이었다(`generate.py` 에 이미 올바른 구현이 있었다).
  그건 `generate.sample()` 로 합쳐 해결했고, 이 스크립트는 **같은 부류의 오타를 사전에** 잡는다.

  AST 만 쓰므로 **torch·GPU 불필요** = 어디서든 수 초에 돌릴 수 있다.

검사
  1. `cfg.X` / `config.X` / `self.cfg.X` 형태의 속성 접근을 모아 TMTConfig 필드와 대조
  2. `tinylm.*` 에서 import 하는 이름이 실제로 그 모듈에 정의돼 있는지(간이)

★한계
  - 정적 분석이라 동적 접근(`getattr(cfg, name)`)은 못 본다.
  - 필드가 **존재**하는지만 본다. 값이 맞는지, 로직이 옳은지는 전혀 모른다.
  - 이걸 통과해도 실행은 실패할 수 있다. **스모크(run_smoke.bat)의 대체가 아니다.**

사용법
  python scripts/check_attrs.py
"""
from __future__ import annotations
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = sorted(list((ROOT / "scripts").glob("*.py")) + list((ROOT / "tinylm").rglob("*.py")))


def config_fields():
    """config.py 를 AST 로 읽어 TMTConfig 의 필드명을 모은다(torch import 없이)."""
    src = (ROOT / "tinylm" / "config.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "TMTConfig":
            out = set()
            for st in node.body:
                if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                    out.add(st.target.id)
                elif isinstance(st, ast.Assign):
                    for t in st.targets:
                        if isinstance(t, ast.Name):
                            out.add(t.id)
                elif isinstance(st, ast.FunctionDef):
                    out.add(st.name)
            return out
    return set()


# 런타임에 붙는 속성(코드에서 cfg.X = ... 로 설정) — 선언 필드가 아니어도 정상
RUNTIME_OK = set()


def collect_runtime_assigned():
    """`cfg.X = ...` 처럼 코드가 나중에 붙이는 속성을 수집(정상으로 취급)."""
    found = set()
    for f in TARGETS:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                            and t.value.id in ("cfg", "config")):
                        found.add(t.attr)
    return found


def main():
    fields = config_fields()
    if not fields:
        print("[!] TMTConfig 를 파싱하지 못했습니다.")
        return 1
    runtime = collect_runtime_assigned()
    known = fields | runtime | RUNTIME_OK
    print(f"TMTConfig 선언 필드 {len(fields)}개 + 런타임 부착 {len(runtime)}개 = 허용 {len(known)}개\n")

    bad = 0
    for f in TARGETS:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[!] {f.relative_to(ROOT)} 파싱 실패: {e}")
            continue
        hits = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id in ("cfg", "config")):
                if node.attr not in known and not node.attr.startswith("_"):
                    hits.append((getattr(node, "lineno", 0), node.attr))
        if hits:
            bad += len(hits)
            print(f"  [E] {f.relative_to(ROOT)}")
            for ln, at in sorted(set(hits)):
                near = [k for k in sorted(known)
                        if k.startswith(at[:4]) or at in k or k in at]
                hint = f"   (혹시 이것? {', '.join(near[:3])})" if near else ""
                print(f"        L{ln}: cfg.{at} — TMTConfig 에 없음{hint}")

    print(f"\n총 {bad}건")
    print("주의: 정적 검사라 getattr() 같은 동적 접근은 못 본다. 필드 '존재'만 보고 값·로직은 모른다.")
    print("      스모크(run_smoke.bat)의 대체가 아니라 그 앞단의 값싼 그물이다.")
    return bad


if __name__ == "__main__":
    sys.exit(main())
