#!/usr/bin/env python3
"""호환 래퍼 — 실제 구현은 tinylm 패키지에 있다.

  python run100m.py all --data ko-en --tokens 300M --steps 2289 --lr 1e-3 --compile
  == python -m tinylm all ...
"""
from tinylm.cli import main

if __name__ == "__main__":
    main()
