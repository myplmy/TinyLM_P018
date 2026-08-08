"""memmap 무작위 크롭 로더. 시드가 같으면 두 런이 정확히 같은 배치를 본다."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class Loader:
    def __init__(self, split, bs, seq, device, cache_dir, seed=1234):
        self.d = np.memmap(Path(cache_dir) / f"{split}.bin", dtype=np.uint16, mode="r")
        self.bs, self.seq, self.device = bs, seq, device
        self.rng = np.random.default_rng(seed)

    def __call__(self):
        ix = self.rng.integers(0, len(self.d) - self.seq - 1, self.bs)
        # ③ 파이썬 루프 대신 NumPy advanced indexing 으로 한 번에 슬라이싱
        x = self.d[ix[:, None] + np.arange(self.seq + 1)].astype(np.int64)
        t = torch.from_numpy(x)
        if self.device == "cuda":
            t = t.pin_memory().to("cuda", non_blocking=True)
        return t[:, :-1], t[:, 1:]
