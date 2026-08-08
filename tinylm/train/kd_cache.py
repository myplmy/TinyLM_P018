"""오프라인 KD 캐싱 — 교사 top-k 로짓을 1회 저장 → 학생이 교사 forward 없이 KD.

정합성: 학습 loader(seed 1234)를 **동일 순서로 replay** 하므로 크롭이 정확히 일치한다.
따라서 캐시는 (data, steps, micro_bs, seq, accum, seed) 에 종속된다 — 이 중 하나라도 다르면 무효.

주의(근사): 온라인 KD는 전체 vocab KL, 캐시는 **top-k KL 근사** → 온라인과 미세 차이(비교 시 유의).
디스크: total_pos x topk. 예) 300M x top16 ≈ vals 9.6GB + idx 19GB.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .. import paths
from .init_utils import load_dense
from ..data import prepare, Loader


def cache_dir(base, topk):
    return paths.RUNS / "kdcache" / f"{base}_top{topk}"


@torch.no_grad()
def build_kd_cache(base, teacher_path, data, n_tokens, steps, micro_bs, seq, accum,
                   topk=16, temp=2.0, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    meta = prepare(data, n_tokens)
    teacher, _ = load_dense(teacher_path, device)
    teacher.eval(); teacher.set_anneal(1.0)
    tr = Loader("train", micro_bs, seq, device, meta["dir"], seed=1234)

    d = cache_dir(base, topk); d.mkdir(parents=True, exist_ok=True)
    total = steps * accum * micro_bs * seq
    vals = np.memmap(d / "vals.f16", dtype=np.float16, mode="w+", shape=(total, topk))
    idx = np.memmap(d / "idx.i32", dtype=np.int32, mode="w+", shape=(total, topk))
    print(f"[kdcache] {total/1e6:.0f}M pos x top{topk} 캐싱 -> {d}  (교사 {Path(teacher_path).name})")

    pos = 0
    import time; t0 = time.time()
    for s in range(steps):
        for _ in range(accum):
            x, _y = tr()
            import torch.nn.functional as F
            with torch.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
                logits = teacher(x)                            # [mb, seq, V]
            # 교사 확률 p=softmax(logits/T) 의 top-k(확률+인덱스)를 저장 (loss는 top-k KL)
            p = F.softmax(logits.float() / temp, dim=-1)       # [mb, seq, V]
            pv, ti = torch.topk(p, topk, dim=-1)               # [mb, seq, K]
            n = x.shape[0] * x.shape[1]
            vals[pos:pos + n] = pv.reshape(n, topk).cpu().numpy().astype(np.float16)
            idx[pos:pos + n] = ti.reshape(n, topk).cpu().numpy().astype(np.int32)
            pos += n
        if s % 50 == 0:
            el = time.time() - t0
            print(f"  kdcache step {s}/{steps}  ({el/(s+1)*1000:.0f} ms/step)")
    vals.flush(); idx.flush()
    (d / "meta.json").write_text(json.dumps(
        {"base": base, "data": data, "steps": steps, "micro_bs": micro_bs,
         "seq": seq, "accum": accum, "topk": topk, "total": total, "seed": 1234,
         "temp": temp}, indent=2))
    print(f"[kdcache] 완료: {pos} pos 저장")


class KdCacheReader:
    """순서대로 micro_bs*seq 블록을 반환. 학습 loader와 lockstep으로 소비해야 정합."""

    def __init__(self, base, topk, micro_bs, seq):
        d = cache_dir(base, topk)
        m = json.loads((d / "meta.json").read_text())
        self.topk = topk
        self.n = micro_bs * seq
        total = m["total"]
        self.vals = np.memmap(d / "vals.f16", dtype=np.float16, mode="r").reshape(total, topk)
        self.idx = np.memmap(d / "idx.i32", dtype=np.int32, mode="r").reshape(total, topk)
        self.pos = 0
        # 정합성 경고
        if m["micro_bs"] != micro_bs or m["seq"] != seq:
            print(f"[kdcache] 경고: 캐시(mb={m['micro_bs']},seq={m['seq']})와 실행 설정이 다름 → 정합 깨짐")

    def seek_step(self, step, accum):
        self.pos = step * accum * self.n

    def next(self, device):
        v = torch.from_numpy(np.ascontiguousarray(self.vals[self.pos:self.pos + self.n]).astype(np.float32)).to(device)
        i = torch.from_numpy(np.ascontiguousarray(self.idx[self.pos:self.pos + self.n]).astype(np.int64)).to(device)
        self.pos += self.n
        return v, i    # [n, K]


def kd_cache_loss(logits, p_t, ti, vocab, temp):
    """온라인 KL(p||q)=sum_V p(log p - log q) 의 top-k 제한형.
    p_t [n,K]=교사 top-k 확률(softmax(t/T) 저장), q는 학생 softmax(s/T) 를 같은 인덱스에서.
    교사 질량이 top-k에 집중되므로 전체 KL을 근사한다."""
    import torch.nn.functional as F
    q_logp = F.log_softmax(logits.reshape(-1, vocab) / temp, dim=-1)   # [n, V]
    q_topk = q_logp.gather(1, ti)                                       # [n, K]
    kl = (p_t * (p_t.clamp_min(1e-9).log() - q_topk)).sum(-1).mean()
    return kl * (temp * temp)
