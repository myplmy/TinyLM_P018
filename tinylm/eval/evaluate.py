"""배포 상태(완전 삼진)에서의 val loss/ppl."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate(model, loader, iters, device, bytes_per_token=None):
    was = model.cfg.quant_anneal
    model.set_anneal(1.0)                      # 완전 삼진 = 배포 상태 (compile 안전)
    # ★freeze: eval 루프 동안 가중치가 고정이므로 삼진화를 1회만 한다.
    #   forward() 는 원래 매 호출 refresh_quant() 를 부른다(학습 중엔 필수). eval 에서는
    #   그게 iters 배의 순수 낭비였다(결과 014). 아래 clear_quant() 가 플래그를 되돌린다.
    model.eval(); model.freeze_quant()
    tot = 0.0
    for _ in range(iters):
        x, y = loader()
        with torch.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            logits = model(x)
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               y.reshape(-1)).float().item()
    model.clear_quant(); model.train(); model.set_anneal(was)
    loss = tot / iters
    out = {"val_loss": loss, "ppl": math.exp(min(loss, 20))}
    if bytes_per_token:
        out["bpb"] = loss / math.log(2) / bytes_per_token
    return out
