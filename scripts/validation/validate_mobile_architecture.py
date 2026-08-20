#!/usr/bin/env python3
"""P018-R1 MobileLLM 구조 수동 검증기.

기본 실행은 config와 정적 산술만 확인하며 모델을 만들지 않는다. 모델 생성,
forward/backward, GPU/CPU 자원 사용 검사는 반드시 사용자가 ``--confirm-heavy``를
명시해 실행한다. 공식 구현과의 weight parity는 별도
``test_mobile_weight_parity.py`` 및 저장 로그가 담당한다.

사용자 실행 예시 (저장소 루트):

    W:\\miniforge3\\envs\\p018\\python.exe \
      scripts\\validation\\validate_mobile_architecture.py --confirm-heavy --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tinylm.config import build_config


EXPECTED_DENSE_PARAMS = 124_635_456
EXPECTED_TIED_G6_PARAMS = 58_280_256
EXPECTED_GROUPS = [list(range(i, i + 6)) for i in range(0, 30, 6)]


def static_checks():
    dense = build_config("mobile125", "dense", 2048, True)
    tied = build_config("mobile125", "tied", 2048, True)

    expected = {
        "model_family": "mobile",
        "tokenizer_kind": "mobilellm_32k",
        "vocab_size": 32000,
        "dim": 576,
        "ffn_dim": 1536,
        "n_q_heads": 9,
        "n_kv_heads": 3,
        "n_layers": 30,
        "share_embedding": True,
        "tie_word_embeddings": False,
    }
    for name, value in expected.items():
        got = getattr(dense, name)
        assert got == value, f"dense {name}: expected {value!r}, got {got!r}"

    assert not dense.tie_mlp and dense.mlp_group == 1
    assert tied.tie_mlp and tied.mlp_group == 6
    assert tied.n_mlp_groups == 5
    groups = [list(range(i, i + tied.mlp_group)) for i in range(0, tied.n_layers, tied.mlp_group)]
    assert groups == EXPECTED_GROUPS

    per_mlp = 3 * tied.dim * tied.ffn_dim
    tied_params = EXPECTED_DENSE_PARAMS - (tied.n_layers - tied.n_mlp_groups) * per_mlp
    assert tied_params == EXPECTED_TIED_G6_PARAMS

    print("[PASS] config/static arithmetic")
    print(f"  groups       : {groups}")
    print(f"  dense unique : {EXPECTED_DENSE_PARAMS:,}")
    print(f"  tied g6      : {EXPECTED_TIED_G6_PARAMS:,}")
    print(f"  reduction    : {(EXPECTED_DENSE_PARAMS-tied_params)/EXPECTED_DENSE_PARAMS:.2%}")
    return dense, tied


def _parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def _run_one(cfg, device, batch, seq):
    import torch
    import torch.nn.functional as F
    from tinylm.model import build_model

    model = build_model(cfg).to(device)
    expected = EXPECTED_TIED_G6_PARAMS if cfg.tie_mlp else EXPECTED_DENSE_PARAMS
    assert _parameter_count(model) == expected
    assert model.emb.weight is model.lm_head.weight
    assert len({id(layer.attn) for layer in model.layers}) == 30

    if cfg.tie_mlp:
        assert len(model.mlps) == 5
        for group_index, members in enumerate(EXPECTED_GROUPS):
            for layer_index in members:
                assert model.layers[layer_index].mlp is model.mlps[group_index]
        assert len({id(layer.mlp) for layer in model.layers}) == 5
    else:
        assert len({id(layer.mlp) for layer in model.layers}) == 30

    tokens = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    logits = model(tokens)
    assert logits.shape == (batch, seq, cfg.vocab_size)
    assert torch.isfinite(logits).all()
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size), tokens[:, 1:].reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    if cfg.tie_mlp:
        grad = model.mlps[0].gate_proj.weight.grad
        assert grad is not None and torch.isfinite(grad).all()

    print(f"[PASS] {'tied-g6' if cfg.tie_mlp else 'dense'} construction/forward/backward "
          f"device={device} loss={float(loss):.6f}")
    del model, tokens, logits, loss
    if device == "cuda":
        torch.cuda.empty_cache()


def heavy_checks(dense_cfg, tied_cfg, device, batch, seq, p018_regression):
    import torch
    from tinylm.model import build_model

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda를 요청했지만 torch.cuda.is_available() == False")
    _run_one(dense_cfg, device, batch, seq)
    _run_one(tied_cfg, device, batch, seq)

    if p018_regression:
        cfg = build_config("m100", "tied", seq, False)
        model = build_model(cfg).to(device)
        assert cfg.model_family == "p018" and _parameter_count(model) > 0
        print(f"[PASS] legacy P018 construction params={_parameter_count(model):,}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-heavy", action="store_true",
                        help="모델 생성 및 forward/backward를 사용자가 명시적으로 허용")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=8)
    parser.add_argument("--p018-regression", action="store_true",
                        help="추가로 기존 m100 tied 모델 생성 회귀검사")
    args = parser.parse_args()

    dense_cfg, tied_cfg = static_checks()
    if not args.confirm_heavy:
        print("[STOP] 모델 검사는 실행하지 않음. 사용자가 --confirm-heavy를 붙여 실행하세요.")
        return 0
    heavy_checks(dense_cfg, tied_cfg, args.device, args.batch, args.seq, args.p018_regression)
    print("[PASS] all requested manual MobileLLM checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
