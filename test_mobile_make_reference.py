"""
TinyLM_P018
MobileLLM-125M Reference Builder

Environment:
    transformers == 4.41.2

Purpose:
    Construct the official MobileLLM 125M implementation and save
    its architecture/configuration/parameter inventory as a reference.

This script MUST be run in the dedicated MobileLLM reference
environment, not necessarily in the TinyLM training environment.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import torch


# ============================================================
# Repository root
# ============================================================

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# Official MobileLLM
# ============================================================

OFFICIAL_DIR = ROOT / "MobileLLM" / "utils"

if not OFFICIAL_DIR.exists():
    raise RuntimeError(
        f"Official MobileLLM directory not found:\n"
        f"{OFFICIAL_DIR}"
    )

if str(OFFICIAL_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_DIR))


try:
    import modeling_llama
except Exception:
    traceback.print_exc()
    raise


# ============================================================
# Constants
# ============================================================

VOCAB_SIZE = 32000
HIDDEN_SIZE = 576
INTERMEDIATE_SIZE = 1536
NUM_LAYERS = 30
NUM_HEADS = 9
NUM_KV_HEADS = 3
HEAD_DIM = 64

MAX_POSITION_EMBEDDINGS = 2048
ROPE_THETA = 10000.0
RMS_NORM_EPS = 1e-5
INITIALIZER_RANGE = 0.02

EXPECTED_PARAMS = 124_635_456

OUTPUT_DIR = ROOT / "parity_reference"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# Configuration
# ============================================================

def build_config():

    from transformers import LlamaConfig

    cfg = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,

        num_hidden_layers=NUM_LAYERS,

        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,

        max_position_embeddings=MAX_POSITION_EMBEDDINGS,

        initializer_range=INITIALIZER_RANGE,
        rms_norm_eps=RMS_NORM_EPS,

        rope_theta=ROPE_THETA,

        hidden_act="silu",

        attention_bias=False,
        mlp_bias=False,

        tie_word_embeddings=False,
    )

    # MobileLLM extensions.
    cfg.share_embedding = True
    cfg.layer_sharing = False
    cfg.pretraining_tp = 1

    # Critical for transformers 4.41.x.
    cfg._attn_implementation = "eager"

    return cfg


# ============================================================
# Parameter inventory
# ============================================================

def build_inventory(model):

    state = model.state_dict()

    inventory = {}

    for name, tensor in state.items():

        inventory[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": tensor.numel(),
        }

    return inventory


# ============================================================
# Architecture summary
# ============================================================

def build_summary(model, cfg):

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    return {
        "model_family": "MobileLLM",
        "variant": "125M",

        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,

        "num_hidden_layers":
            cfg.num_hidden_layers,

        "num_attention_heads":
            cfg.num_attention_heads,

        "num_key_value_heads":
            cfg.num_key_value_heads,

        "head_dim":
            HEAD_DIM,

        "max_position_embeddings":
            cfg.max_position_embeddings,

        "rope_theta":
            ROPE_THETA,

        "rms_norm_eps":
            cfg.rms_norm_eps,

        "initializer_range":
            cfg.initializer_range,

        "share_embedding":
            getattr(
                cfg,
                "share_embedding",
                False,
            ),

        "layer_sharing":
            getattr(
                cfg,
                "layer_sharing",
                False,
            ),

        "parameter_count":
            params,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 72)
    print("MobileLLM-125M REFERENCE BUILDER")
    print("=" * 72)

    print(
        "PyTorch       :",
        torch.__version__,
    )

    import transformers

    print(
        "Transformers  :",
        transformers.__version__,
    )

    if not transformers.__version__.startswith("4.41"):
        raise RuntimeError(
            "This reference builder should be executed "
            "with transformers 4.41.x."
        )

    cfg = build_config()

    model = (
        modeling_llama.LlamaForCausalLM(cfg)
        .eval()
    )

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print("Architecture")
    print("-" * 72)

    print(
        f"hidden       = {cfg.hidden_size}"
    )
    print(
        f"intermediate = {cfg.intermediate_size}"
    )
    print(
        f"layers       = {cfg.num_hidden_layers}"
    )
    print(
        f"Q/KV heads   = "
        f"{cfg.num_attention_heads}/"
        f"{cfg.num_key_value_heads}"
    )
    print(
        f"head dim     = {HEAD_DIM}"
    )
    print(
        f"vocab        = {cfg.vocab_size}"
    )
    print(
        f"context      = "
        f"{cfg.max_position_embeddings}"
    )

    print()
    print(
        f"share_embedding = "
        f"{cfg.share_embedding}"
    )

    print(
        f"parameters      = "
        f"{params:,}"
    )

    assert params == EXPECTED_PARAMS, (
        f"Reference parameter count mismatch:\n"
        f"expected = {EXPECTED_PARAMS:,}\n"
        f"actual   = {params:,}"
    )

    print(
        "[PASS] Official MobileLLM parameter count "
        "matches 124,635,456."
    )

    # --------------------------------------------------------
    # Save config
    # --------------------------------------------------------

    summary = build_summary(
        model,
        cfg,
    )

    with open(
        OUTPUT_DIR / "reference_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save parameter inventory
    # --------------------------------------------------------

    inventory = build_inventory(model)

    with open(
        OUTPUT_DIR / "reference_inventory.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            inventory,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save state dict
    # --------------------------------------------------------

    torch.save(
        model.state_dict(),
        OUTPUT_DIR / "mobilellm_125m_state_dict.pt",
    )

    print()
    print(
        f"[PASS] Reference saved to:\n"
        f"       {OUTPUT_DIR}"
    )

    print()
    print(
        f"state_dict entries = "
        f"{len(inventory)}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()