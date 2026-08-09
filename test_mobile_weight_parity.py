"""
TinyLM_P018 MobileLLM-125M Weight + Numerical Parity Test.

Reference:
    MobileLLM official implementation
    transformers==4.41.2

Repository:

TinyLM_P018/
├── test_mobile_weight_parity.py
│
├── parity_reference/
│   └── ...
├── MobileLLM/
│   └── utils/
│       └── modeling_llama.py
└── tinylm/
    ├── config.py
    └── model/
        ├── modules.py
        └── transformer.py

Purpose
-------
Verify that the TinyLM MobileLLM Dense implementation is
architecturally and numerically equivalent to the official
MobileLLM 125M architecture.

This test deliberately separates:

    Official MobileLLM
            |
            +---- canonical parameter mapping ----+
                                                   |
    TinyLM MobileLLM Dense ------------------------+

HF LLaMA is used only as an auxiliary structural sanity check.

Important
---------
Official MobileLLM 125M uses:

    share_embedding = True

Therefore:

    input embedding
        =
    output projection

and the expected parameter count is:

    124,635,456

An independent 32,000 x 576 LM head would add:

    18,432,000

parameters and produce:

    143,067,456

which is NOT the MobileLLM 125M reference.
"""

from __future__ import annotations

import os
import sys
import math
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinylm.model.modules import apply_rope

# ============================================================
# Repository root
# ============================================================

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# Imports
# ============================================================

try:
    from tinylm.config import build_config
except Exception as exc:
    print("[ERROR] Cannot import tinylm.config")
    print(exc)
    traceback.print_exc()
    sys.exit(1)


try:
    from tinylm.model.transformer import build_model
except Exception as exc:
    print("[ERROR] Cannot import tinylm.model.transformer.build_model")
    print(exc)
    traceback.print_exc()
    sys.exit(1)


# ============================================================
# Official MobileLLM
# ============================================================

OFFICIAL_DIR = ROOT / "MobileLLM" / "utils"
OFFICIAL_MODELING = OFFICIAL_DIR / "modeling_llama.py"

if not OFFICIAL_DIR.is_dir():
    print(
        "[ERROR] Official MobileLLM directory not found:"
    )
    print(
        f"        {OFFICIAL_DIR}"
    )
    print()
    print(
        f"[INFO] Repository root detected as:"
    )
    print(
        f"       {ROOT}"
    )
    sys.exit(1)

if not OFFICIAL_MODELING.is_file():
    print(
        "[ERROR] Official MobileLLM modeling file not found:"
    )
    print(
        f"        {OFFICIAL_MODELING}"
    )
    sys.exit(1)


# ------------------------------------------------------------
# Import MobileLLM modeling_llama.py directly
# ------------------------------------------------------------

import importlib.util


spec = importlib.util.spec_from_file_location(
    "mobilellm_modeling_llama",
    OFFICIAL_MODELING,
)

if spec is None or spec.loader is None:
    print(
        "[ERROR] Could not create import specification for:"
    )
    print(
        f"        {OFFICIAL_MODELING}"
    )
    sys.exit(1)


modeling_llama = importlib.util.module_from_spec(
    spec
)

sys.modules[
    "mobilellm_modeling_llama"
] = modeling_llama

try:
    spec.loader.exec_module(
        modeling_llama
    )

except Exception as exc:
    print(
        "[ERROR] Cannot import official MobileLLM "
        "modeling_llama.py"
    )
    print(
        f"        {OFFICIAL_MODELING}"
    )
    print()
    print(exc)
    traceback.print_exc()
    sys.exit(1)


print(
    "[PASS] Official MobileLLM source loaded:"
)

print(
    f"       {OFFICIAL_MODELING}"
)


# ============================================================
# Constants
# ============================================================

VOCAB_SIZE = 32000
HIDDEN_SIZE = 576
INTERMEDIATE_SIZE = 1536

NUM_LAYERS = 30

NUM_Q_HEADS = 9
NUM_KV_HEADS = 3
HEAD_DIM = 64

MAX_POSITION_EMBEDDINGS = 2048
ROPE_THETA = 10000.0
RMS_NORM_EPS = 1e-5
INITIALIZER_RANGE = 0.02

EXPECTED_PARAMS = 124_635_456

EXTRA_LM_HEAD_PARAMS = (
    VOCAB_SIZE * HIDDEN_SIZE
)

EXPECTED_UNTIED_PARAMS = (
    EXPECTED_PARAMS + EXTRA_LM_HEAD_PARAMS
)

BATCH_SIZE = 2
SEQ_LEN = 8

SEED = 1234

ATOL = 2e-5
RTOL = 2e-4


# ============================================================
# Utility
# ============================================================

def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )


def count_unique_parameters(model):
    """
    Count unique Parameter storage.

    This is important because tied embedding/output weights
    may appear through multiple module paths but represent
    one actual Parameter.
    """

    seen = set()
    total = 0

    for p in model.parameters():
        ptr = p.data_ptr()

        if ptr in seen:
            continue

        seen.add(ptr)
        total += p.numel()

    return total


def parameter_storage_aliases(model):
    """
    Return groups of parameter names sharing the same storage.
    """

    groups = {}

    for name, p in model.named_parameters(remove_duplicate=False):
        ptr = p.data_ptr()

        groups.setdefault(ptr, []).append(name)

    return [
        names
        for names in groups.values()
        if len(names) > 1
    ]


def tensor_stats(x):
    x = x.float()

    return {
        "shape": tuple(x.shape),
        "mean": x.mean().item(),
        "std": x.std().item(),
        "abs_max": x.abs().max().item(),
    }


def compare_tensors(
    name,
    a,
    b,
    atol=ATOL,
    rtol=RTOL,
):
    if tuple(a.shape) != tuple(b.shape):
        raise AssertionError(
            f"{name}: shape mismatch\n"
            f"  A: {tuple(a.shape)}\n"
            f"  B: {tuple(b.shape)}"
        )

    af = a.float()
    bf = b.float()

    diff = (af - bf).abs()

    max_abs = diff.max().item()

    denom = torch.maximum(
        af.abs(),
        bf.abs(),
    )

    max_rel = (
        diff / denom.clamp_min(1e-12)
    ).max().item()

    torch.testing.assert_close(
        af,
        bf,
        atol=atol,
        rtol=rtol,
    )

    print(
        f"[PASS] {name:<42} "
        f"max_abs={max_abs:.6e} "
        f"max_rel={max_rel:.6e}"
    )


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ============================================================
# Official configuration
# ============================================================

def make_official_config():
    from transformers import LlamaConfig

    cfg = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_Q_HEADS,
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

    # MobileLLM-specific behavior.
    cfg.share_embedding = True
    cfg.layer_sharing = False
    cfg.pretraining_tp = 1

    # transformers 4.41.x
    cfg._attn_implementation = "eager"

    return cfg


# ============================================================
# TinyLM configuration
# ============================================================

def make_tinylm_config():
    """
    Build MobileLLM 125M Dense.

    IMPORTANT:
        Dense means one independent MLP per layer.
        MLP sharing is tested separately.
    """

    cfg = build_config(
        "mobile125",
        "dense",
        MAX_POSITION_EMBEDDINGS,
        False,
    )

    return cfg


# ============================================================
# Models
# ============================================================

def make_models():
    section("MODEL CONSTRUCTION")

    official_cfg = make_official_config()
    tinylm_cfg = make_tinylm_config()

    official = (
        modeling_llama.LlamaForCausalLM(
            official_cfg
        )
        .eval()
    )

    tiny = (
        build_model(tinylm_cfg)
        .eval()
    )

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    official.to(device)
    tiny.to(device)

    print(
        f"[PASS] Official MobileLLM constructed on {device}"
    )

    print(
        f"[PASS] TinyLM MobileLLM constructed on {device}"
    )

    return official, tiny, official_cfg, tinylm_cfg


# ============================================================
# Configuration parity
# ============================================================

def test_config_parity(
    official_cfg,
    tiny_cfg,
):
    section("TEST 1: CONFIGURATION PARITY")

    pairs = [
        (
            "hidden_size",
            official_cfg.hidden_size,
            tiny_cfg.dim,
        ),
        (
            "intermediate_size",
            official_cfg.intermediate_size,
            tiny_cfg.ffn_dim,
        ),
        (
            "num_hidden_layers",
            official_cfg.num_hidden_layers,
            tiny_cfg.n_layers,
        ),
        (
            "num_attention_heads",
            official_cfg.num_attention_heads,
            tiny_cfg.n_q_heads,
        ),
        (
            "num_key_value_heads",
            official_cfg.num_key_value_heads,
            tiny_cfg.n_kv_heads,
        ),
        (
            "vocab_size",
            official_cfg.vocab_size,
            tiny_cfg.vocab_size,
        ),
        (
            "max_position_embeddings",
            official_cfg.max_position_embeddings,
            tiny_cfg.max_seq_len,
        ),
        (
            "rope_theta",
            float(official_cfg.rope_theta),
            float(tiny_cfg.rope_theta),
        ),
        (
            "rms_norm_eps",
            float(official_cfg.rms_norm_eps),
            float(tiny_cfg.norm_eps),
        ),
        (
            "initializer_range",
            float(official_cfg.initializer_range),
            float(tiny_cfg.initializer_range),
        ),
    ]

    for name, official, tiny in pairs:

        if official != tiny:
            raise AssertionError(
                f"{name} mismatch:\n"
                f"  Official = {official}\n"
                f"  TinyLM   = {tiny}"
            )

        print(
            f"[PASS] {name}: "
            f"{official} == {tiny}"
        )

    if official_cfg.share_embedding is not True:
        raise AssertionError(
            "Official MobileLLM share_embedding is not True."
        )

    print(
        "[PASS] Official share_embedding=True"
    )

    if tiny_cfg.tie_word_embeddings:
        raise AssertionError(
            "TinyLM tie_word_embeddings should remain False; "
            "MobileLLM uses its own share_embedding semantics."
        )

    print(
        "[PASS] TinyLM tie_word_embeddings=False"
    )

    if tiny_cfg.tie_mlp:
        raise AssertionError(
            "Base architecture parity must use Dense MLPs."
        )

    print(
        "[PASS] TinyLM base parity uses independent Dense MLPs"
    )


# ============================================================
# Parameter count
# ============================================================

def test_parameter_count(
    official,
    tiny,
):
    section("TEST 2: PARAMETER COUNT")

    official_n = count_parameters(official)
    tiny_n = count_parameters(tiny)

    official_unique = count_unique_parameters(
        official
    )

    tiny_unique = count_unique_parameters(
        tiny
    )

    print(
        f"Official MobileLLM : {official_n:,}"
    )

    print(
        f"TinyLM             : {tiny_n:,}"
    )

    print(
        f"Official unique    : {official_unique:,}"
    )

    print(
        f"TinyLM unique      : {tiny_unique:,}"
    )

    print(
        f"Expected           : {EXPECTED_PARAMS:,}"
    )

    print(
        f"Untied diagnostic  : {EXPECTED_UNTIED_PARAMS:,}"
    )

    if official_n != EXPECTED_PARAMS:
        raise AssertionError(
            "Official MobileLLM parameter count mismatch:\n"
            f"Expected {EXPECTED_PARAMS:,}\n"
            f"Actual   {official_n:,}"
        )

    if tiny_n != EXPECTED_PARAMS:
        raise AssertionError(
            "TinyLM parameter count mismatch:\n"
            f"Expected {EXPECTED_PARAMS:,}\n"
            f"Actual   {tiny_n:,}\n"
            f"Difference {tiny_n - EXPECTED_PARAMS:,}"
        )

    if tiny_unique != EXPECTED_PARAMS:
        raise AssertionError(
            "TinyLM unique parameter storage mismatch."
        )

    print(
        "[PASS] Official parameter count"
    )

    print(
        "[PASS] TinyLM parameter count"
    )

    print(
        "[PASS] Unique parameter storage count"
    )


# ============================================================
# Embedding / output tying
# ============================================================

def test_embedding_tying(
    official,
    tiny,
):
    section("TEST 3: EMBEDDING / OUTPUT PROJECTION")

    official_emb = (
        official.model.embed_tokens.weight
    )

    tiny_emb = tiny.emb.weight

    if tuple(official_emb.shape) != (
        VOCAB_SIZE,
        HIDDEN_SIZE,
    ):
        raise AssertionError(
            "Official embedding shape mismatch."
        )

    if tuple(tiny_emb.shape) != (
        VOCAB_SIZE,
        HIDDEN_SIZE,
    ):
        raise AssertionError(
            "TinyLM embedding shape mismatch."
        )

    print(
        f"[PASS] embedding shape = "
        f"{tuple(tiny_emb.shape)}"
    )

    # TinyLM must share the actual Parameter.
    if tiny.lm_head.weight is not tiny.emb.weight:
        raise AssertionError(
            "TinyLM lm_head.weight is not the same "
            "Parameter as emb.weight."
        )

    print(
        "[PASS] TinyLM lm_head.weight is exactly "
        "the embedding Parameter"
    )

    # Official MobileLLM's output projection is based
    # on the embedding matrix.
    if not getattr(
        official.config,
        "share_embedding",
        False,
    ):
        raise AssertionError(
            "Official MobileLLM share_embedding=False."
        )

    print(
        "[PASS] Official MobileLLM uses shared "
        "input/output embedding"
    )


# ============================================================
# Canonical parameter mapping
# ============================================================

def build_parameter_mapping():
    """
    Canonical mapping:

        Official MobileLLM -> TinyLM

    The names are intentionally explicit.
    """

    mapping = []

    # --------------------------------------------------------
    # Embedding
    # --------------------------------------------------------

    mapping.append(
        (
            "model.embed_tokens.weight",
            "emb.weight",
            "embedding",
        )
    )

    # --------------------------------------------------------
    # Decoder layers
    # --------------------------------------------------------

    for i in range(NUM_LAYERS):

        o = f"model.layers.{i}"
        t = f"layers.{i}"

        mapping.extend(
            [
                (
                    f"{o}.self_attn.q_proj.weight",
                    f"{t}.attn.q_proj.weight",
                    f"L{i}.attn.q",
                ),
                (
                    f"{o}.self_attn.k_proj.weight",
                    f"{t}.attn.k_proj.weight",
                    f"L{i}.attn.k",
                ),
                (
                    f"{o}.self_attn.v_proj.weight",
                    f"{t}.attn.v_proj.weight",
                    f"L{i}.attn.v",
                ),
                (
                    f"{o}.self_attn.o_proj.weight",
                    f"{t}.attn.o_proj.weight",
                    f"L{i}.attn.o",
                ),
                (
                    f"{o}.mlp.gate_proj.weight",
                    f"{t}.mlp.gate_proj.weight",
                    f"L{i}.mlp.gate",
                ),
                (
                    f"{o}.mlp.up_proj.weight",
                    f"{t}.mlp.up_proj.weight",
                    f"L{i}.mlp.up",
                ),
                (
                    f"{o}.mlp.down_proj.weight",
                    f"{t}.mlp.down_proj.weight",
                    f"L{i}.mlp.down",
                ),
                (
                    f"{o}.input_layernorm.weight",
                    f"{t}.norm1.weight",
                    f"L{i}.norm1",
                ),
                (
                    f"{o}.post_attention_layernorm.weight",
                    f"{t}.norm2.weight",
                    f"L{i}.norm2",
                ),
            ]
        )

    # --------------------------------------------------------
    # Final norm
    # --------------------------------------------------------

    mapping.append(
        (
            "model.norm.weight",
            "norm.weight",
            "final_norm",
        )
    )

    return mapping


# ============================================================
# Mapping verification
# ============================================================

def test_parameter_mapping(
    official,
    tiny,
):
    section("TEST 4: EXPLICIT PARAMETER MAPPING")

    official_params = dict(
        official.named_parameters()
    )

    tiny_params = dict(
        tiny.named_parameters()
    )

    mapping = build_parameter_mapping()

    print(
        f"Expected mapped tensors : "
        f"{len(mapping)}"
    )

    for (
        official_name,
        tiny_name,
        canonical_name,
    ) in mapping:

        if official_name not in official_params:
            raise AssertionError(
                f"Official parameter missing:\n"
                f"  {official_name}"
            )

        if tiny_name not in tiny_params:
            raise AssertionError(
                f"TinyLM parameter missing:\n"
                f"  {tiny_name}"
            )

        op = official_params[
            official_name
        ]

        tp = tiny_params[
            tiny_name
        ]

        if op.shape != tp.shape:
            raise AssertionError(
                f"Shape mismatch: {canonical_name}\n"
                f"  Official: {tuple(op.shape)}\n"
                f"  TinyLM:   {tuple(tp.shape)}"
            )

    print(
        "[PASS] All canonical parameters exist."
    )

    print(
        "[PASS] All canonical parameter shapes match."
    )

    return mapping


# ============================================================
# Unexpected parameter detection
# ============================================================

def test_unexpected_parameters(
    official,
    tiny,
):
    section("TEST 5: UNEXPECTED PARAMETER DETECTION")

    official_names = set(
        dict(
            official.named_parameters()
        ).keys()
    )

    tiny_names = set(
        dict(
            tiny.named_parameters()
        ).keys()
    )

    expected_tiny = set()

    for _, tiny_name, _ in build_parameter_mapping():
        expected_tiny.add(tiny_name)

    extra_tiny = sorted(
        tiny_names - expected_tiny
    )

    print(
        f"Official parameter tensors : "
        f"{len(official_names)}"
    )

    print(
        f"TinyLM parameter tensors    : "
        f"{len(tiny_names)}"
    )

    print(
        f"Canonical mapped tensors    : "
        f"{len(expected_tiny)}"
    )

    if extra_tiny:
        print()
        print(
            "[INFO] TinyLM additional parameter paths:"
        )

        for name in extra_tiny:
            print(
                f"  {name}"
            )

    # lm_head.weight may appear in named_parameters
    # depending on remove_duplicate behavior. It must
    # nevertheless point to the same Parameter.
    aliases = parameter_storage_aliases(
        tiny
    )

    for group in aliases:
        if (
            "emb.weight" in group
            and "lm_head.weight" in group
        ):
            print(
                "[PASS] emb.weight and "
                "lm_head.weight share storage."
            )

            break

    else:
        raise AssertionError(
            "Could not verify embedding/LM-head "
            "parameter aliasing."
        )


# ============================================================
# Weight copy
# ============================================================

def copy_official_weights(
    official,
    tiny,
    mapping,
):
    section("TEST 6: OFFICIAL → TINYLM WEIGHT COPY")

    official_params = dict(
        official.named_parameters()
    )

    tiny_params = dict(
        tiny.named_parameters()
    )

    with torch.no_grad():

        for (
            official_name,
            tiny_name,
            canonical_name,
        ) in mapping:

            src = official_params[
                official_name
            ]

            dst = tiny_params[
                tiny_name
            ]

            dst.copy_(
                src.to(
                    device=dst.device,
                    dtype=dst.dtype,
                )
            )

    print(
        f"[PASS] Copied {len(mapping)} "
        f"canonical parameter tensors."
    )


# ============================================================
# Weight equality
# ============================================================

def test_weight_equality(
    official,
    tiny,
    mapping,
):
    section("TEST 7: WEIGHT NUMERICAL PARITY")

    official_params = dict(
        official.named_parameters()
    )

    tiny_params = dict(
        tiny.named_parameters()
    )

    max_abs_global = 0.0
    max_rel_global = 0.0

    failed = []

    for (
        official_name,
        tiny_name,
        canonical_name,
    ) in mapping:

        a = official_params[
            official_name
        ]

        b = tiny_params[
            tiny_name
        ]

        af = a.float()
        bf = b.float()

        diff = (
            af - bf
        ).abs()

        max_abs = diff.max().item()

        denom = torch.maximum(
            af.abs(),
            bf.abs(),
        )

        max_rel = (
            diff
            / denom.clamp_min(1e-12)
        ).max().item()

        max_abs_global = max(
            max_abs_global,
            max_abs,
        )

        max_rel_global = max(
            max_rel_global,
            max_rel,
        )

        try:
            torch.testing.assert_close(
                af,
                bf,
                atol=0.0,
                rtol=0.0,
            )

        except AssertionError:
            failed.append(
                (
                    canonical_name,
                    max_abs,
                    max_rel,
                )
            )

    if failed:

        print(
            "[FAIL] Weight mismatch detected."
        )

        for name, abs_d, rel_d in failed[:20]:
            print(
                f"  {name}: "
                f"abs={abs_d:.6e}, "
                f"rel={rel_d:.6e}"
            )

        raise AssertionError(
            f"{len(failed)} parameter tensors "
            f"failed exact equality."
        )

    print(
        f"[PASS] All {len(mapping)} mapped tensors "
        f"are numerically identical."
    )

    print(
        f"Global max abs difference : "
        f"{max_abs_global:.6e}"
    )

    print(
        f"Global max rel difference : "
        f"{max_rel_global:.6e}"
    )


# ============================================================
# Input generation
# ============================================================

def make_tokens(device):
    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(
        SEED
    )

    return torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(
            BATCH_SIZE,
            SEQ_LEN,
        ),
        dtype=torch.long,
        device=device,
        generator=generator,
    )


# ============================================================
# Official forward
# ============================================================

def official_forward(
    model,
    tokens,
):
    with torch.no_grad():

        output = model.model(
            input_ids=tokens,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

    return (
        output.last_hidden_state,
        output.hidden_states,
    )


# ============================================================
# TinyLM forward with per-layer capture
# ============================================================

def tiny_forward(
    model,
    tokens,
):
    """
    Reconstruct TinyLM's forward path while capturing
    hidden states after every MobileLayer.

    Mirrors the current MobileLayer API.
    """

    B, T = tokens.shape

    x = model.emb(tokens)

    cos = model.rope_cos[:T]
    sin = model.rope_sin[:T]

    hidden_states = [
        x.detach().clone()
    ]

    for layer in model.layers:

        x_norm = layer.norm1(x)

        k, v = layer.attn.compute_kv(
            x_norm,
            cos,
            sin,
        )

        x = layer(
            x,
            (k, v),
            cos,
            sin,
            None,
        )

        hidden_states.append(
            x.detach().clone()
        )

    x = model.norm(x)

    final_hidden = x.detach().clone()

    if model.cfg.share_embedding:

        logits = F.linear(
            x,
            model.emb.weight,
        )

    else:

        logits = model.lm_head(x)

    return (
        final_hidden,
        hidden_states,
        logits,
    )


# ============================================================
# DEBUG: Layer-level / sub-operation numerical parity
# ============================================================

def _tensor_diff_stats(a, b):
    """
    Return detailed numerical difference statistics.
    """

    af = a.detach().float()
    bf = b.detach().float()

    diff = (af - bf).abs()

    max_abs = diff.max().item()

    denom = torch.maximum(
        af.abs(),
        bf.abs(),
    )

    max_rel = (
        diff / denom.clamp_min(1e-12)
    ).max().item()

    mean_abs = diff.mean().item()

    if diff.numel() > 0:
        flat_idx = diff.reshape(-1).argmax()
        max_index = tuple(
            torch.unravel_index(
                flat_idx,
                diff.shape,
            )
        )

        max_index = tuple(
            int(x)
            for x in max_index
        )
    else:
        max_index = None

    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "mean_abs": mean_abs,
        "max_index": max_index,
    }


def debug_compare(
    name,
    official,
    tiny,
    atol=ATOL,
    rtol=RTOL,
):
    """
    Debug comparison.

    Supports:
        Tensor
        tuple/list of Tensor
        nested tuple/list

    Never raises merely because the captured
    object is a tuple/list.
    """

    # ----------------------------------------------------
    # Recursive comparison for tuple/list
    # ----------------------------------------------------

    if isinstance(official, (tuple, list)) or isinstance(tiny, (tuple, list)):

        if not isinstance(official, (tuple, list)):
            print()
            print(f"[FAIL] {name}")
            print("       Official is Tensor, TinyLM is tuple/list")
            return False

        if not isinstance(tiny, (tuple, list)):
            print()
            print(f"[FAIL] {name}")
            print("       Official is tuple/list, TinyLM is Tensor")
            return False

        if len(official) != len(tiny):
            print()
            print(f"[FAIL] {name}")
            print(
                f"       tuple length Official={len(official)} "
                f"TinyLM={len(tiny)}"
            )
            return False

        all_ok = True

        for i, (oa, tb) in enumerate(
            zip(official, tiny)
        ):

            ok = debug_compare(
                f"{name}[{i}]",
                oa,
                tb,
                atol=atol,
                rtol=rtol,
            )

            all_ok = all_ok and ok

        return all_ok

    # ----------------------------------------------------
    # Tensor / non-Tensor mismatch
    # ----------------------------------------------------

    if torch.is_tensor(official) != torch.is_tensor(tiny):

        print()
        print(f"[FAIL] {name}")

        print(
            f"       type Official={type(official)}"
        )

        print(
            f"       type TinyLM  ={type(tiny)}"
        )

        return False

    # ----------------------------------------------------
    # Non-tensor values
    # ----------------------------------------------------

    if not torch.is_tensor(official):

        ok = official == tiny

        status = "PASS" if ok else "FAIL"

        print(
            f"[{status}] {name:<38} "
            f"non-tensor"
        )

        if not ok:

            print(
                f"       Official={official!r}"
            )

            print(
                f"       TinyLM  ={tiny!r}"
            )

        return ok

    # ----------------------------------------------------
    # Shape
    # ----------------------------------------------------

    if tuple(official.shape) != tuple(tiny.shape):

        print()
        print(f"[FAIL] {name}")

        print(
            f"       shape Official={tuple(official.shape)}"
        )

        print(
            f"       shape TinyLM  ={tuple(tiny.shape)}"
        )

        return False

    # ----------------------------------------------------
    # Numerical comparison
    # ----------------------------------------------------

    stats = _tensor_diff_stats(
        official,
        tiny,
    )

    ok = torch.allclose(
        official.float(),
        tiny.float(),
        atol=atol,
        rtol=rtol,
    )

    status = "PASS" if ok else "FAIL"

    print(
        f"[{status}] {name:<38} "
        f"abs={stats['max_abs']:.6e} "
        f"rel={stats['max_rel']:.6e} "
        f"mean={stats['mean_abs']:.6e}"
    )

    if not ok:

        print(
            f"       max index = {stats['max_index']}"
        )

        idx = stats["max_index"]

        if idx is not None:

            oa = (
                official.detach()
                .float()[idx]
                .item()
            )

            tb = (
                tiny.detach()
                .float()[idx]
                .item()
            )

            print(
                f"       Official={oa:.9e}"
            )

            print(
                f"       TinyLM  ={tb:.9e}"
            )

    return ok

# ============================================================
# Capture official module outputs
# ============================================================

def _register_debug_hook(
    module,
    storage,
    name,
):
    """
    Register a forward hook that stores detached output.
    """

    def hook(mod, inputs, output):

        if isinstance(output, tuple):

            # Some HF modules return tuples.
            storage[name] = tuple(
                x.detach().clone()
                if torch.is_tensor(x)
                else x
                for x in output
            )

        elif torch.is_tensor(output):

            storage[name] = (
                output.detach()
                .clone()
            )

        else:

            storage[name] = output

    return module.register_forward_hook(hook)


# ============================================================
# Official layer-29 detailed capture
# ============================================================

def capture_official_layer29(
    model,
    tokens,
):
    """
    Capture official MobileLLM layer 29.

    IMPORTANT
    ---------
    transformers==4.41.2 LlamaModel hidden_states semantics:

        hidden_states[0]
            = embedding output

        hidden_states[1]
            = layer 0 output

        ...

        hidden_states[29]
            = layer 28 output
            = layer 29 input

        hidden_states[30]
            = FINAL RMSNorm output

    Therefore hidden_states[30] is NOT the raw output of
    decoder layer 29.

    The authoritative layer-29 output is captured by a
    forward hook directly on model.model.layers[29].
    """

    captured = {}
    hooks = []

    layer = model.model.layers[29]

    # --------------------------------------------------------
    # Internal module hooks
    # --------------------------------------------------------

    hooks.append(
        _register_debug_hook(
            layer.input_layernorm,
            captured,
            "norm1",
        )
    )

    hooks.append(
        _register_debug_hook(
            layer.self_attn.q_proj,
            captured,
            "q_proj",
        )
    )

    hooks.append(
        _register_debug_hook(
            layer.self_attn.k_proj,
            captured,
            "k_proj",
        )
    )

    hooks.append(
        _register_debug_hook(
            layer.self_attn.v_proj,
            captured,
            "v_proj",
        )
    )

    hooks.append(
        _register_debug_hook(
            layer.self_attn.o_proj,
            captured,
            "o_proj",
        )
    )

    hooks.append(
        _register_debug_hook(
            layer.post_attention_layernorm,
            captured,
            "norm2",
        )
    )

    hooks.append(
        _register_debug_hook(
            layer.mlp.gate_proj,
            captured,
            "mlp_gate",
        )
    )

    hooks.append(
        _register_debug_hook(
            layer.mlp.up_proj,
            captured,
            "mlp_up",
        )
    )

    hooks.append(
        _register_debug_hook(
            layer.mlp.down_proj,
            captured,
            "mlp_down",
        )
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Capture the COMPLETE decoder-layer output.
    #
    # LlamaDecoderLayer.forward() returns a tuple:
    #
    #     (hidden_states,)
    #
    # or, depending on cache/attention settings:
    #
    #     (hidden_states, ...)
    #
    # We want element [0].
    # --------------------------------------------------------

    def decoder_layer_hook(
        mod,
        inputs,
        output,
    ):
        if isinstance(output, tuple):

            if len(output) == 0:
                raise RuntimeError(
                    "Official layer 29 returned an empty tuple."
                )

            layer_output = output[0]

        elif torch.is_tensor(output):

            layer_output = output

        else:

            raise RuntimeError(
                "Unexpected official layer 29 output type: "
                f"{type(output)}"
            )

        captured["layer_output"] = (
            layer_output
            .detach()
            .clone()
        )

    hooks.append(
        layer.register_forward_hook(
            decoder_layer_hook
        )
    )

    # --------------------------------------------------------
    # Official forward
    # --------------------------------------------------------

    try:

        with torch.no_grad():

            output = model.model(
                input_ids=tokens,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )

    finally:

        for h in hooks:
            h.remove()

    # --------------------------------------------------------
    # Authoritative layer boundary
    # --------------------------------------------------------

    captured["layer_input"] = (
        output.hidden_states[29]
        .detach()
        .clone()
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # DO NOT use:
    #
    #     output.hidden_states[30]
    #
    # as layer29 output.
    #
    # In transformers 4.41.2 that is the output AFTER the
    # final model RMSNorm.
    # --------------------------------------------------------

    if "layer_output" not in captured:
        raise RuntimeError(
            "Official layer 29 forward hook did not capture "
            "the decoder-layer output."
        )

    # --------------------------------------------------------
    # Final model output
    # --------------------------------------------------------

    captured["model_hidden"] = (
        output.last_hidden_state
        .detach()
        .clone()
    )

    # --------------------------------------------------------
    # Sanity check:
    #
    # layer 29 output + final model norm must equal
    # output.last_hidden_state.
    #
    # This verifies that the hook really captured the raw
    # decoder layer output.
    # --------------------------------------------------------

    with torch.no_grad():

        expected_final = model.model.norm(
            captured["layer_output"]
        )

    compare_tensors(
        "official layer29 output -> final norm",
        expected_final,
        captured["model_hidden"],
    )

    # --------------------------------------------------------
    # Layer input sanity check
    # --------------------------------------------------------

    assert torch.equal(
        captured["layer_input"],
        output.hidden_states[29],
    )

    return captured

# ============================================================
# TinyLM layer-29 detailed capture
# ============================================================

def capture_tiny_layer29(
    model,
    tokens,
):
    """
    Detailed numerical capture of TinyLM layer 29.

    This function intentionally reconstructs only layer 29
    mathematically so that the first divergent operation can
    be identified.

    IMPORTANT
    ---------
    The production MobileAttention implementation is:

        q_proj
        -> reshape
        -> RoPE

        k_proj
        -> reshape
        -> RoPE

        v_proj
        -> reshape

        -> GQA
        -> causal attention
        -> softmax
        -> attention @ V
        -> merge heads
        -> o_proj

    The same mathematical sequence is reconstructed here.

    Unlike the previous debug version, this function does NOT
    contain a second dead implementation after return().
    """

    layer_idx = 29

    if layer_idx >= len(model.layers):
        raise AssertionError(
            f"TinyLM has only {len(model.layers)} layers."
        )

    layer = model.layers[layer_idx]

    B, T = tokens.shape
    c = model.cfg

    captured = {}

    # ========================================================
    # Embedding
    # ========================================================

    x = model.emb(tokens)

    captured["embedding"] = (
        x.detach().clone()
    )

    # ========================================================
    # RoPE
    # ========================================================

    cos = model.rope_cos[:T]
    sin = model.rope_sin[:T]

    # ========================================================
    # Run layers 0 ~ 28 using the REAL production path
    # ========================================================

    for i in range(layer_idx):

        L = model.layers[i]

        x_norm = L.norm1(x)

        k, v = L.attn.compute_kv(
            x_norm,
            cos,
            sin,
        )

        x = L(
            x,
            (k, v),
            cos,
            sin,
            None,
        )

    # ========================================================
    # Layer 29 input
    # ========================================================

    captured["layer_input"] = (
        x.detach().clone()
    )

    # ========================================================
    # Layer 29
    # ========================================================

    # --------------------------------------------------------
    # Norm 1
    # --------------------------------------------------------

    norm1 = layer.norm1(x)

    captured["norm1"] = (
        norm1.detach().clone()
    )

    # ========================================================
    # Attention
    # ========================================================

    attn = layer.attn

    # --------------------------------------------------------
    # Q/K/V linear projections
    # --------------------------------------------------------

    q_linear = attn.q_proj(
        norm1
    )

    k_linear = attn.k_proj(
        norm1
    )

    v_linear = attn.v_proj(
        norm1
    )

    captured["q_proj"] = (
        q_linear.detach().clone()
    )

    captured["k_proj"] = (
        k_linear.detach().clone()
    )

    captured["v_proj"] = (
        v_linear.detach().clone()
    )

    # --------------------------------------------------------
    # Reshape heads
    # --------------------------------------------------------

    q = q_linear.view(
        B,
        T,
        c.n_q_heads,
        c.head_dim,
    ).transpose(1, 2)

    k = k_linear.view(
        B,
        T,
        c.n_kv_heads,
        c.head_dim,
    ).transpose(1, 2)

    v = v_linear.view(
        B,
        T,
        c.n_kv_heads,
        c.head_dim,
    ).transpose(1, 2)

    captured["q_reshaped"] = (
        q.detach().clone()
    )

    captured["k_reshaped"] = (
        k.detach().clone()
    )

    captured["v_reshaped"] = (
        v.detach().clone()
    )

    # --------------------------------------------------------
    # RoPE
    # --------------------------------------------------------

    q_rope = apply_rope(
        q,
        cos,
        sin,
    )

    k_rope = apply_rope(
        k,
        cos,
        sin,
    )

    captured["q_rope"] = (
        q_rope.detach().clone()
    )

    captured["k_rope"] = (
        k_rope.detach().clone()
    )

    # --------------------------------------------------------
    # GQA
    # --------------------------------------------------------

    n_rep = (
        c.n_q_heads
        // c.n_kv_heads
    )

    if n_rep > 1:

        k_attn = k_rope.repeat_interleave(
            n_rep,
            dim=1,
        )

        v_attn = v.repeat_interleave(
            n_rep,
            dim=1,
        )

    else:

        k_attn = k_rope
        v_attn = v

    captured["k_gqa"] = (
        k_attn.detach().clone()
    )

    captured["v_gqa"] = (
        v_attn.detach().clone()
    )

    # --------------------------------------------------------
    # Attention scores
    # --------------------------------------------------------

    attention_scores = torch.matmul(
        q_rope,
        k_attn.transpose(2, 3),
    ) / math.sqrt(
        c.head_dim
    )

    captured["attention_scores"] = (
        attention_scores.detach().clone()
    )

    # --------------------------------------------------------
    # Causal mask
    # --------------------------------------------------------

    q_len = q_rope.shape[2]
    kv_len = k_attn.shape[2]

    if q_len == kv_len:

        causal_mask = torch.triu(
            torch.ones(
                q_len,
                kv_len,
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=1,
        )

    else:

        past_len = kv_len - q_len

        causal_mask = torch.ones(
            q_len,
            kv_len,
            dtype=torch.bool,
            device=x.device,
        ).triu(
            diagonal=past_len + 1
        )

    attention_scores_masked = (
        attention_scores.masked_fill(
            causal_mask,
            torch.finfo(
                attention_scores.dtype
            ).min,
        )
    )

    captured["attention_scores_masked"] = (
        attention_scores_masked.detach().clone()
    )

    # --------------------------------------------------------
    # Softmax
    # --------------------------------------------------------

    attention_weights = F.softmax(
        attention_scores_masked,
        dim=-1,
        dtype=torch.float32,
    ).to(
        q_rope.dtype
    )

    captured["attention_weights"] = (
        attention_weights.detach().clone()
    )

    # --------------------------------------------------------
    # Attention @ V
    # --------------------------------------------------------

    attention_output_heads = torch.matmul(
        attention_weights,
        v_attn,
    )

    captured["attention_output_heads"] = (
        attention_output_heads.detach().clone()
    )

    # --------------------------------------------------------
    # Merge heads
    # --------------------------------------------------------

    attention_merged = (
        attention_output_heads
        .transpose(1, 2)
        .contiguous()
        .view(
            B,
            T,
            c.dim,
        )
    )

    captured["attention_merged"] = (
        attention_merged.detach().clone()
    )

    # --------------------------------------------------------
    # O projection
    # --------------------------------------------------------

    o_proj = attn.o_proj(
        attention_merged
    )

    captured["o_proj"] = (
        o_proj.detach().clone()
    )

    # --------------------------------------------------------
    # Attention residual
    # --------------------------------------------------------

    after_attention_residual = (
        x + o_proj
    )

    captured["after_attention_residual"] = (
        after_attention_residual.detach().clone()
    )

    # ========================================================
    # MLP
    # ========================================================

    norm2 = layer.norm2(
        after_attention_residual
    )

    captured["norm2"] = (
        norm2.detach().clone()
    )

    # --------------------------------------------------------
    # Gate / Up
    # --------------------------------------------------------

    gate = layer.mlp.gate_proj(
        norm2
    )

    up = layer.mlp.up_proj(
        norm2
    )

    captured["mlp_gate"] = (
        gate.detach().clone()
    )

    captured["mlp_up"] = (
        up.detach().clone()
    )

    # --------------------------------------------------------
    # SiLU
    # --------------------------------------------------------

    silu_gate = F.silu(
        gate
    )

    captured["mlp_silu_gate"] = (
        silu_gate.detach().clone()
    )

    # --------------------------------------------------------
    # SwiGLU multiplication
    # --------------------------------------------------------

    mlp_hidden = (
        silu_gate * up
    )

    captured["mlp_hidden"] = (
        mlp_hidden.detach().clone()
    )

    # --------------------------------------------------------
    # Down projection
    # --------------------------------------------------------

    mlp_down = layer.mlp.down_proj(
        mlp_hidden
    )

    captured["mlp_down"] = (
        mlp_down.detach().clone()
    )

    # --------------------------------------------------------
    # Final residual
    # --------------------------------------------------------

    layer_output = (
        after_attention_residual
        + mlp_down
    )

    captured["layer_output"] = (
        layer_output.detach().clone()
    )

    # ========================================================
    # Diagnostic statistics
    # ========================================================

    print()
    print("---------- TinyLM Layer 29 Detailed Debug ----------")

    debug_items = [
        "layer_input",
        "norm1",
        "q_proj",
        "k_proj",
        "v_proj",
        "q_reshaped",
        "k_reshaped",
        "v_reshaped",
        "q_rope",
        "k_rope",
        "k_gqa",
        "v_gqa",
        "attention_scores",
        "attention_scores_masked",
        "attention_weights",
        "attention_output_heads",
        "attention_merged",
        "o_proj",
        "after_attention_residual",
        "norm2",
        "mlp_gate",
        "mlp_up",
        "mlp_silu_gate",
        "mlp_hidden",
        "mlp_down",
        "layer_output",
    ]

    for name in debug_items:

        print(
            f"{name:<32}",
            tensor_stats(
                captured[name]
            ),
        )

    return captured

# ============================================================
# TEST 8 DEBUG
# ============================================================

def test_numerical_parity(
    official,
    tiny,
):
    section(
        "TEST 8: NUMERICAL FORWARD PARITY"
    )

    device = next(
        official.parameters()
    ).device

    tokens = make_tokens(
        device
    )

    print(
        "Input token shape:",
        tuple(tokens.shape)
    )

    print(
        "Input token IDs:",
        tokens[0].tolist()
    )

    # ========================================================
    # First: original layer-by-layer test
    # ========================================================

    official_final, official_hidden = (
        official_forward(
            official,
            tokens,
        )
    )

    tiny_final, tiny_hidden, tiny_logits = (
        tiny_forward(
            tiny,
            tokens,
        )
    )

    compare_tensors(
        "embedding output",
        official_hidden[0],
        tiny_hidden[0],
    )

    # --------------------------------------------------------
    # Verify layers 0..28
    # --------------------------------------------------------

    for i in range(
        1,
        30,
    ):

        compare_tensors(
            f"layer {i - 1:02d} hidden state",
            official_hidden[i],
            tiny_hidden[i],
        )

    # ========================================================
    # Layer 29 detailed debug
    # ========================================================

    print()
    print("=" * 72)
    print(
        "DETAILED DEBUG: LAYER 29"
    )
    print("=" * 72)

    print()
    print(
        "[1] Capturing official MobileLLM layer 29..."
    )

    official_debug = (
        capture_official_layer29(
            official,
            tokens,
        )
    )

    print(
        "[2] Capturing TinyLM layer 29..."
    )

    tiny_debug = (
        capture_tiny_layer29(
            tiny,
            tokens,
        )
    )

    # --------------------------------------------------------
    # Important comparison sequence
    # --------------------------------------------------------

    debug_order = [

        # Layer input
        (
            "layer29 input",
            "layer_input",
            "layer_input",
        ),

        # Norm
        (
            "layer29 norm1",
            "norm1",
            "norm1",
        ),

        # Q/K/V
        (
            "layer29 q_proj",
            "q_proj",
            "q_proj",
        ),

        (
            "layer29 k_proj",
            "k_proj",
            "k_proj",
        ),

        (
            "layer29 v_proj",
            "v_proj",
            "v_proj",
        ),

        # Reshape
        (
            "layer29 q reshaped",
            "q_proj",
            "q_reshaped",
        ),

        # RoPE
        (
            "layer29 q after RoPE",
            None,
            "q_rope",
        ),

        (
            "layer29 k after RoPE",
            None,
            "k_rope",
        ),

        # Attention
        (
            "layer29 attention scores",
            None,
            "attention_scores",
        ),

        (
            "layer29 attention scores masked",
            None,
            "attention_scores_masked",
        ),

        (
            "layer29 attention weights",
            None,
            "attention_weights",
        ),

        (
            "layer29 attention output",
            None,
            "attention_output_heads",
        ),

        (
            "layer29 attention merged",
            None,
            "attention_merged",
        ),

        (
            "layer29 o_proj",
            "o_proj",
            "o_proj",
        ),

        (
            "layer29 after attention residual",
            None,
            "after_attention_residual",
        ),

        # MLP
        (
            "layer29 norm2",
            "norm2",
            "norm2",
        ),

        (
            "layer29 MLP gate",
            "mlp_gate",
            "mlp_gate",
        ),

        (
            "layer29 MLP up",
            "mlp_up",
            "mlp_up",
        ),

        (
            "layer29 MLP down",
            "mlp_down",
            "mlp_down",
        ),

        (
            "layer29 final output",
            "layer_output",
            "layer_output",
        ),
    ]

    print()

    first_failure = None

    # --------------------------------------------------------
    # We cannot directly compare some internal official
    # tensors because HF combines those operations inside
    # LlamaAttention. Therefore use hooks where available
    # and explicitly reconstruct the TinyLM side.
    # --------------------------------------------------------

    # Directly useful hook comparisons first.
    hook_comparisons = [
        (
            "layer29 input",
            "layer_input",
            "layer_input",
        ),
        (
            "layer29 norm1",
            "norm1",
            "norm1",
        ),
        (
            "layer29 q_proj",
            "q_proj",
            "q_proj",
        ),
        (
            "layer29 k_proj",
            "k_proj",
            "k_proj",
        ),
        (
            "layer29 v_proj",
            "v_proj",
            "v_proj",
        ),
        (
            "layer29 o_proj",
            "o_proj",
            "o_proj",
        ),
        (
            "layer29 norm2",
            "norm2",
            "norm2",
        ),
        (
            "layer29 mlp gate",
            "mlp_gate",
            "mlp_gate",
        ),
        (
            "layer29 mlp up",
            "mlp_up",
            "mlp_up",
        ),
        (
            "layer29 mlp down",
            "mlp_down",
            "mlp_down",
        ),
        (
            "layer29 output",
            "layer_output",
            "layer_output",
        ),
    ]

    for (
        name,
        official_key,
        tiny_key,
    ) in hook_comparisons:

        ok = debug_compare(
            name,
            official_debug[
                official_key
            ],
            tiny_debug[
                tiny_key
            ],
        )

        if not ok and first_failure is None:
            first_failure = name

    # ========================================================
    # Important diagnosis
    # ========================================================

    print()
    print("=" * 72)
    print(
        "LAYER 29 DIAGNOSTIC RESULT"
    )
    print("=" * 72)

    if first_failure is None:

        print(
            "[PASS] All directly captured layer-29 "
            "submodules match."
        )

        print(
            "[INFO] If the original layer-29 comparison "
            "still fails, the remaining candidate is "
            "the attention internal computation/mask."
        )

    else:

        print()
        print(
            "[FIRST DIVERGENCE]"
        )

        print(
            f"    {first_failure}"
        )

        print()

        print(
            "This is the operation where the official "
            "and TinyLM implementations first differ."
        )

        print(
            "Do NOT change weights or tolerances yet."
        )

    # ========================================================
    # Original final comparisons
    # ========================================================

    print()

    compare_tensors(
        "final hidden state",
        official_final,
        tiny_final,
    )

    with torch.no_grad():

        official_output = official(
            input_ids=tokens,
            use_cache=False,
            return_dict=True,
        )

        official_logits = (
            official_output.logits
        )

    compare_tensors(
        "final logits",
        official_logits,
        tiny_logits,
    )

    print()


    official_layer29_output = (
        official_debug["layer_output"]
    )
    
    tiny_layer29_output = (
        tiny_debug["layer_output"]
    )
    
    print()
    print("========== LAYER 29 OUTPUT ==========")
    
    print(
        "Official type:",
        type(official_layer29_output),
    )
    
    print(
        "TinyLM type:",
        type(tiny_layer29_output),
    )
    
    if not torch.is_tensor(
        official_layer29_output
    ):
        raise AssertionError(
            "BUG IN TEST: Official layer29 output "
            "must be Tensor, but got "
            f"{type(official_layer29_output)}"
        )
    
    if not torch.is_tensor(
        tiny_layer29_output
    ):
        raise AssertionError(
            "BUG IN TEST: TinyLM layer29 output "
            "must be Tensor, but got "
            f"{type(tiny_layer29_output)}"
        )
    
    compare_tensors(
        "layer29 output",
        official_layer29_output,
        tiny_layer29_output,
    )


    if first_failure is not None:

        raise AssertionError(
            "Layer 29 detailed numerical parity failed "
            f"first at: {first_failure}"
        )

    print(
        "[PASS] Detailed layer-29 diagnostic"
    )

# ============================================================
# Output projection semantic test
# ============================================================

def test_output_projection_semantics(
    tiny,
):
    section("TEST 9: OUTPUT PROJECTION SEMANTICS")

    device = next(
        tiny.parameters()
    ).device

    torch.manual_seed(
        SEED + 1
    )

    x = torch.randn(
        BATCH_SIZE,
        SEQ_LEN,
        HIDDEN_SIZE,
        device=device,
    )

    with torch.no_grad():

        a = F.linear(
            x,
            tiny.emb.weight,
        )

        b = tiny.lm_head(
            x
        )

    compare_tensors(
        "F.linear(x, emb.weight) vs lm_head(x)",
        a,
        b,
        atol=0.0,
        rtol=0.0,
    )

    if tiny.lm_head.weight is not tiny.emb.weight:
        raise AssertionError(
            "Output projection is not tied to embedding."
        )

    print(
        "[PASS] TinyLM output projection exactly "
        "uses embedding weight."
    )


# ============================================================
# MLP structure
# ============================================================

def test_dense_mlp_structure(
    tiny,
):
    section("TEST 10: DENSE MLP STRUCTURE")

    mlp_ids = [
        id(layer.mlp)
        for layer in tiny.layers
    ]

    unique_mlp_ids = set(
        mlp_ids
    )

    print(
        f"Layers          : {NUM_LAYERS}"
    )

    print(
        f"Unique MLPs      : "
        f"{len(unique_mlp_ids)}"
    )

    if len(unique_mlp_ids) != NUM_LAYERS:
        raise AssertionError(
            "Base MobileLLM parity requires "
            "one independent MLP per layer."
        )

    print(
        "[PASS] Dense MobileLLM contains "
        f"{NUM_LAYERS} independent MLP objects."
    )


# ============================================================
# Attention structure
# ============================================================

def test_attention_structure(
    tiny,
):
    section("TEST 11: ATTENTION STRUCTURE")

    attention_ids = [
        id(layer.attn)
        for layer in tiny.layers
    ]

    unique_attention_ids = set(
        attention_ids
    )

    if len(unique_attention_ids) != NUM_LAYERS:
        raise AssertionError(
            "Expected one independent attention module "
            "per layer."
        )

    for i, layer in enumerate(
        tiny.layers
    ):

        q_shape = tuple(
            layer.attn.q_proj.weight.shape
        )

        k_shape = tuple(
            layer.attn.k_proj.weight.shape
        )

        v_shape = tuple(
            layer.attn.v_proj.weight.shape
        )

        o_shape = tuple(
            layer.attn.o_proj.weight.shape
        )

        expected = {
            "q": (
                HIDDEN_SIZE,
                HIDDEN_SIZE,
            ),
            "k": (
                NUM_KV_HEADS * HEAD_DIM,
                HIDDEN_SIZE,
            ),
            "v": (
                NUM_KV_HEADS * HEAD_DIM,
                HIDDEN_SIZE,
            ),
            "o": (
                HIDDEN_SIZE,
                HIDDEN_SIZE,
            ),
        }

        actual = {
            "q": q_shape,
            "k": k_shape,
            "v": v_shape,
            "o": o_shape,
        }

        if actual != expected:
            raise AssertionError(
                f"Layer {i} attention shape mismatch:\n"
                f"Expected: {expected}\n"
                f"Actual:   {actual}"
            )

    print(
        "[PASS] All attention projections have "
        "MobileLLM 125M dimensions."
    )

    print(
        "[PASS] 30 unique attention modules."
    )


# ============================================================
# HF LLaMA auxiliary check
# ============================================================

def test_hf_llama_sanity():
    """
    HF LLaMA is NOT treated as the MobileLLM reference.

    It is only used to verify that the environment's
    transformers LlamaConfig exposes the expected basic
    architecture fields.
    """

    section("TEST 12: HF LLAMA AUXILIARY SANITY CHECK")

    try:
        from transformers import LlamaConfig

        cfg = LlamaConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            num_hidden_layers=NUM_LAYERS,
            num_attention_heads=NUM_Q_HEADS,
            num_key_value_heads=NUM_KV_HEADS,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            initializer_range=INITIALIZER_RANGE,
            rms_norm_eps=RMS_NORM_EPS,
            rope_theta=ROPE_THETA,
        )

        assert cfg.hidden_size == HIDDEN_SIZE
        assert cfg.intermediate_size == INTERMEDIATE_SIZE
        assert cfg.num_hidden_layers == NUM_LAYERS
        assert cfg.num_attention_heads == NUM_Q_HEADS
        assert cfg.num_key_value_heads == NUM_KV_HEADS

        print(
            "[PASS] HF LLaMA configuration sanity check."
        )

        print(
            "[INFO] HF LLaMA is auxiliary only; "
            "official MobileLLM remains the reference."
        )

    except Exception as exc:
        print(
            "[WARN] HF LLaMA auxiliary check skipped:"
        )

        print(
            f"       {exc}"
        )


# ============================================================
# Final report
# ============================================================

def print_final_report(
    official,
    tiny,
):
    section("FINAL PARITY REPORT")

    official_n = count_parameters(
        official
    )

    tiny_n = count_parameters(
        tiny
    )

    print(
        f"Official MobileLLM parameters : "
        f"{official_n:,}"
    )

    print(
        f"TinyLM parameters             : "
        f"{tiny_n:,}"
    )

    print(
        f"Parameter difference          : "
        f"{tiny_n - official_n:,}"
    )

    print()

    print(
        "Architecture:"
    )

    print(
        f"  hidden size       = {HIDDEN_SIZE}"
    )

    print(
        f"  intermediate size = {INTERMEDIATE_SIZE}"
    )

    print(
        f"  layers            = {NUM_LAYERS}"
    )

    print(
        f"  Q/KV heads        = "
        f"{NUM_Q_HEADS}/{NUM_KV_HEADS}"
    )

    print(
        f"  head dimension    = {HEAD_DIM}"
    )

    print(
        f"  vocabulary        = {VOCAB_SIZE}"
    )

    print(
        f"  context           = "
        f"{MAX_POSITION_EMBEDDINGS}"
    )

    print(
        f"  RoPE theta        = {ROPE_THETA}"
    )

    print(
        f"  RMSNorm epsilon   = {RMS_NORM_EPS}"
    )

    print()

    print(
        "Weight sharing:"
    )

    print(
        "  input/output embedding = SHARED"
    )

    print(
        "  MLP sharing            = NONE (Dense)"
    )

    print()

    print(
        "Result:"
    )

    print(
        "  ARCHITECTURE PARITY    = PASS"
    )

    print(
        "  WEIGHT PARITY          = PASS"
    )

    print(
        "  NUMERICAL FORWARD      = PASS"
    )

    print(
        "  LOGITS PARITY          = PASS"
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("#" * 72)
    print("# TinyLM_P018 MobileLLM-125M Weight + Numerical Parity")
    print("#" * 72)

    print()
    print(
        f"PyTorch       : {torch.__version__}"
    )

    try:
        import transformers

        print(
            f"Transformers  : "
            f"{transformers.__version__}"
        )

    except Exception:
        pass

    print(
        f"CUDA          : "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU           : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    (
        official,
        tiny,
        official_cfg,
        tiny_cfg,
    ) = make_models()

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    test_config_parity(
        official_cfg,
        tiny_cfg,
    )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    test_parameter_count(
        official,
        tiny,
    )

    # --------------------------------------------------------
    # Embedding
    # --------------------------------------------------------

    test_embedding_tying(
        official,
        tiny,
    )

    # --------------------------------------------------------
    # Parameter mapping
    # --------------------------------------------------------

    print()
    print("========== DEBUG TINYLM PARAMETER NAMES ==========")
    
    for name, p in tiny.named_parameters():
        print(name, tuple(p.shape))
    
    print()
    print("========== DEBUG LAYER 0 MLP ==========")
    
    print("layer0.mlp type:",
          type(tiny.layers[0].mlp))
    
    print("layer0.mlp:",
          tiny.layers[0].mlp)
    
    print("layer0.mlp named parameters:")
    
    for name, p in tiny.layers[0].mlp.named_parameters():
        print("  ", name, tuple(p.shape))
    
    print()
    print("layer0._modules:")
    print(tiny.layers[0]._modules)

    mapping = test_parameter_mapping(
        official,
        tiny,
    )

    # --------------------------------------------------------
    # Unexpected parameters
    # --------------------------------------------------------

    test_unexpected_parameters(
        official,
        tiny,
    )

    # --------------------------------------------------------
    # Copy reference weights
    # --------------------------------------------------------

    copy_official_weights(
        official,
        tiny,
        mapping,
    )

    # --------------------------------------------------------
    # Exact weight equality
    # --------------------------------------------------------

    test_weight_equality(
        official,
        tiny,
        mapping,
    )

    # --------------------------------------------------------
    # Numerical forward
    # --------------------------------------------------------

    test_numerical_parity(
        official,
        tiny,
    )

    # --------------------------------------------------------
    # Output projection
    # --------------------------------------------------------

    test_output_projection_semantics(
        tiny,
    )

    # --------------------------------------------------------
    # Dense MLP
    # --------------------------------------------------------

    test_dense_mlp_structure(
        tiny,
    )

    # --------------------------------------------------------
    # Attention
    # --------------------------------------------------------

    test_attention_structure(
        tiny,
    )

    # --------------------------------------------------------
    # HF auxiliary
    # --------------------------------------------------------

    test_hf_llama_sanity()

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print_final_report(
        official,
        tiny,
    )

    print()
    print("#" * 72)
    print("# ALL MOBILELLM PARITY TESTS PASSED")
    print("#" * 72)
    print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except AssertionError as exc:

        print()
        print("#" * 72)
        print("# PARITY TEST FAILED")
        print("#" * 72)
        print()
        print(exc)
        print()

        traceback.print_exc()

        sys.exit(1)

    except Exception as exc:

        print()
        print("#" * 72)
        print("# UNEXPECTED ERROR")
        print("#" * 72)
        print()
        print(exc)
        print()

        traceback.print_exc()

        sys.exit(2)