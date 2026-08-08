"""
TinyLM_P018 MobileLLM 125M architecture smoke + parity test.

Run from the repository root:

    python test/test_mobile.py

This test does NOT train the model.

It verifies:

1. MobileLLM 125M configuration
2. MobileLLM 125M Dense construction
3. MobileLLM 125M + MLP-sharing construction
4. parameter registration
5. MLP sharing identity
6. attention non-sharing
7. MobileLLM architecture invariants
8. MobileLLM structural parity
9. Dense/Shared architecture parity
10. forward pass
11. logits shape
12. causal LM loss
13. backward pass
14. NaN / Inf detection
15. shared MLP gradient flow
16. state_dict registration
17. untied embedding / LM-head
18. existing P018 model construction


============================================================
MobileLLM 125M reference
============================================================

    vocab_size              = 32000
    hidden_size             = 576
    intermediate_size       = 1536
    num_hidden_layers       = 30
    num_attention_heads    = 9
    num_key_value_heads     = 3
    head_dim                = 64

    max_position_embeddings = 2048
    rope_theta              = 10000
    rms_norm_eps            = 1e-5

    initializer_range       = 0.02
    tie_word_embeddings     = False
    bias                    = False

P018-R1 Mobile Shared variant:

    same MobileLLM architecture
    tie_mlp = True
    mlp_group = 6

30 layers / group 6 = 5 unique MLP objects

MLP sharing:

    L0-L5       -> MLP 0
    L6-L11      -> MLP 1
    L12-L17     -> MLP 2
    L18-L23     -> MLP 3
    L24-L29     -> MLP 4

Attention is NEVER shared.
Embedding and LM-head are NEVER shared.
"""


from __future__ import annotations

import math
import sys
import traceback

import torch


# ============================================================
# Repository imports
# ============================================================

try:
    from tinylm.config import build_config

except Exception as exc:
    print("[ERROR] Cannot import tinylm.config")
    print(exc)
    sys.exit(1)


# ------------------------------------------------------------
# transformer.py compatibility
# ------------------------------------------------------------

build_model = None
MobileTransformer = None
TiedMLPTransformer = None

try:

    from tinylm.model.transformer import (
        build_model,
        MobileTransformer,
        TiedMLPTransformer,
    )

except ImportError:

    try:

        from tinylm.model.transformer import (
            build_model,
            MobileTransformer,
        )

    except ImportError:

        try:

            from tinylm.model.transformer import (
                MobileTransformer,
                TiedMLPTransformer,
            )

        except ImportError as exc:

            print("[ERROR] Cannot import model classes.")
            print(exc)
            sys.exit(1)


# ============================================================
# Test constants
# ============================================================

SEQ_LEN = 32
BATCH_SIZE = 2
SEED = 1234


# ============================================================
# Actual MobileLLM 125M reference values
# ============================================================

MOBILE_VOCAB_SIZE = 32000

MOBILE_DIM = 576
MOBILE_FFN_DIM = 1536

MOBILE_LAYERS = 30

MOBILE_Q_HEADS = 9
MOBILE_KV_HEADS = 3
MOBILE_HEAD_DIM = 64

MOBILE_MAX_SEQ_LEN = 2048
MOBILE_ROPE_THETA = 10000.0
MOBILE_NORM_EPS = 1e-5
MOBILE_INITIALIZER_RANGE = 0.02

MOBILE_TIE_WORD_EMBEDDINGS = False


# ------------------------------------------------------------
# P018-R1 intervention
# ------------------------------------------------------------

MOBILE_MLP_GROUP = 6

MOBILE_EXPECTED_SHARED_MLPS = (
    MOBILE_LAYERS // MOBILE_MLP_GROUP
)


# ============================================================
# Utility
# ============================================================

def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )


def count_trainable_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def tensor_has_nan_or_inf(x):
    return not torch.isfinite(x).all().item()


def get_layer_mlp(layer):

    if not hasattr(layer, "mlp"):
        raise AssertionError(
            "Mobile layer has no '.mlp' attribute."
        )

    return layer.mlp


def get_layer_attention(layer):

    if not hasattr(layer, "attn"):
        raise AssertionError(
            "Mobile layer has no '.attn' attribute."
        )

    return layer.attn


def get_logits(output):
    """
    Normalize supported model output forms:

        Tensor
        tuple
        dict
    """

    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, tuple):

        if not output:
            raise AssertionError(
                "Model returned an empty tuple."
            )

        return output[0]

    if isinstance(output, dict):

        if "logits" in output:
            return output["logits"]

        if not output:
            raise AssertionError(
                "Model returned an empty dict."
            )

        return next(iter(output.values()))

    raise AssertionError(
        f"Unknown model output type: "
        f"{type(output)}"
    )


def print_model_summary(name, model, cfg):

    total = count_parameters(model)
    trainable = count_trainable_parameters(model)

    print()
    print("=" * 72)
    print(name)
    print("=" * 72)

    print(
        f"model_family : "
        f"{getattr(cfg, 'model_family', '<missing>')}"
    )

    print(
        f"vocab_size   : {cfg.vocab_size}"
    )

    print(
        f"dim          : {cfg.dim}"
    )

    print(
        f"ffn_dim      : {cfg.ffn_dim}"
    )

    print(
        f"layers       : "
        f"{getattr(cfg, 'n_layers', '<missing>')}"
    )

    print(
        f"q_heads      : {cfg.n_q_heads}"
    )

    print(
        f"kv_heads     : {cfg.n_kv_heads}"
    )

    print(
        f"head_dim     : "
        f"{cfg.dim // cfg.n_q_heads}"
    )

    print(
        f"mlp_group    : "
        f"{getattr(cfg, 'mlp_group', '<missing>')}"
    )

    print(
        f"tie_mlp      : "
        f"{getattr(cfg, 'tie_mlp', '<missing>')}"
    )

    print(
        f"tie_word_embeddings : "
        f"{getattr(cfg, 'tie_word_embeddings', '<missing>')}"
    )

    print()

    print(
        f"parameters   : {total:,}"
    )

    print(
        f"trainable    : {trainable:,}"
    )

    print(
        f"parameters M : {total / 1e6:.3f}"
    )


def construct_model(cfg):
    """
    Preferred:

        build_model(cfg)

    Fallback:

        MobileTransformer(cfg)

    Existing P018 configs may use:

        TiedMLPTransformer(cfg)
    """

    if build_model is not None:
        return build_model(cfg)

    if getattr(cfg, "model_family", None) == "mobile":

        if MobileTransformer is None:
            raise RuntimeError(
                "MobileTransformer is not available."
            )

        return MobileTransformer(cfg)

    if TiedMLPTransformer is not None:
        return TiedMLPTransformer(cfg)

    raise RuntimeError(
        "No model construction path available."
    )


# ============================================================
# Test 1
# Exact MobileLLM 125M configuration
# ============================================================

def test_config():

    print()
    print("=" * 72)
    print("TEST 1: MobileLLM 125M configuration")
    print("=" * 72)

    cfg = build_config(
        "mobile125",
        "dense",
        SEQ_LEN,
        False,
    )

    # --------------------------------------------------------
    # Model family
    # --------------------------------------------------------

    assert getattr(
        cfg,
        "model_family",
        None,
    ) == "mobile", (
        "mobile125 config does not have "
        "model_family='mobile'."
    )

    # --------------------------------------------------------
    # Exact architecture
    # --------------------------------------------------------

    assert cfg.vocab_size == MOBILE_VOCAB_SIZE, (
        f"Expected vocab_size={MOBILE_VOCAB_SIZE}, "
        f"got {cfg.vocab_size}"
    )

    assert cfg.dim == MOBILE_DIM, (
        f"Expected dim={MOBILE_DIM}, "
        f"got {cfg.dim}"
    )

    assert cfg.ffn_dim == MOBILE_FFN_DIM, (
        f"Expected ffn_dim={MOBILE_FFN_DIM}, "
        f"got {cfg.ffn_dim}"
    )

    assert cfg.n_layers == MOBILE_LAYERS, (
        f"Expected n_layers={MOBILE_LAYERS}, "
        f"got {cfg.n_layers}"
    )

    assert cfg.n_q_heads == MOBILE_Q_HEADS, (
        f"Expected n_q_heads={MOBILE_Q_HEADS}, "
        f"got {cfg.n_q_heads}"
    )

    assert cfg.n_kv_heads == MOBILE_KV_HEADS, (
        f"Expected n_kv_heads={MOBILE_KV_HEADS}, "
        f"got {cfg.n_kv_heads}"
    )

    assert (
        cfg.dim // cfg.n_q_heads
        == MOBILE_HEAD_DIM
    ), (
        "MobileLLM head dimension mismatch: "
        f"expected {MOBILE_HEAD_DIM}, "
        f"got {cfg.dim // cfg.n_q_heads}"
    )

    assert (
        cfg.n_q_heads % cfg.n_kv_heads == 0
    ), (
        "Q heads must be divisible by KV heads."
    )

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    assert cfg.max_seq_len == MOBILE_MAX_SEQ_LEN, (
        f"Expected max_seq_len={MOBILE_MAX_SEQ_LEN}, "
        f"got {cfg.max_seq_len}"
    )

    # --------------------------------------------------------
    # RoPE
    # --------------------------------------------------------

    assert math.isclose(
        cfg.rope_theta,
        MOBILE_ROPE_THETA,
        rel_tol=0.0,
        abs_tol=1e-6,
    ), (
        f"Expected rope_theta={MOBILE_ROPE_THETA}, "
        f"got {cfg.rope_theta}"
    )

    # --------------------------------------------------------
    # RMSNorm
    # --------------------------------------------------------

    assert math.isclose(
        cfg.norm_eps,
        MOBILE_NORM_EPS,
        rel_tol=0.0,
        abs_tol=1e-12,
    ), (
        f"Expected norm_eps={MOBILE_NORM_EPS}, "
        f"got {cfg.norm_eps}"
    )

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------

    assert math.isclose(
        cfg.initializer_range,
        MOBILE_INITIALIZER_RANGE,
        rel_tol=0.0,
        abs_tol=1e-12,
    ), (
        "initializer_range mismatch: "
        f"expected {MOBILE_INITIALIZER_RANGE}, "
        f"got {cfg.initializer_range}"
    )

    # --------------------------------------------------------
    # Untied embedding / LM head
    # --------------------------------------------------------

    assert (
        cfg.tie_word_embeddings
        is MOBILE_TIE_WORD_EMBEDDINGS
    ), (
        "MobileLLM 125M requires "
        "tie_word_embeddings=False."
    )

    # --------------------------------------------------------
    # Dense reference
    # --------------------------------------------------------

    assert cfg.tie_mlp is False, (
        "Dense MobileLLM reference must not "
        "enable MLP sharing."
    )

    assert cfg.mlp_group == 1, (
        f"Dense MobileLLM must have mlp_group=1, "
        f"got {cfg.mlp_group}"
    )

    print(
        "[PASS] MobileLLM 125M config is valid."
    )

    print()
    print("Reference architecture:")

    print(
        f"  vocab       = {cfg.vocab_size}"
    )

    print(
        f"  hidden      = {cfg.dim}"
    )

    print(
        f"  FFN         = {cfg.ffn_dim}"
    )

    print(
        f"  layers      = {cfg.n_layers}"
    )

    print(
        f"  Q/KV heads  = "
        f"{cfg.n_q_heads}/{cfg.n_kv_heads}"
    )

    print(
        f"  head dim    = "
        f"{cfg.dim // cfg.n_q_heads}"
    )

    print(
        f"  context     = {cfg.max_seq_len}"
    )

    print(
        f"  RoPE theta  = {cfg.rope_theta}"
    )

    print(
        f"  norm eps    = {cfg.norm_eps}"
    )

    print(
        f"  init std    = "
        f"{cfg.initializer_range}"
    )

    print(
        f"  tied output = "
        f"{cfg.tie_word_embeddings}"
    )

    return cfg


# ============================================================
# Test 2
# MobileLLM Dense construction
# ============================================================

def test_mobile_dense():

    print()
    print("=" * 72)
    print("TEST 2: MobileLLM 125M Dense construction")
    print("=" * 72)

    cfg = build_config(
        "mobile125",
        "dense",
        SEQ_LEN,
        False,
    )

    model = construct_model(cfg)

    print_model_summary(
        "MobileLLM 125M Dense",
        model,
        cfg,
    )

    assert len(model.layers) == cfg.n_layers, (
        f"Expected {cfg.n_layers} layers, "
        f"got {len(model.layers)}"
    )

    # --------------------------------------------------------
    # Dense = one MLP per layer
    # --------------------------------------------------------

    mlps = [
        get_layer_mlp(layer)
        for layer in model.layers
    ]

    unique_mlp_ids = len({
        id(mlp)
        for mlp in mlps
    })

    assert unique_mlp_ids == len(mlps), (
        "Mobile Dense unexpectedly shares MLP objects."
    )

    print(
        f"[PASS] {len(mlps)} layers have "
        f"{unique_mlp_ids} unique MLP objects."
    )

    # --------------------------------------------------------
    # Attention = one per layer
    # --------------------------------------------------------

    attentions = [
        get_layer_attention(layer)
        for layer in model.layers
    ]

    unique_attention_ids = len({
        id(attn)
        for attn in attentions
    })

    assert unique_attention_ids == len(attentions), (
        "Mobile Dense attention modules are shared."
    )

    print(
        f"[PASS] {len(attentions)} layers have "
        f"{unique_attention_ids} unique "
        f"attention modules."
    )

    return cfg, model


# ============================================================
# Test 3
# MobileLLM + MLP sharing
# ============================================================

def test_mobile_shared():

    print()
    print("=" * 72)
    print("TEST 3: MobileLLM + MLP-sharing construction")
    print("=" * 72)

    cfg = build_config(
        "mobile125",
        "tied",
        SEQ_LEN,
        False,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The config.py preset itself should now define:
    #
    #     tie_mlp=True
    #     mlp_group=6
    #
    # We deliberately DO NOT silently replace the value here.
    #
    # This makes the test verify the actual preset.
    # --------------------------------------------------------

    assert cfg.tie_mlp is True, (
        "mobile125/tied preset did not enable "
        "tie_mlp=True."
    )

    assert cfg.mlp_group == MOBILE_MLP_GROUP, (
        f"mobile125/tied preset must use "
        f"mlp_group={MOBILE_MLP_GROUP}, "
        f"got {cfg.mlp_group}"
    )

    assert (
        MOBILE_LAYERS % MOBILE_MLP_GROUP == 0
    ), (
        "MobileLLM layer count must be divisible "
        "by the selected MLP sharing group."
    )

    model = construct_model(cfg)

    print_model_summary(
        "MobileLLM 125M + MLP sharing (g6)",
        model,
        cfg,
    )

    assert len(model.layers) == MOBILE_LAYERS

    mlps = [
        get_layer_mlp(layer)
        for layer in model.layers
    ]

    # --------------------------------------------------------
    # Check every group
    #
    # L0-L5
    # L6-L11
    # L12-L17
    # L18-L23
    # L24-L29
    # --------------------------------------------------------

    for start in range(
        0,
        cfg.n_layers,
        cfg.mlp_group,
    ):

        end = min(
            start + cfg.mlp_group,
            cfg.n_layers,
        )

        group_ids = {
            id(mlps[i])
            for i in range(start, end)
        }

        assert len(group_ids) == 1, (
            f"Layers {start}-{end - 1} "
            "do not share one MLP object."
        )

    # --------------------------------------------------------
    # Different groups must be independent.
    # --------------------------------------------------------

    group_roots = []

    for start in range(
        0,
        cfg.n_layers,
        cfg.mlp_group,
    ):

        group_roots.append(
            id(mlps[start])
        )

    assert (
        len(group_roots)
        == len(set(group_roots))
    ), (
        "Different MLP groups unexpectedly "
        "share the same MLP object."
    )

    # --------------------------------------------------------
    # Exact number of unique MLP objects
    # --------------------------------------------------------

    expected_groups = (
        MOBILE_EXPECTED_SHARED_MLPS
    )

    actual_groups = len({
        id(mlp)
        for mlp in mlps
    })

    assert actual_groups == expected_groups, (
        f"Expected {expected_groups} unique MLPs, "
        f"got {actual_groups}"
    )

    print(
        f"[PASS] {cfg.n_layers} layers use "
        f"{actual_groups} shared MLPs."
    )

    # --------------------------------------------------------
    # Print mapping
    # --------------------------------------------------------

    print()
    print("MLP sharing map:")

    for group_idx, start in enumerate(
        range(
            0,
            cfg.n_layers,
            cfg.mlp_group,
        )
    ):

        end = min(
            start + cfg.mlp_group - 1,
            cfg.n_layers - 1,
        )

        print(
            f"  MLP {group_idx}: "
            f"L{start}-L{end}"
        )

    return cfg, model


# ============================================================
# Test 4
# Attention must NEVER be shared
# ============================================================

def test_attention_not_shared(model):

    print()
    print("=" * 72)
    print("TEST 4: Attention non-sharing")
    print("=" * 72)

    attentions = [
        get_layer_attention(layer)
        for layer in model.layers
    ]

    unique_attention_ids = len({
        id(attn)
        for attn in attentions
    })

    assert unique_attention_ids == len(attentions), (
        "Attention modules are unexpectedly shared."
    )

    print(
        f"[PASS] {len(attentions)} layers have "
        f"{unique_attention_ids} unique "
        f"attention modules."
    )


# ============================================================
# Test 5
# Embedding / LM-head must be untied
# ============================================================

def test_embedding_lm_head_not_tied(model):

    print()
    print("=" * 72)
    print("TEST 5: Embedding / LM-head non-sharing")
    print("=" * 72)

    assert hasattr(model, "emb"), (
        "Mobile model has no '.emb'."
    )

    assert hasattr(model, "lm_head"), (
        "Mobile model has no separate '.lm_head'. "
        "MobileLLM 125M requires an untied output head."
    )

    emb = model.emb
    lm_head = model.lm_head

    assert emb.weight is not lm_head.weight, (
        "Embedding and LM-head unexpectedly share "
        "the same Parameter object."
    )

    assert (
        emb.weight.data_ptr()
        != lm_head.weight.data_ptr()
    ), (
        "Embedding and LM-head unexpectedly "
        "share the same storage."
    )

    assert emb.weight.shape == (
        MOBILE_VOCAB_SIZE,
        MOBILE_DIM,
    ), (
        f"Unexpected embedding shape: "
        f"{tuple(emb.weight.shape)}"
    )

    assert lm_head.weight.shape == (
        MOBILE_VOCAB_SIZE,
        MOBILE_DIM,
    ), (
        f"Unexpected LM-head shape: "
        f"{tuple(lm_head.weight.shape)}"
    )

    print(
        "[PASS] input embedding and LM-head "
        "are separate parameters."
    )

    print(
        f"embedding shape = "
        f"{tuple(emb.weight.shape)}"
    )

    print(
        f"lm_head shape   = "
        f"{tuple(lm_head.weight.shape)}"
    )


# ============================================================
# Test 6
# MobileLLM configuration parity
# ============================================================

def test_mobile_config_parity(cfg):

    print()
    print("=" * 72)
    print("TEST 6: MobileLLM configuration parity")
    print("=" * 72)

    checks = {
        "vocab_size": (
            cfg.vocab_size,
            MOBILE_VOCAB_SIZE,
        ),
        "dim": (
            cfg.dim,
            MOBILE_DIM,
        ),
        "ffn_dim": (
            cfg.ffn_dim,
            MOBILE_FFN_DIM,
        ),
        "n_layers": (
            cfg.n_layers,
            MOBILE_LAYERS,
        ),
        "n_q_heads": (
            cfg.n_q_heads,
            MOBILE_Q_HEADS,
        ),
        "n_kv_heads": (
            cfg.n_kv_heads,
            MOBILE_KV_HEADS,
        ),
        "max_seq_len": (
            cfg.max_seq_len,
            MOBILE_MAX_SEQ_LEN,
        ),
    }

    for name, (actual, expected) in checks.items():

        assert actual == expected, (
            f"Parity failure: {name}: "
            f"expected {expected}, got {actual}"
        )

    assert (
        cfg.dim // cfg.n_q_heads
        == MOBILE_HEAD_DIM
    )

    assert (
        cfg.n_q_heads // cfg.n_kv_heads
        == 3
    ), (
        "MobileLLM 125M should use "
        "3:1 Q-head/KV-head ratio."
    )

    assert math.isclose(
        cfg.rope_theta,
        MOBILE_ROPE_THETA,
        rel_tol=0.0,
        abs_tol=1e-6,
    )

    assert math.isclose(
        cfg.norm_eps,
        MOBILE_NORM_EPS,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    assert math.isclose(
        cfg.initializer_range,
        MOBILE_INITIALIZER_RANGE,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    assert cfg.tie_word_embeddings is False

    print(
        "[PASS] TinyLM Mobile config matches "
        "the MobileLLM 125M reference."
    )

    print()
    print("Parity summary:")

    print(
        "  hidden size       : "
        f"{cfg.dim} == {MOBILE_DIM}"
    )

    print(
        "  FFN size          : "
        f"{cfg.ffn_dim} == {MOBILE_FFN_DIM}"
    )

    print(
        "  layers            : "
        f"{cfg.n_layers} == {MOBILE_LAYERS}"
    )

    print(
        "  attention heads   : "
        f"{cfg.n_q_heads} == {MOBILE_Q_HEADS}"
    )

    print(
        "  KV heads          : "
        f"{cfg.n_kv_heads} == {MOBILE_KV_HEADS}"
    )

    print(
        "  head dimension    : "
        f"{cfg.dim // cfg.n_q_heads} "
        f"== {MOBILE_HEAD_DIM}"
    )

    print(
        "  context           : "
        f"{cfg.max_seq_len} "
        f"== {MOBILE_MAX_SEQ_LEN}"
    )

    print(
        "  RoPE theta        : "
        f"{cfg.rope_theta} "
        f"== {MOBILE_ROPE_THETA}"
    )

    print(
        "  RMSNorm eps       : "
        f"{cfg.norm_eps} "
        f"== {MOBILE_NORM_EPS}"
    )

    print(
        "  initializer range : "
        f"{cfg.initializer_range} "
        f"== {MOBILE_INITIALIZER_RANGE}"
    )

    print(
        "  tied embeddings   : "
        f"{cfg.tie_word_embeddings}"
    )


# ============================================================
# Test 7
# Dense / Shared architectural parity
# ============================================================

def test_dense_shared_parity(
    dense_cfg,
    dense_model,
    shared_cfg,
    shared_model,
):

    print()
    print("=" * 72)
    print("TEST 7: Dense / Shared architecture parity")
    print("=" * 72)

    # --------------------------------------------------------
    # Config dimensions must be identical.
    # --------------------------------------------------------

    fields = [
        "vocab_size",
        "dim",
        "ffn_dim",
        "n_q_heads",
        "n_kv_heads",
        "max_seq_len",
        "rope_theta",
        "norm_eps",
        "initializer_range",
        "tie_word_embeddings",
    ]

    for field in fields:

        dense_value = getattr(
            dense_cfg,
            field,
        )

        shared_value = getattr(
            shared_cfg,
            field,
        )

        assert dense_value == shared_value, (
            f"Dense/Shared parity failure: "
            f"{field}: "
            f"dense={dense_value}, "
            f"shared={shared_value}"
        )

    assert dense_cfg.tie_mlp is False
    assert shared_cfg.tie_mlp is True

    assert dense_cfg.mlp_group == 1
    assert shared_cfg.mlp_group == MOBILE_MLP_GROUP

    # --------------------------------------------------------
    # Layer count
    # --------------------------------------------------------

    assert (
        len(dense_model.layers)
        == len(shared_model.layers)
        == MOBILE_LAYERS
    )

    # --------------------------------------------------------
    # Attention objects remain independent in both models.
    # --------------------------------------------------------

    dense_attn = [
        get_layer_attention(layer)
        for layer in dense_model.layers
    ]

    shared_attn = [
        get_layer_attention(layer)
        for layer in shared_model.layers
    ]

    assert len({
        id(x)
        for x in dense_attn
    }) == MOBILE_LAYERS

    assert len({
        id(x)
        for x in shared_attn
    }) == MOBILE_LAYERS

    # --------------------------------------------------------
    # Dense has 30 MLPs.
    # Shared has 5 MLPs.
    # --------------------------------------------------------

    dense_mlps = [
        get_layer_mlp(layer)
        for layer in dense_model.layers
    ]

    shared_mlps = [
        get_layer_mlp(layer)
        for layer in shared_model.layers
    ]

    dense_unique = len({
        id(x)
        for x in dense_mlps
    })

    shared_unique = len({
        id(x)
        for x in shared_mlps
    })

    assert dense_unique == MOBILE_LAYERS

    assert shared_unique == MOBILE_EXPECTED_SHARED_MLPS

    print(
        "[PASS] Dense and Shared preserve "
        "the same MobileLLM base architecture."
    )

    print(
        f"  Dense unique MLPs  : {dense_unique}"
    )

    print(
        f"  Shared unique MLPs : {shared_unique}"
    )

    print(
        f"  Attention modules   : "
        f"{MOBILE_LAYERS} / {MOBILE_LAYERS}"
    )

    print(
        "  Difference          : "
        "MLP sharing only"
    )


# ============================================================
# Test 8
# Structural parity
#
# Verify major tensor dimensions without depending on
# implementation-specific parameter names.
# ============================================================

def test_structural_parity(model):

    print()
    print("=" * 72)
    print("TEST 8: MobileLLM structural parity")
    print("=" * 72)

    # --------------------------------------------------------
    # Embedding
    # --------------------------------------------------------

    assert hasattr(model, "emb")

    assert tuple(model.emb.weight.shape) == (
        MOBILE_VOCAB_SIZE,
        MOBILE_DIM,
    )

    # --------------------------------------------------------
    # LM head
    # --------------------------------------------------------

    assert hasattr(model, "lm_head")

    assert tuple(model.lm_head.weight.shape) == (
        MOBILE_VOCAB_SIZE,
        MOBILE_DIM,
    )

    # --------------------------------------------------------
    # Layer count
    # --------------------------------------------------------

    assert len(model.layers) == MOBILE_LAYERS

    # --------------------------------------------------------
    # Attention structural checks.
    #
    # These checks intentionally inspect parameter shapes
    # rather than relying on exact class names.
    # --------------------------------------------------------

    first_attn = get_layer_attention(
        model.layers[0]
    )

    attention_parameter_shapes = {
        name: tuple(param.shape)
        for name, param
        in first_attn.named_parameters()
    }

    assert attention_parameter_shapes, (
        "Attention module contains no parameters."
    )

    # Search for projection dimensions.
    #
    # Expected:
    #
    # Q projection:
    #     9 * 64 = 576
    #
    # K/V projection:
    #     3 * 64 = 192
    #
    # Output:
    #     576 -> 576
    #
    found_576 = False
    found_192 = False

    for shape in attention_parameter_shapes.values():

        if len(shape) != 2:
            continue

        out_dim, in_dim = shape

        if (
            out_dim == MOBILE_DIM
            and in_dim == MOBILE_DIM
        ):
            found_576 = True

        if (
            out_dim == MOBILE_KV_HEADS * MOBILE_HEAD_DIM
            and in_dim == MOBILE_DIM
        ):
            found_192 = True

    assert found_576, (
        "Could not find expected 576-dimensional "
        "attention projection."
    )

    assert found_192, (
        "Could not find expected 192-dimensional "
        "K/V projection."
    )

    # --------------------------------------------------------
    # MLP structural checks
    # --------------------------------------------------------

    first_mlp = get_layer_mlp(
        model.layers[0]
    )

    mlp_parameter_shapes = {
        name: tuple(param.shape)
        for name, param
        in first_mlp.named_parameters()
    }

    assert mlp_parameter_shapes, (
        "MLP module contains no parameters."
    )

    found_ffn = False

    for shape in mlp_parameter_shapes.values():

        if len(shape) != 2:
            continue

        out_dim, in_dim = shape

        if (
            out_dim == MOBILE_FFN_DIM
            and in_dim == MOBILE_DIM
        ):
            found_ffn = True
            break

    assert found_ffn, (
        "Could not find expected "
        "1536 x 576 MLP projection."
    )

    print(
        "[PASS] MobileLLM structural dimensions "
        "are present in the instantiated model."
    )

    print(
        f"  embedding : "
        f"{tuple(model.emb.weight.shape)}"
    )

    print(
        f"  LM head   : "
        f"{tuple(model.lm_head.weight.shape)}"
    )

    print(
        f"  Q dim     : "
        f"{MOBILE_Q_HEADS * MOBILE_HEAD_DIM}"
    )

    print(
        f"  KV dim    : "
        f"{MOBILE_KV_HEADS * MOBILE_HEAD_DIM}"
    )

    print(
        f"  FFN dim   : "
        f"{MOBILE_FFN_DIM}"
    )


# ============================================================
# Test 9
# Forward
# ============================================================

def run_forward(model, cfg, name):

    print()
    print("=" * 72)
    print(f"TEST 9: Forward - {name}")
    print("=" * 72)

    device = next(
        model.parameters()
    ).device

    torch.manual_seed(SEED)

    tokens = torch.randint(
        low=0,
        high=cfg.vocab_size,
        size=(
            BATCH_SIZE,
            SEQ_LEN,
        ),
        device=device,
        dtype=torch.long,
    )

    model.train()

    output = model(tokens)

    logits = get_logits(output)

    expected_shape = (
        BATCH_SIZE,
        SEQ_LEN,
        cfg.vocab_size,
    )

    assert tuple(logits.shape) == expected_shape, (
        f"Wrong logits shape.\n"
        f"Expected: {expected_shape}\n"
        f"Actual:   {tuple(logits.shape)}"
    )

    assert not tensor_has_nan_or_inf(logits), (
        "Forward produced NaN or Inf."
    )

    print(
        f"[PASS] logits shape = "
        f"{tuple(logits.shape)}"
    )

    print(
        f"logits mean = "
        f"{logits.float().mean().item():.6f}"
    )

    print(
        f"logits std  = "
        f"{logits.float().std().item():.6f}"
    )

    return logits, tokens


# ============================================================
# Test 10
# Causal LM loss + backward
# ============================================================

def run_loss_and_backward(
    model,
    cfg,
    logits,
    tokens,
    name,
):

    print()
    print("=" * 72)
    print(
        f"TEST 10: Loss + backward - {name}"
    )
    print("=" * 72)

    shift_logits = (
        logits[:, :-1, :]
        .contiguous()
    )

    shift_labels = (
        tokens[:, 1:]
        .contiguous()
    )

    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(
            -1,
            shift_logits.size(-1),
        ),
        shift_labels.view(-1),
    )

    assert torch.isfinite(loss).item(), (
        f"Loss is not finite: {loss.item()}"
    )

    print(
        f"loss = {loss.item():.6f}"
    )

    model.zero_grad(
        set_to_none=True
    )

    loss.backward()

    checked = 0

    for name_param, param in (
        model.named_parameters()
    ):

        if param.grad is None:
            continue

        if not torch.isfinite(
            param.grad
        ).all().item():

            raise AssertionError(
                f"NaN/Inf gradient in "
                f"{name_param}"
            )

        checked += 1

    assert checked > 0, (
        "No parameter received a gradient."
    )

    print(
        f"[PASS] backward succeeded; "
        f"{checked} parameter tensors "
        f"received gradients."
    )


# ============================================================
# Test 11
# Shared MLP gradient flow
# ============================================================

def test_shared_gradients(model):

    print()
    print("=" * 72)
    print("TEST 11: Shared MLP gradient flow")
    print("=" * 72)

    mlp0 = get_layer_mlp(
        model.layers[0]
    )

    first_param = None

    for p in mlp0.parameters():

        first_param = p
        break

    assert first_param is not None, (
        "Shared MLP has no parameters."
    )

    assert first_param.grad is not None, (
        "Shared MLP did not receive a gradient."
    )

    grad_norm = (
        first_param.grad
        .float()
        .norm()
        .item()
    )

    assert math.isfinite(grad_norm), (
        "Shared MLP gradient is NaN/Inf."
    )

    assert grad_norm > 0.0, (
        "Shared MLP gradient norm is zero."
    )

    print(
        f"[PASS] shared MLP gradient norm = "
        f"{grad_norm:.6e}"
    )


# ============================================================
# Test 12
# Parameter registration
# ============================================================

def test_parameter_registration(model):

    print()
    print("=" * 72)
    print("TEST 12: Parameter registration")
    print("=" * 72)

    state = model.state_dict()

    assert len(state) > 0, (
        "state_dict is empty."
    )

    print(
        f"[PASS] state_dict contains "
        f"{len(state)} entries."
    )

    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------

    mobile_mlp_keys = [
        key
        for key in state
        if ".mlp." in key
    ]

    assert mobile_mlp_keys, (
        "No '.mlp.' parameters found "
        "in state_dict."
    )

    print(
        f"[PASS] found "
        f"{len(mobile_mlp_keys)} "
        f"MLP state entries."
    )

    # --------------------------------------------------------
    # LM head
    # --------------------------------------------------------

    lm_head_keys = [
        key
        for key in state
        if key.startswith("lm_head.")
    ]

    assert lm_head_keys, (
        "No 'lm_head.*' entries found. "
        "MobileLLM requires an untied LM head."
    )

    print(
        f"[PASS] found "
        f"{len(lm_head_keys)} "
        f"LM-head state entries."
    )


# ============================================================
# Test 13
# Existing P018 must still work
# ============================================================

def test_existing_p018():

    print()
    print("=" * 72)
    print("TEST 13: Existing P018 construction")
    print("=" * 72)

    cfg = build_config(
        "m100",
        "tied",
        SEQ_LEN,
        False,
    )

    assert getattr(
        cfg,
        "model_family",
        "p018",
    ) == "p018", (
        "Existing P018 config unexpectedly changed."
    )

    model = construct_model(cfg)

    n_params = count_parameters(model)

    assert n_params > 0

    print(
        "[PASS] existing P018 model still builds."
    )

    print(
        f"P018 parameter count = "
        f"{n_params:,} "
        f"({n_params / 1e6:.3f}M)"
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("#" * 72)
    print(
        "# TinyLM_P018 MobileLLM 125M "
        "Architecture Smoke + Parity Test"
    )
    print("#" * 72)

    print()
    print(
        f"PyTorch : {torch.__version__}"
    )

    print(
        f"CUDA    : "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU     : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # 1. Exact MobileLLM config
    # --------------------------------------------------------

    dense_cfg = test_config()

    # --------------------------------------------------------
    # 2. Dense
    # --------------------------------------------------------

    dense_cfg, dense_model = (
        test_mobile_dense()
    )

    # --------------------------------------------------------
    # 3. Shared g6
    # --------------------------------------------------------

    shared_cfg, shared_model = (
        test_mobile_shared()
    )

    # --------------------------------------------------------
    # 4. Attention non-sharing
    # --------------------------------------------------------

    test_attention_not_shared(
        dense_model
    )

    test_attention_not_shared(
        shared_model
    )

    # --------------------------------------------------------
    # 5. Embedding / LM-head
    # --------------------------------------------------------

    test_embedding_lm_head_not_tied(
        dense_model
    )

    test_embedding_lm_head_not_tied(
        shared_model
    )

    # --------------------------------------------------------
    # 6. Config parity
    # --------------------------------------------------------

    test_mobile_config_parity(
        dense_cfg
    )

    # --------------------------------------------------------
    # 7. Dense / Shared parity
    # --------------------------------------------------------

    test_dense_shared_parity(
        dense_cfg,
        dense_model,
        shared_cfg,
        shared_model,
    )

    # --------------------------------------------------------
    # 8. Structural parity
    # --------------------------------------------------------

    test_structural_parity(
        dense_model
    )

    # Also verify the shared model preserves
    # the same MobileLLM dimensions.
    test_structural_parity(
        shared_model
    )

    # --------------------------------------------------------
    # 9. Forward Dense
    # --------------------------------------------------------

    dense_logits, dense_tokens = (
        run_forward(
            dense_model,
            dense_cfg,
            "MobileLLM 125M Dense",
        )
    )

    # --------------------------------------------------------
    # 10. Loss + backward Dense
    # --------------------------------------------------------

    run_loss_and_backward(
        dense_model,
        dense_cfg,
        dense_logits,
        dense_tokens,
        "MobileLLM 125M Dense",
    )

    # --------------------------------------------------------
    # 11. Forward Shared
    # --------------------------------------------------------

    shared_logits, shared_tokens = (
        run_forward(
            shared_model,
            shared_cfg,
            "MobileLLM 125M + MLP sharing g6",
        )
    )

    # --------------------------------------------------------
    # 12. Loss + backward Shared
    # --------------------------------------------------------

    run_loss_and_backward(
        shared_model,
        shared_cfg,
        shared_logits,
        shared_tokens,
        "MobileLLM 125M + MLP sharing g6",
    )

    # --------------------------------------------------------
    # 13. Shared gradient
    # --------------------------------------------------------

    test_shared_gradients(
        shared_model
    )

    # --------------------------------------------------------
    # 14. Parameter registration
    # --------------------------------------------------------

    test_parameter_registration(
        shared_model
    )

    # --------------------------------------------------------
    # 15. Existing P018
    # --------------------------------------------------------

    test_existing_p018()

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    print()
    print("#" * 72)
    print("# ALL TESTS PASSED")
    print("#" * 72)
    print()

    print(
        "MobileLLM 125M Dense and "
        "MobileLLM 125M + MLP-sharing g6 "
        "are construction/forward/backward "
        "compatible."
    )

    print()
    print("MobileLLM reference:")

    print(
        "  Dense : "
        "576d / 1536ffn / 30L / 9Q / 3KV"
    )

    print(
        "  Shared : "
        "same architecture / MLP group=6"
    )

    print(
        f"  Shared MLP objects : "
        f"{MOBILE_EXPECTED_SHARED_MLPS}"
    )

    print()
    print(
        "MLP sharing:"
    )

    print(
        "  MLP 0 : L0-L5"
    )

    print(
        "  MLP 1 : L6-L11"
    )

    print(
        "  MLP 2 : L12-L17"
    )

    print(
        "  MLP 3 : L18-L23"
    )

    print(
        "  MLP 4 : L24-L29"
    )

    print()
    print(
        "Next step: integrate "
        "init_utils.py / trainer.py / kd_cache.py "
        "for the P018-R1 KD experiments."
    )

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
        print("# TEST FAILED")
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