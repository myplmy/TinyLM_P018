"""
TinyLM_P018 Mobile architecture smoke test.

Run from the repository root:

    python test_mobile.py

This test does NOT train the model.
It verifies:

1. mobile100 config construction
2. Mobile Dense model construction
3. Mobile Shared MLP construction
4. parameter registration
5. MLP sharing identity
6. attention non-sharing
7. forward pass
8. logits shape
9. causal LM loss
10. backward pass
11. NaN / Inf detection
12. existing P018 model construction

Expected repository layout:

TinyLM_P018/
    tinylm/
        config.py
        model/
            modules.py
            transformer.py
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


# transformer.py may expose different factory names depending
# on the implementation. We first try build_model(), then fall
# back to MobileTransformer / TiedMLPTransformer.
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
        TiedMLPTransformer = None
    except ImportError:
        from tinylm.model.transformer import (
            MobileTransformer,
            TiedMLPTransformer,
        )

        build_model = None


# ============================================================
# Test configuration
# ============================================================

SEQ_LEN = 32
BATCH_SIZE = 2

# Small vocabulary IDs are sufficient for a smoke test.
# The actual config vocab_size is used to validate the range.
SEED = 1234


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


def print_model_summary(name, model, cfg):
    total = count_parameters(model)
    trainable = count_trainable_parameters(model)

    print()
    print("=" * 72)
    print(name)
    print("=" * 72)

    print(f"model_family : {getattr(cfg, 'model_family', '<missing>')}")
    print(f"dim          : {cfg.dim}")
    print(f"ffn_dim      : {cfg.ffn_dim}")
    print(f"layers       : {getattr(cfg, 'n_layers', '<missing>')}")
    print(f"q_heads      : {cfg.n_q_heads}")
    print(f"kv_heads     : {cfg.n_kv_heads}")
    print(f"mlp_group    : {getattr(cfg, 'mlp_group', '<missing>')}")
    print(f"tie_mlp      : {getattr(cfg, 'tie_mlp', '<missing>')}")

    print()
    print(f"parameters   : {total:,}")
    print(f"trainable    : {trainable:,}")
    print(f"parameters M : {total / 1e6:.3f}")


def construct_model(cfg):
    """
    Use the project's factory if available.

    This deliberately keeps the test compatible with either:

        build_model(cfg)

    or direct:

        MobileTransformer(cfg)
    """

    if build_model is not None:
        return build_model(cfg)

    if getattr(cfg, "model_family", None) == "mobile":
        return MobileTransformer(cfg)

    if TiedMLPTransformer is not None:
        return TiedMLPTransformer(cfg)

    raise RuntimeError(
        "No model construction path available."
    )


def get_layer_mlp(layer):
    """
    Current MobileLayer is expected to expose:

        layer.mlp

    This helper makes the test fail clearly if the interface
    differs.
    """

    if not hasattr(layer, "mlp"):
        raise AssertionError(
            "Mobile layer has no '.mlp' attribute."
        )

    return layer.mlp


def get_layer_attention(layer):
    """
    Current MobileLayer is expected to expose:

        layer.attn
    """

    if not hasattr(layer, "attn"):
        raise AssertionError(
            "Mobile layer has no '.attn' attribute."
        )

    return layer.attn


# ============================================================
# Test 1
# Config
# ============================================================

def test_config():
    print()
    print("=" * 72)
    print("TEST 1: mobile100 configuration")
    print("=" * 72)

    cfg = build_config(
        "mobile100",
        "dense",
        SEQ_LEN,
        False,
    )

    assert getattr(cfg, "model_family", None) == "mobile", (
        "mobile100 config does not have model_family='mobile'"
    )

    assert cfg.dim == 768, (
        f"Expected dim=768, got {cfg.dim}"
    )

    assert cfg.ffn_dim == 2048, (
        f"Expected ffn_dim=2048, got {cfg.ffn_dim}"
    )

    assert cfg.n_q_heads == 12, (
        f"Expected n_q_heads=12, got {cfg.n_q_heads}"
    )

    assert cfg.n_kv_heads == 3, (
        f"Expected n_kv_heads=3, got {cfg.n_kv_heads}"
    )

    assert cfg.n_q_heads % cfg.n_kv_heads == 0

    print("[PASS] mobile100 config is valid.")

    return cfg


# ============================================================
# Test 2
# Mobile Dense
# ============================================================

def test_mobile_dense():
    print()
    print("=" * 72)
    print("TEST 2: Mobile Dense construction")
    print("=" * 72)

    cfg = build_config(
        "mobile100",
        "dense",
        SEQ_LEN,
        False,
    )

    model = construct_model(cfg)

    print_model_summary(
        "Mobile Dense",
        model,
        cfg,
    )

    assert len(model.layers) == cfg.n_layers, (
        f"Expected {cfg.n_layers} layers, "
        f"got {len(model.layers)}"
    )

    # Dense means every layer owns a distinct MLP.
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
        f"[PASS] {len(mlps)} layers "
        f"have {unique_mlp_ids} unique MLP objects."
    )

    return cfg, model


# ============================================================
# Test 3
# Mobile Shared
# ============================================================

def test_mobile_shared():
    print()
    print("=" * 72)
    print("TEST 3: Mobile Shared MLP construction")
    print("=" * 72)

    # Start from the same Mobile architecture.
    cfg = build_config(
        "mobile100",
        "tied",
        SEQ_LEN,
        False,
    )

    # P018's existing convention:
    #
    #   tie_mlp=True
    #   mlp_group=4
    #
    # means four consecutive layers share one MLP.
    cfg.tie_mlp = True
    cfg.mlp_group = 4

    model = construct_model(cfg)

    print_model_summary(
        "Mobile Shared (MLP group=4)",
        model,
        cfg,
    )

    assert len(model.layers) == cfg.n_layers

    mlps = [
        get_layer_mlp(layer)
        for layer in model.layers
    ]

    # --------------------------------------------------------
    # Expected:
    #
    # 0 1 2 3 -> same MLP
    # 4 5 6 7 -> same MLP
    # 8 9 10 11 -> same MLP
    # ...
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
            f"Layers {start}:{end} do not share "
            f"one MLP object."
        )
    
    # Different groups must have different MLPs.
    group_roots = []
    
    for start in range(
        0,
        cfg.n_layers,
        cfg.mlp_group,
    ):
        group_roots.append(
            id(mlps[start])
        )
    
    assert len(group_roots) == len(set(group_roots)), (
        "Different MLP groups unexpectedly share "
        "the same MLP object."
    )
    
    expected_groups = math.ceil(
        cfg.n_layers / cfg.mlp_group
    )
    
    actual_groups = len(set(
        id(mlp)
        for mlp in mlps
    ))
    
    assert actual_groups == expected_groups, (
        f"Expected {expected_groups} unique MLPs, "
        f"got {actual_groups}"
    )
    
    print(
        f"[PASS] {cfg.n_layers} layers use "
        f"{actual_groups} shared MLPs."
    )
    
    return cfg, model


# ============================================================
# Test 4
# Attention must NOT be shared
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
        f"{unique_attention_ids} unique attention modules."
    )


# ============================================================
# Test 5
# Forward
# ============================================================

def run_forward(model, cfg, name):
    print()
    print("=" * 72)
    print(f"TEST 5: Forward - {name}")
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
    
    # --------------------------------------------------------
    # Accept either:
    #
    #   logits
    #
    # or:
    #
    #   (logits, ...)
    #
    # or dictionary-like output.
    # --------------------------------------------------------
    
    if isinstance(output, torch.Tensor):
        logits = output
    
    elif isinstance(output, tuple):
        logits = output[0]
    
    elif isinstance(output, dict):
        if "logits" in output:
            logits = output["logits"]
        else:
            # Some P018 code may use a different key.
            logits = next(iter(output.values()))
    
    else:
        raise AssertionError(
            f"Unknown model output type: "
            f"{type(output)}"
        )
    
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
        f"[PASS] logits shape = {tuple(logits.shape)}"
    )
    
    print(
        f"logits mean = {logits.float().mean().item():.6f}"
    )
    
    print(
        f"logits std  = {logits.float().std().item():.6f}"
    )
    
    return logits, tokens


# ============================================================
# Test 6
# Causal LM loss
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
    print(f"TEST 6: Loss + backward - {name}")
    print("=" * 72)
    
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens[:, 1:].contiguous()
    
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
    
    for name_param, param in model.named_parameters():
        
        if param.grad is None:
            continue
            
        if not torch.isfinite(
            param.grad
        ).all().item():
            raise AssertionError(
                f"NaN/Inf gradient in {name_param}"
            )
            
        checked += 1
    
    assert checked > 0, (
        "No parameter received a gradient."
    )
    
    print(
        f"[PASS] backward succeeded; "
        f"{checked} parameter tensors received gradients."
    )


# ============================================================
# Test 7
# Shared parameters must receive gradients
# ============================================================

def test_shared_gradients(model):
    print()
    print("=" * 72)
    print("TEST 7: Shared MLP gradient flow")
    print("=" * 72)
    
    mlp0 = get_layer_mlp(
        model.layers[0]
    )
    
    # Find the first trainable parameter.
    first_param = None
    
    for p in mlp0.parameters():
        first_param = p
        break
    
    assert first_param is not None
    
    assert first_param.grad is not None, (
        "Shared MLP did not receive a gradient."
    )
    
    grad_norm = first_param.grad.float().norm().item()
    
    assert math.isfinite(grad_norm)
    assert grad_norm > 0.0, (
        "Shared MLP gradient norm is zero."
    )
    
    print(
        f"[PASS] shared MLP gradient norm = "
        f"{grad_norm:.6e}"
    )


# ============================================================
# Test 8
# Parameter registration
# ============================================================

def test_parameter_registration(model):
    print()
    print("=" * 72)
    print("TEST 8: Parameter registration")
    print("=" * 72)
    
    state = model.state_dict()
    
    assert len(state) > 0, (
        "state_dict is empty."
    )
    
    print(
        f"[PASS] state_dict contains "
        f"{len(state)} entries."
    )
    
    # Check that Mobile MLP parameters are present.
    mobile_mlp_keys = [
        key
        for key in state
        if ".mlp." in key
    ]
    
    assert mobile_mlp_keys, (
        "No '.mlp.' parameters found in state_dict."
    )
    
    print(
        f"[PASS] found {len(mobile_mlp_keys)} "
        f"MLP state entries."
    )


# ============================================================
# Test 9
# Existing P018 must still work
# ============================================================

def test_existing_p018():
    print()
    print("=" * 72)
    print("TEST 9: Existing P018 construction")
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
        f"[PASS] existing P018 model still builds."
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
    print("# TinyLM_P018 Mobile Architecture Smoke Test")
    print("#" * 72)
    
    print()
    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(
            f"GPU     : "
            f"{torch.cuda.get_device_name(0)}"
        )
    
    # --------------------------------------------------------
    # 1. Config
    # --------------------------------------------------------
    
    test_config()
    
    # --------------------------------------------------------
    # 2. Mobile Dense
    # --------------------------------------------------------
    
    dense_cfg, dense_model = test_mobile_dense()
    
    # --------------------------------------------------------
    # 3. Mobile Shared
    # --------------------------------------------------------
    
    shared_cfg, shared_model = test_mobile_shared()
    
    # --------------------------------------------------------
    # 4. Attention identity
    # --------------------------------------------------------
    
    test_attention_not_shared(
        shared_model
    )
    
    # --------------------------------------------------------
    # 5. Forward Dense
    # --------------------------------------------------------
    
    dense_logits, dense_tokens = run_forward(
        dense_model,
        dense_cfg,
        "Mobile Dense",
    )
    
    # --------------------------------------------------------
    # 6. Loss + backward Dense
    # --------------------------------------------------------
    
    run_loss_and_backward(
        dense_model,
        dense_cfg,
        dense_logits,
        dense_tokens,
        "Mobile Dense",
    )
    
    # --------------------------------------------------------
    # 7. Forward Shared
    # --------------------------------------------------------
    
    shared_logits, shared_tokens = run_forward(
        shared_model,
        shared_cfg,
        "Mobile Shared",
    )
    
    # --------------------------------------------------------
    # 8. Loss + backward Shared
    # --------------------------------------------------------
    
    run_loss_and_backward(
        shared_model,
        shared_cfg,
        shared_logits,
        shared_tokens,
        "Mobile Shared",
    )
    
    # --------------------------------------------------------
    # 9. Shared gradient
    # --------------------------------------------------------
    
    test_shared_gradients(
        shared_model
    )
    
    # --------------------------------------------------------
    # 10. state_dict
    # --------------------------------------------------------
    
    test_parameter_registration(
        shared_model
    )
    
    # --------------------------------------------------------
    # 11. Existing P018
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
        "Mobile Dense and Mobile Shared are "
        "construction/forward/backward compatible."
    )
    print()
    print(
        "Next step: integrate init_utils.py / trainer.py "
        "for actual KD experiments."
    )
    print()


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