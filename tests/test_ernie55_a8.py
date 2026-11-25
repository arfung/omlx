# SPDX-License-Identifier: Apache-2.0
"""A8 prefill for ERNIE 5.5 Flash routed experts (omlx.patches.ernie5_5.a8)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from omlx.model_settings import oq_a8_model_supported
from omlx.patches.ernie5_5 import a8

ernie = pytest.importorskip("mlx_vlm.models.ernie5_5_moe")
language = pytest.importorskip("mlx_vlm.models.ernie5_5_moe.language")
switch_layers = pytest.importorskip("mlx_vlm.models.switch_layers")

MXFP4 = {"group_size": 32, "bits": 4, "mode": "mxfp4"}


def _switch(experts=8, width=128, hidden=128, mode=MXFP4, seed=0):
    """A small routed-expert block with ERNIE's activation, quantized."""
    mx.random.seed(seed)
    config = type(
        "Config", (), {"activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0}
    )
    switch = switch_layers.SwitchGLU(
        width, hidden, experts, activation=language.SiTU(config)
    )
    switch.update(
        tree_unflatten(
            [
                (name, (mx.random.normal(value.shape) * 0.05).astype(mx.bfloat16))
                for name, value in tree_flatten(switch.parameters())
            ]
        )
    )
    nn.quantize(switch, **mode)
    mx.eval(switch.parameters())
    return switch


def _inputs(tokens, width=128, experts=8, topk=3, seed=1):
    mx.random.seed(seed)
    x = (mx.random.normal((1, tokens, width)) * 0.5).astype(mx.bfloat16)
    scores = mx.random.normal((1, tokens, experts))
    ids = mx.argpartition(-scores, kth=topk - 1, axis=-1)[..., :topk]
    return x, ids


def test_settings_offer_a8_for_ernie():
    assert oq_a8_model_supported("ernie5_5_moe")
    assert not oq_a8_model_supported("ernie4_5_moe")


def test_supported_reads_only_mxfp4_experts():
    assert a8.supported(_switch())
    affine = _switch(mode={"group_size": 64, "bits": 4, "mode": "affine"})
    assert not a8.supported(affine)


def test_routed_declines_short_blocks_and_other_dtypes():
    switch = _switch()
    x, ids = _inputs(8)
    # Decode and verify blocks keep the A16 path whatever the setting says.
    assert a8.routed(switch, x, ids, min_tokens=1) is None
    x, ids = _inputs(256)
    assert a8.routed(switch, x.astype(mx.float32), ids, min_tokens=128) is None
    assert a8.routed(switch, x, ids, min_tokens=1024) is None


def test_routed_matches_the_a16_experts():
    switch = _switch()
    x, ids = _inputs(256)
    got = a8.routed(switch, x, ids, min_tokens=128)
    if got is None:
        pytest.skip("INT8 tensor-unit kernels are not available on this machine")
    want = switch(x, ids)
    assert got.shape == want.shape
    got, want = got.astype(mx.float32), want.astype(mx.float32)
    error = (mx.linalg.norm(got - want) / mx.linalg.norm(want)).item()
    # INT8 activations: a few percent, far below one in a routing mix-up.
    assert error < 0.05


def test_routed_keeps_each_token_with_its_own_experts():
    switch = _switch()
    x, ids = _inputs(256)
    got = a8.routed(switch, x, ids, min_tokens=128)
    if got is None:
        pytest.skip("INT8 tensor-unit kernels are not available on this machine")
    # Reversing the expert order of every token reverses its outputs.
    flipped = a8.routed(switch, x, ids[..., ::-1], min_tokens=128)
    assert mx.allclose(
        flipped.astype(mx.float32), got[..., ::-1, :].astype(mx.float32), atol=1e-2
    )


def _tiny_model():
    config = ernie.ModelConfig.from_dict(
        {
            "model_type": "ernie5_5_moe",
            "hidden_size": 64,
            "vocab_size": 128,
            "num_hidden_layers": 3,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "head_dim": 16,
            "q_lora_rank": 32,
            "kv_lora_rank": 16,
            "qk_rope_head_dim": 8,
            "o_groups": 2,
            "o_lora_rank": 16,
            "sliding_window": 4,
            "compress_ratios": [8, 8, -2],
            "hybrid_mla_num_attention_heads": 4,
            "hybrid_mla_q_lora_rank": 32,
            "hybrid_mla_kv_lora_rank": 16,
            "hybrid_mla_qk_nope_head_dim": 8,
            "hybrid_mla_qk_rope_head_dim": 8,
            "hybrid_mla_v_head_dim": 16,
            "index_n_heads": 8,
            "index_head_dim": 16,
            "index_topk": 4,
            # The gather reads experts in whole 64-row tiles and scale tiles
            # of four 32-code groups, so the smallest readable expert is 128.
            "moe_latent_size": 128,
            "moe_intermediate_size": 128,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
        }
    )
    model = ernie.Model(config)
    nn.quantize(
        model,
        class_predicate=lambda path, module: (
            dict(MXFP4)
            if "switch_mlp" in path and hasattr(module, "to_quantized")
            else False
        ),
    )
    return model


def test_tag_offers_the_backend_on_every_moe_layer():
    model = _tiny_model()
    layers = model.language_model.layers
    # Layer 0 is dense; layers 1 and 2 route experts.
    assert a8.tag(model, 128) == 2
    assert getattr(layers[0].ffn, a8.BACKEND, None) is None
    for layer in layers[1:]:
        backend = getattr(layer.ffn, a8.BACKEND)
        assert backend.func is a8.routed
        assert backend.keywords == {"min_tokens": 128}
    a8.untag(model)
    assert all(getattr(layer.ffn, a8.BACKEND, None) is None for layer in layers)


def test_tag_skips_experts_it_cannot_read():
    model = ernie.Model(_tiny_model().config)  # unquantized experts
    assert a8.tag(model, 128) == 0


def test_model_uses_the_backend_only_when_it_answers():
    model = _tiny_model()
    model.eval()
    ffn = model.language_model.layers[1].ffn
    x = (mx.random.normal((1, 16, 64)) * 0.5).astype(mx.float32)
    plain = ffn(x)
    calls = []

    def declining(switch, latent, ids):
        calls.append(latent.shape)
        return None

    ffn.routed_backend = declining
    assert mx.allclose(ffn(x), plain, atol=1e-6)
    # The backend sees the latent rows, not the layer input.
    assert calls == [(1, 16, 128)]

    ffn.routed_backend = lambda switch, latent, ids: switch(latent, ids) * 0
    zeroed = ffn(x)
    assert not mx.allclose(zeroed, plain, atol=1e-6)
