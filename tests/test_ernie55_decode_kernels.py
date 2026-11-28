# SPDX-License-Identifier: Apache-2.0
"""ERNIE 5.5 fused decode kernels against the operations they replace."""

import importlib

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten, tree_unflatten
from mlx_vlm.models import switch_layers
from mlx_vlm.models.deepseek_v4.hyper_connection import hc_expand
from mlx_vlm.models.mla import MultiLinear

from omlx.patches.mlx_vlm_ernie5_5_compat import apply_mlx_vlm_ernie5_5_compat_patch

apply_mlx_vlm_ernie5_5_compat_patch()
kernels = importlib.import_module("mlx_vlm.models.ernie5_5_moe.decode_kernels")
language = importlib.import_module("mlx_vlm.models.ernie5_5_moe.language")

pytestmark = pytest.mark.skipif(
    not kernels.available(), reason="the decode kernels need the Metal GPU backend"
)

HALF = mx.bfloat16


@pytest.fixture(autouse=True)
def _switches():
    saved = (kernels.ENABLED, kernels.FP8_MIN_ROWS)
    yield
    kernels.ENABLED, kernels.FP8_MIN_ROWS = saved


def _relative(got, want):
    got, want = got.astype(mx.float32), want.astype(mx.float32)
    return (mx.linalg.norm(got - want) / mx.linalg.norm(want)).item()


def _same(got, want):
    return bool((got == want).all().item())


class _HyperConfig:
    hc_mult = 4
    hc_sinkhorn_iters = 20
    hc_eps = 1e-6
    rms_norm_eps = 1e-6
    hidden_size = 64


def _hyper(width, seed):
    config = type("Config", (_HyperConfig,), {"hidden_size": width})
    mx.random.seed(seed)
    connection = language.HyperConnection(config)
    connection.eval()
    connection.fn = mx.random.normal(connection.fn.shape) * 0.02
    connection.base = mx.random.normal(connection.base.shape) * 0.5
    connection.scale = mx.array([0.7, 1.3, 0.9])
    norm = nn.RMSNorm(width, eps=1e-6)
    norm.weight = (1 + 0.1 * mx.random.normal((width,))).astype(HALF)
    return connection, norm


@pytest.mark.parametrize("width", (64, 1000, 4096))
@pytest.mark.parametrize("rows", (1, 3, 8))
def test_hyper_connection_boundary_matches_composed(width, rows):
    connection, norm = _hyper(width, 0)
    x = (mx.random.normal((1, rows, 4, width)) * 0.7).astype(HALF)
    collapsed, post, comb = connection(x)
    want = norm(collapsed)

    mixes = kernels.hc_project(x, connection.fn, 1e-6)
    got, got_post, got_comb = kernels.hc_collapse_norm(
        x, mixes, connection.scale, connection.base, norm.weight, 20, 1e-6, 1e-6
    )
    # The projection sums in a different order, which can move a collapsed
    # value by one half-precision step.
    assert _relative(got, want) < 2e-3
    assert mx.allclose(got_post, post, atol=1e-5)
    assert mx.allclose(got_comb, comb, atol=1e-5)

    # The same boundary through the wrapper the model calls.
    stream, normed, _, _ = language._hc_boundary(connection, norm, x, None)
    assert stream is x
    assert _same(normed, got)


@pytest.mark.parametrize("width", (64, 4096))
@pytest.mark.parametrize("rows", (2, 5, 8))
def test_expand_shares_a_launch_with_the_next_projection(width, rows):
    first, first_norm = _hyper(width, 0)
    second, second_norm = _hyper(width, 1)
    x = (mx.random.normal((1, rows, 4, width)) * 0.7).astype(HALF)
    branch = (mx.random.normal((1, rows, width)) * 0.5).astype(HALF)
    _, post, comb = first(x)
    want_stream = hc_expand(branch, x, post, comb)
    want_collapsed, want_post, want_comb = second(want_stream)

    stream, mixes = kernels.hc_expand_project(branch, x, post, comb, second.fn, 1e-6)
    assert _relative(stream, want_stream) < 1e-4
    got, got_post, got_comb = kernels.hc_collapse_norm(
        stream, mixes, second.scale, second.base, second_norm.weight, 20, 1e-6, 1e-6
    )
    assert _relative(got, second_norm(want_collapsed)) < 2e-3
    assert mx.allclose(got_post, want_post, atol=1e-4)
    assert mx.allclose(got_comb, want_comb, atol=1e-4)


def test_hyper_connection_kernels_decline_what_they_do_not_cover():
    connection, norm = _hyper(64, 0)
    long = (mx.random.normal((1, kernels.MAX_ROWS + 1, 4, 64))).astype(HALF)
    assert kernels.hc_project(long, connection.fn, 1e-6) is None
    full = mx.random.normal((1, 2, 4, 64))
    assert kernels.hc_project(full, connection.fn, 1e-6) is None
    # The wrapper then runs the composed operations.
    _, normed, _, _ = language._hc_boundary(connection, norm, full, None)
    assert mx.allclose(normed, norm(connection(full)[0]), atol=1e-5)


@pytest.mark.parametrize("rows", (1, 3, 8))
def test_router_matches_composed(rows):
    mx.random.seed(0)
    experts, width, top_k = 512, 1024, 10
    w0 = (mx.random.normal((experts, width)) * 0.03).astype(HALF)
    w1 = (mx.random.normal((experts, width)) * 0.03).astype(HALF)
    bias = mx.random.normal((experts,)) * 0.01
    scaling = mx.random.uniform(2.0, 4.0, (experts,))
    x = (mx.random.normal((rows, width)) * 0.8).astype(HALF)

    wide = x.astype(mx.float32)
    want = mx.sigmoid(wide @ w0.astype(mx.float32).T) + mx.sigmoid(
        wide @ w1.astype(mx.float32).T
    )
    scores = kernels.router_scores(x, w0, w1)
    assert mx.allclose(scores, want, atol=1e-5)

    ids, weights = kernels.router_select(scores, bias, scaling, top_k, True)
    # Best first, the order of a full sort.
    want_ids = mx.argsort(-(want + bias), axis=-1)[:, :top_k]
    assert _same(ids, want_ids.astype(mx.uint32))
    picked = mx.take_along_axis(want, want_ids, axis=-1)
    want_weights = picked / picked.sum(axis=-1, keepdims=True) * scaling[want_ids]
    assert mx.allclose(weights, want_weights, atol=1e-5)

    _, plain = kernels.router_select(scores, bias, scaling, top_k, False)
    assert mx.allclose(plain, picked * scaling[want_ids], atol=1e-5)


def test_router_selection_breaks_ties_toward_the_lower_expert():
    scores = mx.zeros((1, 64))
    scores[0, 40] = 1.0
    ids, _ = kernels.router_select(scores, mx.zeros((64,)), mx.ones((64,)), 4, True)
    assert ids.tolist() == [[40, 0, 1, 2]]


class _Activation:
    activation_situ_beta = 4.0
    activation_situ_linear_beta = 25.0


def _switch(experts=16, width=1024, hidden=1024, seed=0):
    mx.random.seed(seed)
    switch = switch_layers.SwitchGLU(
        width, hidden, experts, activation=language.SiTU(_Activation)
    )
    switch.update(
        tree_unflatten(
            [
                (name, (mx.random.normal(value.shape) * 0.03).astype(HALF))
                for name, value in tree_flatten(switch.parameters())
            ]
        )
    )
    nn.quantize(switch, group_size=32, bits=4, mode="mxfp4")
    mx.eval(switch.parameters())
    return switch


@pytest.mark.parametrize("rows", (1, 2, 4, 8))
def test_routed_experts_match_each_row_run_alone(rows):
    switch = _switch()
    mx.random.seed(rows)
    x = (mx.random.normal((rows, 1024)) * 0.6).astype(HALF)
    # Ten of sixteen experts per row, so rows share most of them.
    ids = mx.stack([mx.random.permutation(16)[:10] for _ in range(rows)])
    ids = ids.astype(mx.uint32)
    got = kernels.routed_experts(switch, x, ids, 4.0, 25.0)
    want = mx.concatenate(
        [switch(x[row][None, None], ids[row][None, None])[0] for row in range(rows)]
    )
    assert got.shape == (rows, 10, 1024)
    assert _same(got, want)


def test_routed_experts_decline_other_formats():
    affine = switch_layers.SwitchGLU(
        1024, 1024, 8, activation=language.SiTU(_Activation)
    )
    nn.quantize(affine, group_size=64, bits=4)
    x = mx.random.normal((1, 1024)).astype(HALF)
    ids = mx.arange(8, dtype=mx.uint32)[None]
    assert kernels.routed_experts(affine, x, ids, 4.0, 25.0) is None
    assert (
        kernels.routed_experts(_switch(), x.astype(mx.float32), ids, 4.0, 25.0) is None
    )


@pytest.mark.parametrize("rows", (1, 5, 8))
def test_combine_norm_matches_composed(rows):
    mx.random.seed(rows)
    y = (mx.random.normal((rows, 10, 2048)) * 0.4).astype(HALF)
    weights = mx.random.uniform(0.1, 0.5, (rows, 10))
    norm = nn.RMSNorm(2048, eps=1e-6)
    norm.weight = (1 + 0.1 * mx.random.normal((2048,))).astype(HALF)
    want = norm((y.astype(mx.float32) * weights[..., None]).sum(axis=-2).astype(HALF))
    got = kernels.combine_norm(y, weights, norm.weight, 1e-6)
    assert _relative(got, want) < 2e-3


def _mxfp8(outputs, width, seed):
    mx.random.seed(seed)
    linear = nn.Linear(width, outputs, bias=False)
    linear.weight = (mx.random.normal((outputs, width)) * 0.02).astype(HALF)
    quantized = linear.to_quantized(group_size=32, bits=8, mode="mxfp8")
    mx.eval(quantized.parameters())
    return quantized


@pytest.mark.parametrize(
    "outputs,width", ((2048, 256), (1024, 1024), (512, 4096), (256, 8192))
)
@pytest.mark.parametrize("rows", (1, 3, 8))
def test_mxfp8_projection_is_the_one_row_product_of_every_row(outputs, width, rows):
    kernels.FP8_MIN_ROWS = 1
    layer = _mxfp8(outputs, width, 0)
    x = (mx.random.normal((1, rows, width)) * 0.5).astype(HALF)
    got = kernels.fp8_linear(layer, x)
    want = mx.concatenate([layer(x[:, row : row + 1]) for row in range(rows)], axis=1)
    assert _same(got, want)


def test_mxfp8_projection_leaves_short_blocks_to_mlx():
    layer = _mxfp8(256, 1024, 0)
    x = (mx.random.normal((1, 8, 1024)) * 0.5).astype(HALF)
    assert kernels.FP8_MIN_ROWS == 3
    assert kernels.fp8_linear(layer, x[:, :2]) is None
    assert kernels.fp8_linear(layer, x[:, :3]) is not None
    assert kernels.projects_block(x) and not kernels.projects_block(x[:, :2])
    # Longer blocks, other precisions and other formats decline too.
    assert kernels.fp8_linear(layer, mx.concatenate([x, x], axis=1)) is None
    assert kernels.fp8_linear(layer, x.astype(mx.float32)) is None
    affine = nn.Linear(1024, 256, bias=False).to_quantized(group_size=64, bits=8)
    assert kernels.fp8_linear(affine, x) is None
    # The wrapper the model calls falls back to the layer itself.
    assert _same(language._linear(layer, x[:, :2]), layer(x[:, :2]))


@pytest.mark.parametrize("rows", (3, 6))
def test_grouped_and_gated_mxfp8_projections(rows):
    mx.random.seed(rows)
    grouped = MultiLinear(1024, 256, 4)
    grouped.weight = (mx.random.normal(grouped.weight.shape) * 0.02).astype(HALF)
    grouped = grouped.to_quantized(group_size=32, bits=8, mode="mxfp8")
    x = (mx.random.normal((1, rows, 4, 1024)) * 0.5).astype(HALF)
    want = mx.concatenate(
        [
            grouped(x[:, row : row + 1].transpose(0, 2, 1, 3))
            .transpose(0, 2, 1, 3)
            .flatten(-2)
            for row in range(rows)
        ],
        axis=1,
    )
    assert _same(kernels.fp8_grouped(grouped, x), want)

    gate, up = _mxfp8(512, 1024, 1), _mxfp8(512, 1024, 2)
    activation = language.SiTU(_Activation)
    x = (mx.random.normal((1, rows, 1024)) * 0.5).astype(HALF)
    want = mx.concatenate(
        [
            activation(up(x[:, row : row + 1]), gate(x[:, row : row + 1]))
            for row in range(rows)
        ],
        axis=1,
    )
    assert _same(kernels.fp8_gate_up(gate, up, x, 4.0, 25.0), want)


def test_pair_projection_reads_half_precision_weights_in_place():
    mx.random.seed(0)
    w0 = (mx.random.normal((96, 512)) * 0.05).astype(HALF)
    w1 = (mx.random.normal((96, 512)) * 0.05).astype(HALF)
    x = (mx.random.normal((1, 3, 512))).astype(HALF)
    got = kernels.pair_project(x, w0, w1)
    wide = x[0].astype(mx.float32)
    assert got.shape == (2, 3, 96)
    assert mx.allclose(got[0], wide @ w0.astype(mx.float32).T, atol=1e-4)
    assert mx.allclose(got[1], wide @ w1.astype(mx.float32).T, atol=1e-4)


@pytest.mark.parametrize("rows", (1, 2, 8, 11))
def test_vocabulary_head_matches_composed(rows):
    mx.random.seed(rows)
    vocab, width = 1003, 256
    weight = (mx.random.normal((vocab, width)) * 0.2).astype(HALF)
    hidden = mx.random.normal((rows, width))
    want = hidden @ weight.astype(mx.float32).T
    assert mx.allclose(kernels.vocab_head(hidden, weight), want, atol=1e-3)

    multimax = language.MultiMax()
    multimax.multimax_ranges = mx.array([1.5, 2.0, 0.5, 0.25])
    multimax.multimax_ts = mx.array([-0.13, 0.28, -0.1, -0.03])
    shaped = kernels.vocab_head(hidden, weight, multimax.packed())
    assert mx.allclose(shaped, multimax(want), atol=2e-3)

    if rows <= kernels.MAX_ROWS:
        base = mx.random.normal((rows, vocab))
        with_base = kernels.vocab_head(hidden, weight, multimax.packed(), base)
        assert mx.allclose(with_base, multimax(want + base), atol=2e-3)


def test_multimax_packing_follows_the_weights():
    multimax = language.MultiMax()
    first = multimax.packed()
    assert multimax.packed() is first
    multimax.multimax_ts = mx.array([1.0, 2.0, 3.0, 4.0])
    assert multimax.packed().tolist()[4:] == [1.0, 2.0, 3.0, 4.0]


def _moe(seed=0):
    """A latent MoE small enough to test and wide enough for every kernel."""
    config = type(
        "Config",
        (_Activation,),
        {
            "hidden_size": 256,
            "moe_latent_size": 512,
            "moe_intermediate_size": 512,
            "n_routed_experts": 16,
            "num_experts_per_tok": 4,
            "n_shared_experts": 1,
            "norm_topk_prob": True,
            "routed_scaling_factor": 2.5,
            "rms_norm_eps": 1e-6,
        },
    )
    mx.random.seed(seed)
    moe = language.LatentMoE(config)
    moe.update(
        tree_unflatten(
            [
                (
                    name,
                    (
                        (1 + 0.1 * mx.random.normal(value.shape))
                        if "norm" in name
                        else mx.random.normal(value.shape) * 0.05
                    ).astype(HALF),
                )
                for name, value in tree_flatten(moe.parameters())
                if "routed_scaling" not in name and "correction" not in name
            ]
        )
    )

    def predicate(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        if "switch_mlp" in path:
            return {"group_size": 32, "bits": 4, "mode": "mxfp4"}
        return {"group_size": 32, "bits": 8, "mode": "mxfp8"}

    nn.quantize(moe, class_predicate=predicate)
    moe.eval()
    mx.eval(moe.parameters())
    return moe


@pytest.mark.parametrize("rows", (1, 4, 8))
def test_latent_moe_block_runs_on_the_kernels_and_matches_composed(rows, monkeypatch):
    moe = _moe()
    x = (mx.random.normal((1, rows, 256)) * 0.5).astype(HALF)
    calls = []
    real = kernels.routed_experts

    def spy(*args):
        calls.append(args[1].shape)
        return real(*args)

    monkeypatch.setattr(kernels, "routed_experts", spy)
    got = moe(x)
    mx.eval(got)
    assert calls == [(rows, 512)]

    kernels.ENABLED = False
    calls.clear()
    want = moe(x)
    mx.eval(want)
    assert calls == []
    # Same experts and the same sums, up to half-precision rounding of the
    # combine and the block-size dependence of MLX's own projections.
    assert _relative(got, want) < 2e-2


def test_block_result_does_not_depend_on_the_block(monkeypatch):
    """A row of a verify block gets the routed output it gets on its own."""
    moe = _moe()
    x = (mx.random.normal((1, 6, 256)) * 0.5).astype(HALF)
    seen = {}
    real = kernels.routed_experts

    def spy(switch, latent, ids, *rest):
        out = real(switch, latent, ids, *rest)
        seen[latent.shape[0]] = (latent, ids, out)
        return out

    monkeypatch.setattr(kernels, "routed_experts", spy)
    mx.eval(moe(x))
    latent, ids, out = seen[6]
    pairs = [
        real(moe.switch_mlp, latent[row : row + 2], ids[row : row + 2], 4.0, 25.0)
        for row in range(0, 6, 2)
    ]
    assert _same(out, mx.concatenate(pairs))
