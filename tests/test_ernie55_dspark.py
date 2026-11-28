# SPDX-License-Identifier: Apache-2.0
"""ERNIE 5.5 Flash DSpark: target captures, checkpoint mapping and round loop."""

import copy
import importlib
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten
from mlx_vlm.models.deepseek_v4 import hyper_connection
from mlx_vlm.speculative import dflash
from mlx_vlm.speculative.drafters import DRAFTER_KIND_BY_MODEL_TYPE

from omlx.patches.mlx_vlm_ernie5_5_compat import apply_mlx_vlm_ernie5_5_compat_patch

apply_mlx_vlm_ernie5_5_compat_patch()
ernie = importlib.import_module("mlx_vlm.models.ernie5_5_moe")
language = importlib.import_module("mlx_vlm.models.ernie5_5_moe.language")
ernie5_5_dspark = importlib.import_module(
    "mlx_vlm.speculative.drafters.ernie5_5_dspark"
)
policy = importlib.import_module("mlx_vlm.speculative.drafters.ernie5_5_dspark.policy")
split = importlib.import_module("mlx_vlm.speculative.drafters.ernie5_5_dspark.split")
mtp_split = importlib.import_module("mlx_vlm.speculative.drafters.ernie5_5_mtp.split")
Model, ModelConfig = ernie.Model, ernie.ModelConfig
_dflash_next_block_size, _dflash_rounds = (
    dflash._dflash_next_block_size,
    dflash._dflash_rounds,
)
CopyIndex, DepthPolicy = policy.CopyIndex, policy.DepthPolicy
Ernie55DsparkSplitter, unbundle = split.Ernie55DsparkSplitter, split.unbundle
Ernie55MTPSplitter = mtp_split.Ernie55MTPSplitter

TINY_CONFIG = {
    "model_type": "ernie5_5_moe",
    "hidden_size": 64,
    "vocab_size": 128,
    "num_hidden_layers": 2,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "head_dim": 16,
    "q_lora_rank": 32,
    "kv_lora_rank": 16,
    "qk_rope_head_dim": 8,
    "o_groups": 2,
    "o_lora_rank": 16,
    "sliding_window": 4,
    "compress_ratios": [8, -2],
    "hybrid_mla_num_attention_heads": 4,
    "hybrid_mla_q_lora_rank": 32,
    "hybrid_mla_kv_lora_rank": 16,
    "hybrid_mla_qk_nope_head_dim": 8,
    "hybrid_mla_qk_rope_head_dim": 8,
    "hybrid_mla_v_head_dim": 16,
    "index_n_heads": 8,
    "index_head_dim": 16,
    "index_topk": 4,
    "moe_latent_size": 64,
    "moe_intermediate_size": 64,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
}


def _randomize(module, seed):
    """Float32 random parameters, so no two expert scores or logits tie."""
    mx.random.seed(seed)
    params = []
    for name, value in tree_flatten(module.parameters()):
        if "norm" in name and name.endswith("weight"):
            params.append((name, mx.ones(value.shape, dtype=mx.float32)))
        else:
            params.append((name, mx.random.normal(value.shape, dtype=mx.float32) * 0.1))
    module.update(tree_unflatten(params))
    mx.eval(module.parameters())


def _target():
    config = copy.deepcopy(TINY_CONFIG)
    model = Model(ModelConfig.from_dict(config))
    _randomize(model, 0)
    return config, model


def _drafter(config, block_size=4):
    draft_config = ernie5_5_dspark.ModelConfig.from_dict(
        {
            "text_config": config,
            "n_mtp_layers": 2,
            "target_layer_ids": [0, 1],
            "mask_token_id": 3,
            "markov_rank": 8,
            "block_size": block_size,
        }
    )
    draft = ernie5_5_dspark.Model(draft_config)
    _randomize(draft, 1)
    return draft


def _greedy(logits):
    return mx.argmax(logits, axis=-1)


def test_capture_layer_ids_returns_stream_means():
    _, model = _target()
    lm = model.language_model
    tokens = mx.array([[5, 9, 2, 44, 7, 81]])

    captured = lm(tokens, cache=model.make_cache(), capture_layer_ids=[0, 1])
    assert [h.shape for h in captured.hidden_states] == [(1, 6, 64)] * 2

    final = lm(tokens, cache=model.make_cache(), return_hidden=True).hidden_states[0]
    streams = final.reshape(1, 6, lm.config.hc_mult, 64)
    assert mx.allclose(captured.hidden_states[1], streams.mean(axis=2), atol=1e-5)
    # The logits are unchanged by capturing.
    plain = lm(tokens, cache=model.make_cache())
    assert mx.allclose(captured.logits, plain.logits, atol=1e-5)


def test_preview_config_selects_the_dspark_splitter():
    assert DRAFTER_KIND_BY_MODEL_TYPE["ernie5_5_dspark"] == "dflash"
    base = {"model_type": "ernie5_5_moe"}
    assert type(Ernie55MTPSplitter.from_config(base)) is Ernie55MTPSplitter
    preview = {**base, "dspark_target_layer_ids": [40, 41, 42], "dspark_block_size": 7}
    splitter = Ernie55MTPSplitter.from_config(preview)
    assert type(splitter) is Ernie55DsparkSplitter
    assert splitter.select_keys("model.mtp.0.norm.weight", preview)
    assert not splitter.select_keys("model.layers.0.norm.weight", preview)
    assert splitter.extra_config(
        {**preview, "dspark_noise_token_id": 1024, "dspark_markov_rank": 256}
    ) == {
        "n_mtp_layers": 3,
        "target_layer_ids": [40, 41, 42],
        "mask_token_id": 1024,
        "markov_rank": 256,
        "block_size": 8,
    }


def test_bundled_tensors_map_onto_the_drafter():
    hidden, hc, experts, vocab, rank = 8, 4, 2, 16, 4
    stage = "model.mtp.1."
    source = {
        "model.mtp.0.self_attn.linear_q_down_proj.weight": mx.zeros((4, hidden)),
        "model.mtp.0.self_attn.vha_postmix_U": mx.zeros((2, 2)),
        "model.mtp.0.self_attn.core_attention.attn_sink": mx.zeros((2,)),
        "model.mtp.0.self_attn.gate_proj.weight": mx.zeros((8, 4)),
        "model.mtp.0.self_attn.linear_o_group_proj": mx.zeros((2 * 3, 6)),
        "model.mtp.0.self_attention_hyper_connection.mapping_proj.weight": mx.zeros(
            (hc * hidden, 24)
        ),
        "model.mtp.0.self_attention_hyper_connection.bias": mx.zeros((24,)),
        "model.mtp.0.self_attention_hyper_connection.alpha_pre": mx.array([1.0]),
        "model.mtp.0.self_attention_hyper_connection.alpha_post": mx.array([2.0]),
        "model.mtp.0.self_attention_hyper_connection.alpha_res": mx.array([3.0]),
        stage + "block_sparse_moe.gate.weight_1": mx.zeros((experts, hidden)),
        stage + "block_sparse_moe.shared_experts.w2.weight": mx.zeros((hidden, 4)),
        "model.mtp.feature_projection.weight": mx.zeros((hidden, 3 * hidden)),
        "model.mtp.feature_norm.weight": mx.ones((hidden,)),
        "model.mtp.markov_head.markov_w1.weight": mx.zeros((rank, vocab)),
        "model.mtp.markov_head.markov_w2.weight": mx.zeros((vocab, rank)),
        "model.mtp.mhc_contract.hc_head_fn": mx.zeros((hc * hidden, hc)),
        "model.mtp.mhc_contract.hc_head_base": mx.zeros((hc,)),
        "model.mtp.norm.weight": mx.ones((hidden,)),
        "model.mtp.confidence_head.projection.weight": mx.zeros((1, hidden + rank)),
    }
    for expert in range(experts):
        source[stage + f"block_sparse_moe.experts.{expert}.w1.weight"] = mx.full(
            (4, 4), float(expert)
        )

    context = SimpleNamespace(
        args=SimpleNamespace(o_groups=2, o_lora_rank=3, n_routed_experts=experts)
    )
    out = ernie5_5_dspark.Model.sanitize(context, unbundle(source))

    shapes = {key: value.shape for key, value in out.items()}
    assert shapes == {
        "stages.0.attn.wq_a.weight": (4, hidden),
        "stages.0.attn.vha.U": (2, 2),
        "stages.0.attn.attn_sink": (2,),
        "stages.0.attn.o_gate.weight": (8, 4),
        "stages.0.attn.wo_a.weight": (2, 3, 6),
        "stages.0.attn_hc.fn": (24, hc * hidden),
        "stages.0.attn_hc.base": (24,),
        "stages.0.attn_hc.scale": (3,),
        "stages.1.ffn.gate.weight_1": (experts, hidden),
        "stages.1.ffn.shared_experts.down_proj.weight": (hidden, 4),
        "stages.1.ffn.switch_mlp.gate_proj.weight": (experts, 4, 4),
        "stages.0.main_proj.weight": (hidden, 3 * hidden),
        "stages.0.main_norm.weight": (hidden,),
        "markov_head.markov_w1.weight": (vocab, rank),
        "markov_head.markov_w2.weight": (vocab, rank),
        "stages.1.hc_head.fn": (hc, hc * hidden),
        "stages.1.hc_head.base": (hc,),
        "stages.1.norm.weight": (hidden,),
    }
    assert out["stages.0.attn_hc.scale"].tolist() == [1.0, 2.0, 3.0]
    # Transposed checkpoint tensors are stored row-major, not as strided views:
    # the hyper-connection projection is several times slower on a view.
    for key in (
        "stages.0.attn_hc.fn",
        "stages.1.hc_head.fn",
        "markov_head.markov_w1.weight",
    ):
        assert np.asarray(out[key]).flags["C_CONTIGUOUS"], key
    assert out["stages.1.ffn.switch_mlp.gate_proj.weight"][1, 0, 0].item() == 1.0
    # Already-mapped weights pass through unchanged.
    assert ernie5_5_dspark.Model.sanitize(context, out) is out


def _decode_one_by_one(model, prompt, count):
    lm = model.language_model
    cache = model.make_cache()
    token = _greedy(lm(mx.array([prompt]), cache=cache).logits[:, -1])
    tokens = []
    for _ in range(count):
        tokens.append(token.item())
        token = _greedy(lm(token[:, None], cache=cache).logits[:, -1])
    return tokens


def _decode_speculative(model, draft, prompt, count, block_size, history=None):
    lm = model.language_model
    cache = model.make_cache()
    output = lm(
        mx.array([prompt]),
        cache=cache,
        capture_layer_ids=list(draft.config.target_layer_ids),
    )
    hidden = mx.concatenate(output.hidden_states, axis=-1)
    tokens = [_greedy(output.logits[:, -1]).item()]
    draft.begin_request(history)
    draft.observe(tokens[0])
    for token, _ in _dflash_rounds(
        model,
        draft,
        cache,
        hidden,
        first_bonus=tokens[0],
        max_tokens=count,
        sampler=_greedy,
        draft_block_size=block_size,
        greedy_sampling=True,
        use_model_initial_block_size=False,
    ):
        tokens.append(int(token))
        draft.observe(int(token))
    return tokens


def test_dspark_rounds_reproduce_greedy_decoding():
    # Thirty tokens cross the tiny model's window (4) and compression ratio (8)
    # several times, so rejected drafts roll both caches back.
    config, model = _target()
    prompt = [5, 9, 2, 44, 7, 81]
    reference = _decode_one_by_one(model, prompt, 30)
    for block_size in (2, 4):
        draft = _drafter(config, block_size)
        tokens = _decode_speculative(model, draft, prompt, 30, block_size)
        assert tokens == reference
        assert len(draft.accept_lens) > 0
        assert all(0 <= n <= block_size - 1 for n in draft.accept_lens)


@pytest.mark.parametrize("rows", [1, 4])
def test_layers_handed_to_the_gpu_as_they_are_built_change_nothing(rows, monkeypatch):
    config, model = _target()
    model.eval()
    lm = model.language_model
    prompt = mx.array([[5, 9, 2, 44, 7, 81, 3, 12]])
    block = mx.array([[21 + i for i in range(rows)]])
    logits = []
    for every in (0, 1):
        monkeypatch.setattr(language, "SUBMIT_LAYERS", every)
        cache = model.make_cache()
        mx.eval(lm(prompt, cache=cache).logits)
        logits.append(lm(block, cache=cache).logits)
        mx.eval(logits[-1])
    assert mx.array_equal(logits[0], logits[1])


def test_only_decode_and_verify_blocks_hand_layers_over(monkeypatch):
    config, model = _target()
    model.eval()
    lm = model.language_model
    handed = []
    monkeypatch.setattr(
        language, "_submit", lambda index, caches, *arrays: handed.append(index)
    )
    cache = model.make_cache()
    wide = language.DECODE_BLOCK_SIZE + 1
    mx.eval(lm(mx.array([[3 + i for i in range(wide)]]), cache=cache).logits)
    assert handed == []
    mx.eval(lm(mx.array([[40, 41]]), cache=cache).logits)
    assert handed == list(range(len(lm.model.layers)))


def test_a_layer_is_handed_over_with_its_cache_state(monkeypatch):
    """A cache update left lazy keeps every earlier step's Metal event alive."""
    config, model = _target()
    model.eval()
    lm = model.language_model
    cache = model.make_cache()
    mx.eval(lm(mx.array([[5, 9, 2, 44]]), cache=cache).logits)
    calls = []
    monkeypatch.setattr(mx, "async_eval", lambda *args: calls.append(args))
    logits = lm(mx.array([[40, 41]]), cache=cache).logits
    monkeypatch.undo()
    mx.eval(logits)
    assert len(calls) == len(lm.model.layers)
    for (arrays, states), layer_cache in zip(calls, cache):
        assert arrays and all(isinstance(a, mx.array) for a in arrays)
        want = len(tree_flatten(layer_cache.state))
        assert want and len(tree_flatten(states)) == want


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16, mx.float16])
def test_collapse_kernel_reads_parameters_of_any_float_type(dtype):
    """A model cast to half precision hands the kernel half-precision factors."""
    if hyper_connection._hc_sinkhorn_collapse_kernel is None:
        pytest.skip("needs the GPU kernel")
    mx.random.seed(3)
    x = (mx.random.normal((1, 3, 4, 64)) * 0.5).astype(mx.bfloat16)
    mixes = mx.random.normal((1, 3, 24))
    scale = mx.array([1.1, 0.9, 1.2]).astype(dtype)
    base = (mx.random.normal((24,)) * 0.5).astype(dtype)
    want = hyper_connection._hc_ops(
        x, x.astype(mx.float32), mixes, scale, base, 4, 20, 1e-6
    )
    got = hyper_connection._hc_kernel(x, None, mixes, scale, base, 4, 20, 1e-6)
    for kernel, ops in zip(got, want):
        assert mx.allclose(
            kernel.astype(mx.float32), ops.astype(mx.float32), atol=2e-2, rtol=2e-2
        )


def test_drafter_rejects_a_mismatched_target():
    config, model = _target()
    other = dict(config, hidden_size=32)
    draft = _drafter(other)
    try:
        draft.bind(model)
    except ValueError as error:
        assert "incompatible" in str(error)
    else:
        raise AssertionError("expected an incompatible target to be rejected")


@pytest.mark.parametrize("rows", [1, 2, 3, 8, 11])
def test_project_head_reads_a_half_precision_weight_in_place(rows):
    _, model = _target()
    lm = model.language_model
    weight = lm.lm_head.weight.astype(mx.bfloat16)
    lm.lm_head.weight = weight
    hidden = mx.random.normal((1, rows, 64)).astype(mx.bfloat16)

    # The product of the fp32 hidden states with the weight's exact values.
    expected = hidden.astype(mx.float32) @ weight.astype(mx.float32).T
    logits = lm.project_head(hidden)
    assert logits.dtype == mx.float32
    assert logits.shape == (1, rows, 128)
    assert mx.allclose(logits, expected, atol=1e-5, rtol=1e-5)
    assert mx.allclose(
        lm._project_logits(hidden), lm.multimax(expected), atol=1e-5, rtol=1e-5
    )


def test_project_head_follows_a_reloaded_weight():
    _, model = _target()
    lm = model.language_model
    weight = lm.lm_head.weight.astype(mx.bfloat16)
    hidden = mx.random.normal((2, 1, 64)).astype(mx.bfloat16)
    lm.lm_head.weight = weight
    first = lm.project_head(hidden)
    lm.lm_head.weight = (weight.astype(mx.float32) * 2).astype(mx.bfloat16)
    assert mx.allclose(lm.project_head(hidden), 2 * first, atol=1e-5, rtol=1e-5)


def test_project_head_leaves_a_float32_weight_alone():
    _, model = _target()
    lm = model.language_model
    hidden = mx.random.normal((1, 2, 64))
    assert mx.array_equal(lm.project_head(hidden), lm.lm_head(hidden))


def _stream(rows, seed=3):
    mx.random.seed(seed)
    return mx.random.normal((1, rows, 4, 64))


@pytest.mark.parametrize("rows", [1, 3, 8])
def test_compiled_short_block_paths_match_the_plain_ops(rows):
    """Decode-sized calls take compiled paths; they must not change the numbers."""
    _, model = _target()
    layer = model.language_model.model.layers[0]
    x = _stream(rows)

    mixed = layer.attn_hc(x)
    plain = layer.attn_hc._mix(x)
    for got, want in zip(mixed, plain):
        assert mx.allclose(got, want, atol=1e-6, rtol=1e-6)

    h = x[:, :, 0]
    # Layer 0 is the dense one; layer 1 routes experts.
    ffn = model.language_model.model.layers[1].ffn
    assert mx.allclose(
        ffn(h), ffn._forward(h, ffn.e_score_correction_bias), atol=1e-6, rtol=1e-6
    )


def test_router_scores_match_the_two_product_form():
    _, model = _target()
    gate = model.language_model.model.layers[1].ffn.gate
    x = mx.random.normal((1, 3, 64))
    stacked = gate._scores(x, gate.stacked())
    separate = gate._scores(x, None)
    assert mx.allclose(stacked, separate, atol=1e-6, rtol=1e-6)
    # A long chunk takes the GEMM form of the same product.
    wide = mx.random.normal((1, 40, 64))
    assert mx.allclose(
        gate._scores(wide, gate.stacked()), gate._scores(wide, None), atol=1e-6
    )
    # The stacked copy follows a weight reload.
    gate.weight = gate.weight * 2
    assert mx.allclose(
        gate._scores(x, gate.stacked()), gate._scores(x, None), atol=1e-6
    )


def test_virtual_heads_fold_follows_a_weight_reload():
    _, model = _target()
    vha = model.language_model.model.layers[0].attn.vha
    x = mx.random.normal((1, 2, 4, 16))

    def reference():
        mix = mx.eye(vha.U.shape[0]) + vha.V @ vha.U.T
        return mix @ x

    assert mx.allclose(vha(x), reference(), atol=1e-6)
    vha.U = vha.U * 3
    assert mx.allclose(vha(x), reference(), atol=1e-6)


def test_depth_policy_follows_acceptance():
    policy = DepthPolicy(7)
    start = policy.proposals()
    assert 2 <= start <= 4
    # Whole blocks accepted: the policy drafts deeper, up to the block.
    for _ in range(40):
        drafted = policy.proposals()
        policy.update(drafted, drafted)
    assert policy.proposals() == 7
    assert policy.proposals(3) == 3
    # Nothing accepted: it falls back to a single proposal.
    for _ in range(40):
        policy.update(0, policy.proposals())
    assert policy.proposals() == 1
    # Reaching a depth implies reaching the one before it.
    assert all(a >= b for a, b in zip(policy.reach, policy.reach[1:]))


def test_depth_policy_settles_where_a_proposal_stops_paying():
    # The first proposal is always accepted and the second never is.
    policy = DepthPolicy(7)
    drafted = []
    for _ in range(300):
        drafted.append(policy.proposals())
        policy.update(1, drafted[-1])
    settled = drafted[100:]
    # A fully accepted one-token block keeps suggesting a second proposal, so
    # it is retried, but it never drafts deeper than that.
    assert max(settled) <= 2
    assert settled.count(1) > 0


def test_copy_index_continues_an_earlier_occurrence():
    index = CopyIndex(order=3)
    index.extend([1, 2, 3, 4, 5, 6, 7, 8])
    assert index.continuation(4) == []
    index.extend([9, 2, 3, 4])
    # "2 3 4" was followed by "5 6 7 8 9 ..." the first time.
    assert index.continuation(3) == [5, 6, 7]
    assert index.continuation(10) == [5, 6, 7, 8, 9, 2, 3, 4]
    index.append(5)
    assert index.continuation(2) == [6, 7]
    index.append(40)
    assert index.continuation(2) == []


def test_copy_index_needs_a_full_window():
    index = CopyIndex(order=6)
    index.extend([1, 2, 3, 1, 2, 3])
    assert index.continuation(4) == []


@pytest.mark.parametrize("adaptive,copy", [(True, False), (True, True), (False, True)])
def test_drafting_policies_do_not_change_greedy_output(adaptive, copy):
    """Depth and copy drafts only choose what is verified, never what is emitted."""
    config, model = _target()
    # A prompt that repeats itself, so copy candidates exist.
    prompt = [5, 9, 2, 44, 7, 81, 3, 5, 9, 2, 44, 7, 81, 3, 5, 9, 2, 44, 7, 81]
    reference = _decode_one_by_one(model, prompt, 30)
    draft = _drafter(config, 8)
    draft.adaptive_depth = adaptive
    draft.copy_drafts = copy
    tokens = _decode_speculative(
        model, draft, prompt, 30, 8, history=prompt if copy else None
    )
    assert tokens == reference
    assert all(1 <= n <= 7 for n in draft.draft_lens)
    if not adaptive and not copy:
        assert set(draft.draft_lens) == {7}


def _head_draft(draft, anchor, hidden, cache, block_size):
    """What the head drafts on its own, with copy drafting switched off."""
    saved, draft._copy_draft = draft._copy_draft, None
    try:
        return draft.draft_block(anchor, hidden, cache, block_size, _greedy)
    finally:
        draft._copy_draft = saved


def test_copy_draft_is_used_only_when_the_head_agrees():
    config, model = _target()
    draft = _drafter(config, 8)
    draft.reset(model)
    draft.begin_request([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6])
    block = draft.choose_next_block_size(8)
    assert block == 8
    assert draft._copy_draft == [7, 8, 9, 10, 11, 12, 1]

    hidden = mx.random.normal((1, 1, 2 * 64))
    cache = draft.make_cache()
    head_tokens = _head_draft(draft, 6, hidden, cache, 8)
    cache = draft.make_cache()
    tokens = draft.draft_block(6, hidden, cache, 8, _greedy)
    if head_tokens[0, 0].item() == 7:
        assert tokens.tolist() == [[7, 8, 9, 10, 11, 12, 1]]
        assert draft.copy_rounds == 1
    else:
        assert tokens.tolist() == head_tokens.tolist()
        assert draft.copy_rounds == 0
        assert draft._copy_draft is None


def test_round_loop_asks_the_drafter_for_its_block_size():
    class Chooser:
        prefer_requested_block_size = True

        def choose_next_block_size(self, block_total):
            return 5

    assert _dflash_next_block_size(Chooser(), 8, 100) == 5
    assert _dflash_next_block_size(Chooser(), 4, 100) == 4
    assert _dflash_next_block_size(Chooser(), 8, 3) == 3
    # Without the hook the stock behaviour is unchanged.
    stock = SimpleNamespace(prefer_requested_block_size=True)
    assert _dflash_next_block_size(stock, 8, 100) == 8


def _prefilled(model, length, seed):
    mx.random.seed(seed)
    cache = model.make_cache()
    model.language_model(mx.random.randint(1, 128, (1, length)), cache=cache)
    mx.eval([entry.state for entry in cache])
    return cache


def _cache_arrays(cache):
    return [
        value
        for entry in cache
        for _, value in tree_flatten(entry.state)
        if value is not None
    ]


@pytest.mark.parametrize(
    "contexts,blocks",
    [((37, 11), (3, 5)), ((5, 60, 130), (2, 4, 2)), ((9, 9, 9, 9), (2, 2, 2, 2))],
)
def test_verify_rows_match_each_request_verified_alone(contexts, blocks):
    # Blocks of two or more rows: the requests' rows share every token-wise
    # launch and each request attends over its own cache.
    _, model = _target()
    lm = model.language_model
    layers = [0, len(lm.layers) - 1]
    caches = [_prefilled(model, n, seed) for seed, n in enumerate(contexts)]
    mx.random.seed(99)
    inputs = [mx.random.randint(1, 128, (1, rows)) for rows in blocks]
    alone = []
    for cache, block in zip(caches, inputs):
        own = copy.deepcopy(cache)
        out = lm(block, cache=own, capture_layer_ids=layers, speculative_verify=True)
        alone.append((out, own))
    together = lm.speculative_verify_rows(inputs, caches, capture_layer_ids=layers)
    for index, ((logits, captured, transaction), (out, own)) in enumerate(
        zip(together, alone)
    ):
        assert mx.array_equal(logits, out.logits)
        assert all(mx.array_equal(a, b) for a, b in zip(captured, out.hidden_states))
        # Each request keeps a different part of its block.
        keep = 1 + index % blocks[index]
        transaction.commit([keep])
        out.gdn_states.commit([keep])
        ours, theirs = _cache_arrays(caches[index]), _cache_arrays(own)
        assert [a.shape for a in ours] == [b.shape for b in theirs]
        assert all(mx.array_equal(a, b) for a, b in zip(ours, theirs))


def test_verify_rows_leave_the_caches_alone_when_they_fail():
    _, model = _target()
    lm = model.language_model
    caches = [_prefilled(model, 20, 0), _prefilled(model, 30, 1)]
    before = [[mx.array(a) for a in _cache_arrays(cache)] for cache in caches]
    too_many = [mx.zeros((1, 5), dtype=mx.int32), mx.zeros((1, 5), dtype=mx.int32)]
    with pytest.raises(ValueError):
        lm.speculative_verify_rows(too_many, caches)
    # A transaction that is opened and abandoned restores its cache.
    blocks = [mx.zeros((1, 3), dtype=mx.int32), mx.zeros((1, 4), dtype=mx.int32)]
    for _, _, transaction in lm.speculative_verify_rows(blocks, caches):
        transaction.abort()
    for cache, held in zip(caches, before):
        now = _cache_arrays(cache)
        assert [a.shape for a in now] == [b.shape for b in held]
        assert all(mx.array_equal(a, b) for a, b in zip(now, held))


def test_request_state_keeps_two_requests_apart():
    config, model = _target()
    draft = _drafter(config, 8)
    draft.reset(model)
    draft.begin_request([1, 2, 3, 4, 5, 6, 7])
    first = draft.request_state()
    draft.reset(model)
    draft.begin_request(None)
    draft.accept_lens.append(2)
    draft.draft_lens.append(3)
    second = draft.request_state()
    draft.use_request(first, depth_floor=0.5)
    assert draft.accept_lens == [] and draft._copy.tokens == [1, 2, 3, 4, 5, 6, 7]
    assert draft.depth_floor == 0.5
    draft.observe(8)
    draft.use_request(second)
    assert draft.accept_lens == [2] and draft._copy is None
    assert draft.depth_floor is None
    assert first["_copy"].tokens[-1] == 8


def test_depth_floor_makes_a_proposal_earn_more():
    policy = DepthPolicy(7)
    policy.reach = [1.0, 0.8, 0.5, 0.33, 0.25, 0.1, 0.05, 0.02]
    assert policy.proposals() == 4
    assert policy.proposals(floor=0.3) == 3
    assert policy.proposals(floor=0.6) == 1


def _target_hiddens(config, lengths, seed):
    mx.random.seed(seed)
    width = config["hidden_size"] * 2  # two target layers feed the drafter
    return [mx.random.normal((1, length, width)) for length in lengths]


@pytest.mark.parametrize("blocks", [(2, 2), (4, 3), (2, 5, 1), (3, 1, 4)])
def test_draft_block_bases_match_each_request_drafted_alone(blocks):
    config, model = _target()
    draft = _drafter(config, 8)
    count = len(blocks)
    alone_caches = [draft.reset(model) for _ in blocks]
    shared_caches = [draft.reset(model) for _ in blocks]
    # A first round that reads some context, then one that reads the rows a
    # round committed.
    for seed, lengths in enumerate(((5, 3, 7)[:count], (1, 2, 3)[:count])):
        hiddens = _target_hiddens(config, lengths, seed)
        anchors = [11 + 5 * r + seed for r in range(count)]
        alone = [
            (
                draft._logits(
                    draft._hidden(
                        draft._inputs(mx.array([anchor]), size - 1), hidden, cache
                    )
                )
                if size > 1
                else None
            )
            for anchor, hidden, cache, size in zip(
                anchors, hiddens, alone_caches, blocks
            )
        ]
        together = draft.draft_block_bases(anchors, hiddens, shared_caches, blocks)
        for size, expected, got in zip(blocks, alone, together):
            if size == 1:
                assert got is None
                continue
            assert got.shape == expected.shape == (1, size - 1, config["vocab_size"])
            np.testing.assert_allclose(
                np.array(got), np.array(expected), rtol=1e-4, atol=1e-4
            )
    for expected, got in zip(alone_caches, shared_caches):
        for a, b in zip(expected, got):
            assert a.offset == b.offset
            if a.keys is None:
                assert b.keys is None
                continue
            np.testing.assert_allclose(
                np.array(b.keys.astype(mx.float32)),
                np.array(a.keys.astype(mx.float32)),
                rtol=1e-4,
                atol=1e-4,
            )


def test_draft_block_takes_its_base_from_a_shared_forward():
    config, model = _target()
    draft = _drafter(config, 8)
    hiddens = _target_hiddens(config, (4, 6), 3)
    anchors, blocks = [9, 21], [4, 5]
    alone = [
        draft.draft_block(anchor, hidden, draft.reset(model), size, _greedy)
        for anchor, hidden, size in zip(anchors, hiddens, blocks)
    ]
    caches = [draft.reset(model) for _ in blocks]
    bases = draft.draft_block_bases(anchors, hiddens, caches, blocks)
    calls = []
    stock = draft._hidden
    draft._hidden = lambda *args, **kwargs: calls.append(1) or stock(*args, **kwargs)
    together = [
        draft.draft_block(anchor, hidden, cache, size, _greedy, base=base)
        for anchor, hidden, cache, size, base in zip(
            anchors, hiddens, caches, blocks, bases
        )
    ]
    assert not calls  # the stages are not run again
    assert [t.tolist() for t in together] == [t.tolist() for t in alone]
