# SPDX-License-Identifier: Apache-2.0
"""Block drafters on the vlm_mtp path (ERNIE 5.5 Flash Preview's DSpark head).

The vlm_mtp path hands a request to mlx-vlm's own round loop. An MTP head
uses ``_mtp_rounds``; a block drafter uses ``_dflash_rounds`` and is seeded
with the target's captured layer outputs instead of its final hidden state.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

import omlx.scheduler as scheduler_module
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler
from omlx.speculative import vlm_mtp


def _fake_drafter_model(model_type: str, target_layer_ids=(40, 41, 42)):
    model = MagicMock()
    model.config = SimpleNamespace(
        model_type=model_type, target_layer_ids=list(target_layer_ids)
    )
    model.accept_lens = []
    return model


def test_load_accepts_the_ernie_dspark_head():
    fake_model = _fake_drafter_model("ernie5_5_dspark")
    with patch.object(
        vlm_mtp, "_vlm_load_drafter", return_value=(fake_model, "dflash")
    ):
        drafter = vlm_mtp.load_vlm_mtp_drafter("/models/ernie")
    assert isinstance(drafter, vlm_mtp.VLMMTPDrafter)
    assert drafter.draft_kind == "dflash"
    assert drafter.default_block_size == 6


def test_load_still_rejects_other_block_drafters():
    # Other DFlash drafters run on Lightning MTP, not on this path.
    fake_model = _fake_drafter_model("qwen3_dflash")
    with patch.object(
        vlm_mtp, "_vlm_load_drafter", return_value=(fake_model, "dflash")
    ):
        assert vlm_mtp.load_vlm_mtp_drafter("/models/draft") is None


def test_mtp_drafters_have_no_default_block_size():
    drafter = vlm_mtp.VLMMTPDrafter(_fake_drafter_model("qwen3_5_mtp"), "mtp", "/p")
    assert drafter.default_block_size is None


@pytest.mark.parametrize(
    "config,bundled",
    [
        ({"model_type": "ernie5_5_moe", "dspark_target_layer_ids": [40, 41, 42]}, True),
        # The Base checkpoint ships ECHO, which this path does not serve.
        ({"model_type": "ernie5_5_moe"}, False),
        ({"model_type": "qwen3_5", "dspark_target_layer_ids": [1]}, False),
        ([], False),
    ],
)
def test_bundled_block_drafter_path(tmp_path, config, bundled):
    (tmp_path / "config.json").write_text(json.dumps(config))
    expected = str(tmp_path) if bundled else None
    assert vlm_mtp.bundled_block_drafter_path(tmp_path) == expected


def test_bundled_block_drafter_path_without_a_config(tmp_path):
    assert vlm_mtp.bundled_block_drafter_path(tmp_path) is None
    assert vlm_mtp.bundled_block_drafter_path(None) is None


def test_decode_dispatches_a_block_drafter_to_dflash_rounds():
    drafter = vlm_mtp.VLMMTPDrafter(
        _fake_drafter_model("ernie5_5_dspark"), "dflash", "/p"
    )
    target = SimpleNamespace()
    sampler = MagicMock()
    hidden = mx.zeros((1, 3, 12))
    captured = {}

    def fake_rounds(model, draft_model, prompt_cache, seen_hidden, **kwargs):
        captured.update(kwargs, model=model, draft_model=draft_model, hidden=seen_hidden)
        yield 11, None
        yield 22, None

    with (
        patch.object(vlm_mtp, "_dflash_rounds", side_effect=fake_rounds),
        patch.object(vlm_mtp, "_mtp_rounds") as mtp_rounds,
        patch.object(vlm_mtp, "_sync_and_clear_cache") as clear,
    ):
        tokens = list(
            vlm_mtp.run_vlm_mtp_decode(
                target_language_model=target,
                drafter=drafter,
                prompt_cache=[],
                hidden=hidden,
                shared_kv_states={},
                first_bonus=mx.array([7]),
                max_tokens=16,
                sampler=sampler,
                draft_block_size=6,
                greedy=True,
            )
        )

    assert tokens == [7, 11, 22]
    mtp_rounds.assert_not_called()
    clear.assert_not_called()
    assert captured["model"] is target
    assert captured["draft_model"] is drafter.model
    assert captured["hidden"] is hidden
    assert captured["first_bonus"] == 7
    assert captured["draft_block_size"] == 6
    assert captured["greedy_sampling"] is True
    assert captured["sampler"] is sampler


def test_decode_drains_the_buffer_pool_every_few_dozen_rounds():
    model = _fake_drafter_model("ernie5_5_dspark")
    drafter = vlm_mtp.VLMMTPDrafter(model, "dflash", "/p")
    rounds = 2 * vlm_mtp._BLOCK_DRAFT_CLEAR_ROUNDS + 5

    def fake_rounds(*args, **kwargs):
        # mlx-vlm rebinds the stats lists when the round loop starts.
        model.accept_lens = []
        for index in range(rounds):
            model.accept_lens.append(1)
            yield index, None
            yield index, None

    with (
        patch.object(vlm_mtp, "_dflash_rounds", side_effect=fake_rounds),
        patch.object(vlm_mtp, "_sync_and_clear_cache") as clear,
    ):
        tokens = list(
            vlm_mtp.run_vlm_mtp_decode(
                target_language_model=SimpleNamespace(),
                drafter=drafter,
                prompt_cache=[],
                hidden=mx.zeros((1, 1, 4)),
                shared_kv_states={},
                first_bonus=1,
                max_tokens=1000,
                sampler=MagicMock(),
            )
        )

    assert len(tokens) == 1 + 2 * rounds
    assert clear.call_count == 2


def test_decode_refuses_a_batch_for_a_block_drafter():
    drafter = vlm_mtp.VLMMTPDrafter(
        _fake_drafter_model("ernie5_5_dspark"), "dflash", "/p"
    )
    with pytest.raises(ValueError, match="one request at a time"):
        list(
            vlm_mtp.run_vlm_mtp_decode(
                target_language_model=SimpleNamespace(),
                drafter=drafter,
                prompt_cache=[],
                hidden=mx.zeros((2, 1, 4)),
                shared_kv_states={},
                first_bonus=mx.array([1, 2]),
                max_tokens=8,
                sampler=MagicMock(),
            )
        )


class _CapturingLanguageModel:
    """Returns two layer captures for every position it is called with."""

    def __init__(self):
        self.calls = []

    def rollback_speculative_cache(self):
        pass

    def __call__(self, tokens, cache=None, **kwargs):
        self.calls.append(kwargs)
        length = tokens.shape[1]
        logits = mx.zeros((1, length, 5))
        logits[:, -1, 2] = 5.0
        return SimpleNamespace(
            logits=logits,
            hidden_states=[
                mx.full((1, length, 4), float(layer))
                for layer in kwargs["capture_layer_ids"]
            ],
        )


class _Adapter:
    def __init__(self):
        self._language_model = _CapturingLanguageModel()

    def __call__(self, *args, **kwargs):
        return self._language_model(*args, **kwargs)


@pytest.mark.parametrize("temperature,greedy", [(0.0, True), (0.7, False)])
def test_scheduler_seeds_a_block_drafter_with_layer_captures(
    mock_model, mock_tokenizer, temperature, greedy
):
    scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
    scheduler._vlm_mtp_drafter = vlm_mtp.VLMMTPDrafter(
        _fake_drafter_model("ernie5_5_dspark", target_layer_ids=(3, 4)), "dflash", "/p"
    )
    scheduler._vlm_mtp_draft_block_size = 6
    adapter = _Adapter()
    scheduler.model = adapter
    request = Request(
        request_id="req-block",
        prompt=[1],
        sampling_params=SamplingParams(max_tokens=8, temperature=temperature),
    )
    request.prompt_token_ids = [1, 2, 3]
    request.rope_deltas = 0.0
    cache = [SimpleNamespace(state=mx.array([0]))]
    captured = {}

    def fake_decode(**kwargs):
        captured.update(kwargs)
        return iter([kwargs["first_bonus"]])

    with patch.object(scheduler_module, "run_vlm_mtp_decode", side_effect=fake_decode):
        uid = scheduler._route_to_vlm_mtp(
            request,
            cache,
            [1, 2, 3],
            lambda logits: mx.argmax(logits, axis=-1),
            state_machine=object(),
        )

    assert uid is not None
    # The final forward asks for the drafter's layers, not the MTP hidden state.
    assert adapter._language_model.calls == [{"capture_layer_ids": [3, 4]}]
    # Every position's captures are kept and concatenated layer-wise.
    assert captured["hidden"].shape == (1, 3, 8)
    assert captured["hidden"][0, 0].tolist() == [3.0] * 4 + [4.0] * 4
    assert captured["first_bonus"] == 2
    assert captured["draft_block_size"] == 6
    assert captured["greedy"] is greedy
