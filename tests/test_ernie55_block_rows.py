# SPDX-License-Identifier: Apache-2.0
"""ERNIE 5.5 block drafting for several requests that share a verify forward.

Each request keeps its own cache and drafter state; a round verifies all the
requests' blocks in one forward. Whatever the other requests do, a request
must decode what it decodes alone.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

pytest.importorskip("mlx_vlm.models.ernie5_5_moe")
helpers = pytest.importorskip("mlx_vlm.tests.test_ernie55_dspark")

from mlx_vlm.speculative.dflash import _dflash_rounds  # noqa: E402

from omlx.speculative.block_rows import SharedBlockRounds  # noqa: E402

PROMPTS = (
    [5, 9, 2, 44, 7, 81],
    [11, 3, 60, 61, 62, 8, 19, 100, 4],
    [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82],
)


def _greedy(logits):
    return mx.argmax(logits, axis=-1)


def _sample(logprobs):
    return mx.random.categorical(logprobs)


def _admit(rounds, model, draft, prompt, count, sampler=_greedy, greedy=True):
    lm = model.language_model
    cache = model.make_cache()
    output = lm(
        mx.array([prompt]),
        cache=cache,
        capture_layer_ids=list(draft.config.target_layer_ids),
    )
    hidden = mx.concatenate(output.hidden_states, axis=-1)
    first = int(sampler(output.logits[:, -1]).item())
    stream = rounds.add(
        cache, hidden, first, count, sampler, greedy=greedy, history=prompt
    )
    return stream, cache


def _drain(streams, outputs, steps=None):
    """Step every live stream, as the scheduler does, until all have ended."""
    live = dict(streams)
    done = 0
    while live and (steps is None or done < steps):
        for key in list(live):
            try:
                outputs[key].extend(next(live[key]))
            except StopIteration:
                del live[key]
        done += 1
    return live


@pytest.fixture(scope="module")
def target():
    return helpers._target()


def test_requests_drafting_together_decode_what_they_decode_alone(target):
    config, model = target
    count = 30
    reference = [helpers._decode_one_by_one(model, p, count) for p in PROMPTS]
    draft = helpers._drafter(config, 4)
    rounds = SharedBlockRounds(model, draft, block_size=4)
    streams, outputs = {}, {}
    for index, prompt in enumerate(PROMPTS):
        streams[index], _ = _admit(rounds, model, draft, prompt, count)
        outputs[index] = []
    _drain(streams, outputs)
    assert [outputs[i] for i in range(len(PROMPTS))] == reference
    # Every round verified all three, and the forward held their rows.
    assert rounds.rounds < count
    assert not rounds.rows


def test_requests_may_join_and_leave_between_rounds(target):
    config, model = target
    counts = (30, 12, 22)
    reference = [
        helpers._decode_one_by_one(model, p, n) for p, n in zip(PROMPTS, counts)
    ]
    draft = helpers._drafter(config, 4)
    rounds = SharedBlockRounds(model, draft, block_size=4)
    streams, outputs = {}, {0: [], 1: [], 2: []}
    streams[0], _ = _admit(rounds, model, draft, PROMPTS[0], counts[0])
    live = _drain(streams, outputs, steps=3)
    live[1], _ = _admit(rounds, model, draft, PROMPTS[1], counts[1])
    live = _drain(live, outputs, steps=2)
    live[2], _ = _admit(rounds, model, draft, PROMPTS[2], counts[2])
    _drain(live, outputs)
    assert [outputs[i] for i in range(3)] == reference
    assert not rounds.rows


def test_a_closed_request_leaves_the_others_undisturbed(target):
    config, model = target
    count = 24
    reference = helpers._decode_one_by_one(model, PROMPTS[0], count)
    draft = helpers._drafter(config, 4)
    rounds = SharedBlockRounds(model, draft, block_size=4)
    outputs = {0: [], 1: []}
    streams = {}
    streams[0], _ = _admit(rounds, model, draft, PROMPTS[0], count)
    streams[1], _ = _admit(rounds, model, draft, PROMPTS[1], count)
    live = _drain(streams, outputs, steps=3)
    live.pop(1).close()
    assert len(rounds.rows) == 1
    _drain(live, outputs)
    assert outputs[0] == reference


def test_the_caches_hold_exactly_the_emitted_tokens(target):
    config, model = target
    count = 17
    draft = helpers._drafter(config, 4)
    rounds = SharedBlockRounds(model, draft, block_size=4)
    streams, caches, outputs = {}, {}, {}
    for index, prompt in enumerate(PROMPTS[:2]):
        streams[index], caches[index] = _admit(rounds, model, draft, prompt, count)
        outputs[index] = []
    _drain(streams, outputs)
    for index, prompt in enumerate(PROMPTS[:2]):
        # The last token is sampled but not yet fed, as in plain decoding.
        held = caches[index][-1].caches[0].offset
        assert held == len(prompt) + count - 1


def test_one_sampled_request_matches_the_stock_round_loop(target):
    config, model = target
    count, prompt = 26, PROMPTS[1]
    lm = model.language_model

    def stock():
        draft = helpers._drafter(config, 4)
        cache = model.make_cache()
        output = lm(
            mx.array([prompt]),
            cache=cache,
            capture_layer_ids=list(draft.config.target_layer_ids),
        )
        hidden = mx.concatenate(output.hidden_states, axis=-1)
        tokens = [int(_sample(output.logits[:, -1]).item())]
        draft.begin_request(prompt)
        draft.observe(tokens[0])
        for token, _ in _dflash_rounds(
            model,
            draft,
            cache,
            hidden,
            first_bonus=tokens[0],
            max_tokens=count,
            sampler=_sample,
            draft_block_size=4,
            greedy_sampling=False,
            use_model_initial_block_size=False,
        ):
            tokens.append(int(token))
            draft.observe(int(token))
        return tokens

    def shared():
        draft = helpers._drafter(config, 4)
        rounds = SharedBlockRounds(model, draft, block_size=4)
        outputs = {0: []}
        stream, _ = _admit(
            rounds, model, draft, prompt, count, sampler=_sample, greedy=False
        )
        _drain({0: stream}, outputs)
        return outputs[0]

    mx.random.seed(7)
    expected = stock()
    mx.random.seed(7)
    assert shared() == expected
    assert len(expected) == count


def test_sampled_requests_keep_their_own_random_streams(target):
    config, model = target
    count = 20

    def run(prompts):
        draft = helpers._drafter(config, 4)
        rounds = SharedBlockRounds(model, draft, block_size=4)
        streams, outputs = {}, {}
        mx.random.seed(11)
        for index, prompt in enumerate(prompts):
            streams[index], _ = _admit(
                rounds, model, draft, prompt, count, sampler=_sample, greedy=False
            )
            outputs[index] = []
        _drain(streams, outputs)
        return outputs

    together = run(PROMPTS[:2])
    assert all(len(tokens) == count for tokens in together.values())
    # The same two requests again draw the same numbers.
    assert run(PROMPTS[:2]) == together


@pytest.mark.parametrize("sampler,greedy", [(_greedy, True), (_sample, False)])
def test_one_drafter_forward_serves_the_requests_of_a_round(target, sampler, greedy):
    config, model = target
    count = 24

    def run(share):
        draft = helpers._drafter(config, 8)
        if not share:
            draft.draft_block_bases = None
        rounds = SharedBlockRounds(model, draft, block_size=8)
        drafted = []
        stock = rounds._draft

        def record(row, size, base=None):
            stock(row, size, base)
            drafted.append((row, row.draft_tokens.tolist()))

        rounds._draft = record
        mx.random.seed(11)
        streams, outputs = {}, {}
        for index, prompt in enumerate(PROMPTS):
            streams[index], _ = _admit(
                rounds, model, draft, prompt, count, sampler=sampler, greedy=greedy
            )
            outputs[index] = []
        order = {id(stream.row): index for index, stream in streams.items()}
        _drain(streams, outputs)
        return (
            outputs,
            [(order[id(row)], tokens) for row, tokens in drafted],
            rounds,
        )

    outputs, drafted, rounds = run(share=True)
    apart_outputs, apart_drafted, apart_rounds = run(share=False)
    # All but the first round, which reads each prompt into the drafter, and
    # the rounds after the other requests have finished.
    assert rounds.shared_drafts > rounds.rounds // 2
    assert apart_rounds.shared_drafts == 0
    # The requests are drafted the same tokens either way.
    assert drafted == apart_drafted
    assert outputs == apart_outputs


def test_block_sizes_fit_the_verify_forward(target):
    config, model = target
    draft = helpers._drafter(config, 8)
    rounds = SharedBlockRounds(model, draft, block_size=8)
    for prompt in PROMPTS:
        _admit(rounds, model, draft, prompt, 40)
    sizes = rounds._block_sizes(list(rounds.rows))
    assert sum(sizes) <= rounds.limit
    assert all(size >= 2 for size in sizes)
    assert rounds.max_requests == 3
    assert not rounds.has_room()


@pytest.mark.parametrize(
    "drafting,plain,peers,contended",
    [
        (0, 0, 0, False),  # alone
        (0, 0, 1, False),  # two starting together draft together
        (0, 0, 2, True),  # three starting together batch without drafting
        (1, 0, 0, False),  # joins a request that is drafting
        (2, 0, 1, False),  # and a third may join two
        (3, 0, 0, True),  # the rounds are full
        (1, 1, 0, True),  # a peer already decodes without drafting
        (0, 1, 0, True),
    ],
)
def test_route_lets_a_few_requests_draft_together(
    monkeypatch, caplog, drafting, plain, peers, contended
):
    import logging
    from types import SimpleNamespace

    import omlx.scheduler as scheduler_mod
    from omlx.scheduler import Scheduler

    monkeypatch.setattr(
        scheduler_mod,
        "shared_block_rounds",
        lambda *args, **kwargs: SimpleNamespace(max_requests=3),
    )
    request = SimpleNamespace(
        request_id="req",
        sampling_params=SimpleNamespace(
            max_tokens=64, stop_token_ids=None, logprobs=False
        ),
        rope_deltas=0.0,
    )
    sched = SimpleNamespace(
        _vlm_mtp_drafter=object(),
        _vlm_mtp_active={-i: object() for i in range(1, drafting + 1)},
        waiting=[object()] * peers,
        running={f"r{i}": object() for i in range(drafting + plain)},
        prefilling=[request],
        model=SimpleNamespace(),
    )
    with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
        uid = Scheduler._route_to_vlm_mtp(
            sched, request, [object()], [42], lambda logits: logits, object()
        )
    # The stub model has no language model, so a request that passes the
    # gate stops at the next check.
    assert uid is None
    assert ("scheduler contention" in caplog.text) is contended
    assert ("rollback_speculative_cache" in caplog.text) is not contended
    assert "drafter is busy" not in caplog.text
