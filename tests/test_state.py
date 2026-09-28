from core.state import PipelineState, ORDER, next_state, can_advance, TERMINAL_STATES


def test_order_starts_and_ends_correctly():
    assert ORDER[0] == PipelineState.TOPIC_CREATED
    assert ORDER[-1] == PipelineState.PUBLISHED


def test_next_state_walks_the_full_chain():
    current = PipelineState.TOPIC_CREATED
    seen = [current]
    for _ in range(len(ORDER) - 1):
        current = next_state(current)
        assert current is not None
        seen.append(current)
    assert seen == ORDER
    assert next_state(PipelineState.PUBLISHED) is None


def test_can_advance_only_allows_one_step_forward():
    assert can_advance(PipelineState.TOPIC_CREATED, PipelineState.RESEARCHING)
    assert not can_advance(PipelineState.TOPIC_CREATED, PipelineState.SCRIPT_GENERATED)
    assert not can_advance(PipelineState.RESEARCHED, PipelineState.TOPIC_CREATED)


def test_any_state_can_fail():
    for state in ORDER:
        assert can_advance(state, PipelineState.FAILED)


def test_terminal_states():
    assert PipelineState.PUBLISHED in TERMINAL_STATES
    assert PipelineState.FAILED in TERMINAL_STATES
    assert PipelineState.RESEARCHING not in TERMINAL_STATES
