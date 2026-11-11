import os
from pathlib import Path

import pytest

from agent_tool_response_fault_tester.engine import load_scenario, record_ollama

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("TOOLFAULT_TEST_OLLAMA") != "1",
    reason="set TOOLFAULT_TEST_OLLAMA=1 to run a real local model",
)
def test_local_ollama_tool_call_smoke():
    scenario = load_scenario(Path(__file__).parents[1] / "examples/order-fulfillment/scenario.yaml")
    trace = record_ollama(
        scenario, model=os.getenv("TOOLFAULT_MODEL", "qwen3:4b"), max_turns=12, timeout=90
    )
    assert trace.agent == "ollama"
    assert trace.evaluation.verdict in {"pass", "fail", "inconclusive"}
    assert any(event.type == "model_tool_call" for event in trace.events)
