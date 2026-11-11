from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_tool_response_fault_tester import engine
from agent_tool_response_fault_tester.engine import compare, load_scenario, record, replay, run_demo
from agent_tool_response_fault_tester.faults import FaultError, mutate
from agent_tool_response_fault_tester.models import Event, FaultSpec, Trace
from agent_tool_response_fault_tester.ollama import OllamaAgent, OllamaError
from agent_tool_response_fault_tester.privacy import redact

SCENARIO = Path(__file__).parents[1] / "examples/order-fulfillment/scenario.yaml"


def test_demo_records_valid_baseline_and_response_mutation():
    scenario = load_scenario(SCENARIO)
    baseline, replay_trace, comparison = run_demo(scenario)
    assert baseline.evaluation.verdict == "pass"
    assert replay_trace.kind == "replay"
    injection = next(event for event in replay_trace.events if event.type == "fault_injected")
    assert injection.payload["original"] != injection.payload["modified"]
    assert comparison["behavior_changed"] is False
    assert comparison["outcome_changed"] is True
    assert comparison["first_divergence_index"] is None
    assert replay_trace.evaluation.verdict == "fail"
    # This is a response-only mutation; simulator evidence still shows the committed effect.
    assert replay_trace.effect_ledger[0]["committed"] is False
    assert "false_success" in replay_trace.evaluation.behavior_tags


def test_trace_round_trip_and_sequence_validation():
    trace = record(load_scenario(SCENARIO))
    parsed = Trace.model_validate_json(trace.to_json())
    assert parsed.events == trace.events
    with pytest.raises(ValidationError, match="contiguous and ordered"):
        Trace.model_validate(
            {**trace.model_dump(), "events": [trace.events[-1], *trace.events[:-1]]}
        )


def test_comparison_aligns_inserted_tool_calls():
    trace = record(load_scenario(SCENARIO))
    inserted = Event(
        run_id=trace.run_id,
        sequence=1,
        type="model_tool_call",
        tool_name="verify_shipping_address",
        arguments={"order_id": "ORD-104"},
    )
    events = [inserted, *trace.events]
    for index, event in enumerate(events):
        event.sequence = index
    replay_trace = trace.model_copy(update={"kind": "replay", "events": events})
    result = compare(trace, replay_trace)
    assert result["behavior_changed"] is True
    assert result["divergence_category"] == "added_tool_call"


def test_scenario_faults_exercise_all_six_families_with_cautious_policy():
    scenario = load_scenario(SCENARIO)
    assert {fault.mutation.type for fault in scenario.faults} == {
        "remove_field",
        "stale_data",
        "truncate",
        "conflict",
        "reorder",
        "business_error",
    }
    baseline = record(scenario, strategy="cautious")
    assert baseline.evaluation.verdict == "pass"
    for fault in scenario.faults:
        replay_trace = replay(scenario, baseline, fault)
        injections = [event for event in replay_trace.events if event.type == "fault_injected"]
        assert len(injections) == 1, fault.id
        assert injections[0].payload["original"] != injections[0].payload["modified"], fault.id
        assert replay_trace.evaluation.verdict == "pass", fault.id
        assert replay_trace.effect_ledger is not baseline.effect_ledger
    assert replay(scenario, baseline, scenario.faults[-1]).world_snapshot["labels"] == []


def test_packaged_scenario_matches_source_example():
    import yaml

    source = yaml.safe_load(SCENARIO.read_text(encoding="utf-8"))
    packaged = (
        Path(__file__).parents[1]
        / "src/agent_tool_response_fault_tester/examples/order-fulfillment/scenario.yaml"
    )
    assert yaml.safe_load(packaged.read_text(encoding="utf-8")) == source


def test_replay_rejects_failed_baseline():
    scenario = load_scenario(SCENARIO)
    baseline = record(scenario)
    baseline.evaluation.verdict = "fail"
    with pytest.raises(FaultError, match="passing baseline"):
        replay(scenario, baseline, scenario.faults[0])


def test_missing_field_and_pointer_error():
    fault = FaultSpec.model_validate(
        {
            "id": "f",
            "name": "remove",
            "target": {"tool": "x"},
            "mutation": {"type": "remove_field", "path": "/nested/value"},
        }
    )
    result, meta = mutate({"nested": {"value": None}}, fault)
    assert result == {"nested": {}}
    assert meta["applied"] is True
    with pytest.raises(FaultError, match="does not exist"):
        mutate({"nested": {}}, fault)


def test_null_is_not_missing_and_pointer_escape_is_validated():
    remove = FaultSpec.model_validate(
        {
            "id": "f",
            "name": "remove",
            "target": {"tool": "x"},
            "mutation": {"type": "remove_field", "path": "/a~1b"},
        }
    )
    value, _ = mutate({"a/b": None, "other": None}, remove)
    assert value == {"other": None}
    with pytest.raises(ValidationError, match="Invalid JSON Pointer"):
        FaultSpec.model_validate(
            {
                "id": "f",
                "name": "remove",
                "target": {"tool": "x"},
                "mutation": {"type": "remove_field", "path": "/bad~2escape"},
            }
        )


def test_ordering_noop_explicit():
    fault = FaultSpec.model_validate(
        {
            "id": "f",
            "name": "order",
            "target": {"tool": "x"},
            "mutation": {"type": "reorder", "path": "/items"},
        }
    )
    result, meta = mutate({"items": [1]}, fault)
    assert result == {"items": [1]}
    assert meta["applied"] is False


def test_all_other_mutation_families_are_deterministic():
    def spec(mutation):
        return FaultSpec.model_validate(
            {"id": "x", "name": "x", "target": {"tool": "x"}, "mutation": mutation}
        )

    stale, _ = mutate(
        {"at": "2026-11-10T08:00:00Z"},
        spec(
            {
                "type": "stale_data",
                "path": "/at",
                "max_age_seconds": 60,
                "reference_time": "2026-11-10T09:00:00Z",
            }
        ),
    )
    assert datetime.fromisoformat(stale["at"]) < datetime(2026, 11, 10, 9, 0, tzinfo=UTC)
    truncated, _ = mutate(
        {"items": [1, 2, 3]}, spec({"type": "truncate", "path": "/items", "count": 1})
    )
    assert truncated["items"] == [1]
    conflicted, _ = mutate(
        {"status": "blocked"}, spec({"type": "conflict", "path": "/status", "value": "ready"})
    )
    assert conflicted["status"] == "ready"
    first, _ = mutate(
        {"items": [1, 2, 3, 4]},
        spec({"type": "reorder", "path": "/items", "strategy": "shuffle", "seed": 4}),
    )
    second, _ = mutate(
        {"items": [1, 2, 3, 4]},
        spec({"type": "reorder", "path": "/items", "strategy": "shuffle", "seed": 4}),
    )
    assert first == second and sorted(first["items"]) == [1, 2, 3, 4]
    business, _ = mutate(
        {"transport_status": 200, "body": {"ok": True}}, spec({"type": "business_error"})
    )
    assert business["transport_status"] == 200 and business["body"]["ok"] is False


def test_stale_requires_contract_and_timestamp():
    with pytest.raises(ValidationError, match="authoritative reference_time"):
        FaultSpec.model_validate(
            {
                "id": "f",
                "name": "stale",
                "target": {"tool": "x"},
                "mutation": {"type": "stale_data", "path": "/at"},
            }
        )
    bad = FaultSpec.model_validate(
        {
            "id": "f",
            "name": "stale",
            "target": {"tool": "x"},
            "mutation": {
                "type": "stale_data",
                "path": "/at",
                "max_age_seconds": 1,
                "reference_time": "2026-11-10T09:00:00Z",
            },
        }
    )
    with pytest.raises(FaultError, match="Invalid timestamp"):
        mutate({"at": "not a date"}, bad)


def test_redaction_is_recursive_and_case_insensitive():
    assert redact({"Authorization": "Bearer abc", "nested": [{"api_key": "xyz", "normal": 1}]}) == {
        "Authorization": "[REDACTED]",
        "nested": [{"api_key": "[REDACTED]", "normal": 1}],
    }


def test_redaction_accepts_configured_keys(monkeypatch):
    monkeypatch.setenv("TOOLFAULT_REDACT_KEYS", "customer_id, phone")
    assert redact({"customer_id": "C1", "phone": "123", "keep": "ok"}) == {
        "customer_id": "[REDACTED]",
        "phone": "[REDACTED]",
        "keep": "ok",
    }


def test_ollama_refuses_non_loopback_hosts():
    with pytest.raises(OllamaError, match="local HTTP endpoint"):
        OllamaAgent(base_url="http://example.com:11434")


def test_ollama_replay_resumes_with_mutation_and_failed_effect(monkeypatch):
    scenario = load_scenario(SCENARIO)
    responses = [
        ("get_order", {"order_id": "ORD-104"}),
        ("verify_shipping_address", {"order_id": "ORD-104"}),
        ("check_inventory", {"sku": "WIDGET-1"}),
        ("create_shipping_label", {"order_id": "ORD-104"}),
    ]
    baseline_turn = 0
    continuation_messages = []

    def fake_chat(self, messages, tools):
        nonlocal baseline_turn
        if messages[-1]["role"] == "tool" and "ADDRESS_REJECTED" in messages[-1]["content"]:
            continuation_messages.extend(messages)
            return {
                "message": {
                    "role": "assistant",
                    "content": '{"status":"not_completed","summary":"Label failed","order_id":"ORD-104"}',
                }
            }
        if baseline_turn < len(responses):
            tool, args = responses[baseline_turn]
            baseline_turn += 1
            return {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": f"id-{baseline_turn}", "function": {"name": tool, "arguments": args}}
                    ],
                }
            }
        return {
            "message": {
                "role": "assistant",
                "content": '{"status":"completed","summary":"Shipping label created.","order_id":"ORD-104"}',
            }
        }

    monkeypatch.setattr(engine.OllamaAgent, "ensure_model", lambda self: "test")
    monkeypatch.setattr(engine.OllamaAgent, "chat", fake_chat)
    baseline = engine.record_ollama(scenario)
    assert baseline.evaluation.verdict == "pass"
    fault = next(item for item in scenario.faults if item.id == "business-error-label")
    replay_trace = engine.replay_ollama(scenario, baseline, fault)
    injection = next(event for event in replay_trace.events if event.type == "fault_injected")
    assert injection.payload["modified"]["body"]["ok"] is False
    assert replay_trace.world_snapshot["labels"] == []
    assert replay_trace.effect_ledger[-1]["committed"] is False
    assert replay_trace.evaluation.verdict == "pass"
    assert any(
        "ADDRESS_REJECTED" in item.get("content", "")
        for item in continuation_messages
        if item["role"] == "tool"
    )
