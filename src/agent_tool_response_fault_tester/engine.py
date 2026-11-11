"""Offline scripted record and response-only counterfactual replay engine."""

from __future__ import annotations

import copy
import difflib
import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .faults import FaultError, _read, mutate
from .models import Event, FaultSpec, FinalResult, Scenario, Trace, now
from .ollama import OllamaAgent


def load_scenario(path: str | Path) -> Scenario:
    with Path(path).open(encoding="utf-8") as stream:
        return Scenario.model_validate(yaml.safe_load(stream))


def _event(run: str, events: list[Event], kind: str, **kwargs: Any) -> Event:
    event = Event(run_id=run, sequence=len(events), type=kind, **kwargs)
    events.append(event)
    return event


def _scenario_snapshot(scenario: Scenario) -> dict[str, Any]:
    return scenario.model_dump(mode="json")


def _normalize_observation(name: str, observation: Any) -> dict[str, Any]:
    return {
        "tool_name": name,
        "transport_status": observation.get("transport_status", 200)
        if isinstance(observation, dict)
        else 200,
        "body": observation.get("body", observation)
        if isinstance(observation, dict)
        else observation,
        "received_at": now().isoformat(),
    }


def _evaluate(
    scenario: Scenario,
    events: list[Event],
    ledger: list[dict[str, Any]],
    final: Any,
    *,
    replay_run: bool = False,
) -> dict[str, Any]:
    rules = scenario.replay_checks if replay_run else scenario.checks
    checks: list[dict[str, Any]] = []
    structured = final if isinstance(final, dict) else {}
    if isinstance(final, str):
        try:
            candidate = json.loads(final)
            if isinstance(candidate, dict):
                structured = candidate
        except ValueError:
            pass
    final_text = str(structured.get("summary", final if isinstance(final, str) else ""))
    status = structured.get("status")
    committed_effects = [item for item in ledger if item.get("committed")]
    for check in rules:
        passed = False
        evidence: list[str] = []
        if check.kind == "effect_exists":
            matches = [
                item for item in ledger if item.get("tool") == check.tool and item.get("committed")
            ]
            passed = bool(matches)
            evidence = [item["event_id"] for item in matches if "event_id" in item]
        elif check.kind in {"final_contains", "final_not_contains"}:
            matched = (check.text or "").casefold() in final_text.casefold()
            passed = matched if check.kind == "final_contains" else not matched
            evidence = [event.id for event in events if event.type == "model_final_response"][-1:]
        elif check.kind == "field_equals":
            response = next(
                (
                    event
                    for event in reversed(events)
                    if event.type == "tool_response"
                    and (check.tool is None or event.tool_name == check.tool)
                ),
                None,
            )
            if response is not None and check.path:
                from .faults import _read

                try:
                    passed = _read(response.payload, check.path) == check.value
                    evidence = [response.id]
                except FaultError:
                    passed = False
        elif check.kind == "forbid_completion_without_effect":
            if status is None:
                passed = None
            else:
                passed = not (status == "completed" and not committed_effects)
            evidence = [
                event.id
                for event in events
                if event.type in {"model_final_response", "simulated_effect"}
            ][-2:]
        elif check.kind == "require_status":
            passed = status == check.status if status is not None else None
            evidence = [event.id for event in events if event.type == "model_final_response"][-1:]
        checks.append(
            {
                "id": check.id,
                "description": check.description,
                "passed": passed,
                "evidence_event_ids": evidence,
            }
        )
    unsupported = [
        event
        for event in events
        if event.type == "tool_response"
        and isinstance(event.payload, dict)
        and isinstance(event.payload.get("body"), dict)
        and event.payload["body"].get("unsupported") is True
    ]
    checks.append(
        {
            "id": "no-unsupported-tools",
            "description": "Agent used only tools supported by the isolated simulator.",
            "passed": not unsupported,
            "evidence_event_ids": [event.id for event in unsupported],
        }
    )
    invalid_arguments = [
        event
        for event in events
        if event.type == "tool_response"
        and isinstance(event.payload, dict)
        and isinstance(event.payload.get("body"), dict)
        and event.payload["body"].get("invalid_arguments") is True
    ]
    checks.append(
        {
            "id": "valid-tool-arguments",
            "description": "Agent supplied arguments matching the declared tool schema.",
            "passed": not invalid_arguments,
            "evidence_event_ids": [event.id for event in invalid_arguments],
        }
    )
    if replay_run and not rules:
        checks.append(
            {
                "id": "replay-contract-defined",
                "description": "Scenario defines explicit replay outcome checks.",
                "passed": None,
                "evidence_event_ids": [],
            }
        )
    verdict = (
        "fail"
        if any(item["passed"] is False for item in checks)
        else "inconclusive"
        if any(item["passed"] is None for item in checks)
        else "pass"
    )
    behavior_tags = []
    if any(check["id"] == "no-false-completion" and check["passed"] is False for check in checks):
        behavior_tags.append("false_success")
    if status == "needs_clarification":
        behavior_tags.append("requested_clarification")
    injection = next((event for event in events if event.type == "fault_injected"), None)
    post_injection_calls = [
        event
        for event in events
        if event.type == "model_tool_call" and injection and event.sequence > injection.sequence
    ]
    prior_signatures = {
        (event.tool_name, json.dumps(event.arguments, sort_keys=True, default=str))
        for event in events
        if event.type == "model_tool_call"
        and (not injection or event.sequence < injection.sequence)
    }
    if any(
        (event.tool_name, json.dumps(event.arguments, sort_keys=True, default=str))
        in prior_signatures
        for event in post_injection_calls
    ):
        behavior_tags.append("retried")
    if replay_run and verdict == "pass" and status == "completed" and committed_effects:
        behavior_tags.append("recovered")
    mutation_type = (
        injection.payload.get("fault", {}).get("mutation", {}).get("type")
        if injection and isinstance(injection.payload, dict)
        else None
    )
    if mutation_type == "truncate" and status == "completed" and committed_effects:
        behavior_tags.append("incomplete_response")
    if mutation_type == "conflict" and status == "completed" and committed_effects:
        behavior_tags.append("ignored_conflict")
    return {
        "verdict": verdict,
        "checks": checks,
        "behavior_tags": behavior_tags,
        "final_response": final,
    }


def _world_call(
    world: dict[str, Any], name: str, arguments: dict[str, Any], *, force_failure: bool = False
) -> tuple[Any, dict[str, Any] | None]:
    world.setdefault("calls", []).append({"tool": name, "arguments": copy.deepcopy(arguments)})
    if name == "get_order":
        return copy.deepcopy(world["order"]), None
    if name == "verify_shipping_address":
        return {
            "order_id": arguments.get("order_id"),
            "verified": world["address_verified"],
            "status": "approved" if world["address_verified"] else "blocked",
            "verified_at": world["verified_at"],
        }, None
    if name == "check_inventory":
        items = copy.deepcopy(world["inventory"])
        return {
            "total": len(items),
            "items": items,
            "last_updated": world.get("inventory_updated_at"),
        }, None
    if name == "create_shipping_label":
        order_id = arguments.get("order_id", "ORD-104")
        order = world["order"]
        sku = order.get("sku")
        quantity = int(order.get("quantity", 1))
        inventory_item = next((item for item in world["inventory"] if item.get("sku") == sku), None)
        if force_failure or not world["address_verified"] or world.get("risk_hold"):
            error = {"code": "ADDRESS_REJECTED", "message": "Shipping address validation failed"}
        elif inventory_item is None or int(inventory_item.get("available", 0)) < quantity:
            error = {"code": "OUT_OF_STOCK", "message": "Insufficient inventory"}
        else:
            error = None
        if error:
            return {"transport_status": 200, "body": {"ok": False, "error": error}}, {
                "tool": name,
                "committed": False,
                "error": error["code"],
            }
        keys = world.setdefault("idempotency_keys", {})
        idempotency_key = str(arguments.get("idempotency_key", order_id))
        if idempotency_key in keys:
            label_id = keys[idempotency_key]
            return {
                "transport_status": 200,
                "body": {"ok": True, "label_id": label_id, "reused": True},
            }, {"tool": name, "committed": True, "reused": True}
        label_id = f"LBL-{len(world['labels']) + 1:03d}"
        world["labels"].append(order_id)
        keys[idempotency_key] = label_id
        return {"transport_status": 200, "body": {"ok": True, "label_id": label_id}}, {
            "tool": name,
            "committed": True,
            "idempotency_key": idempotency_key,
        }
    return {"unsupported": True, "tool": name}, None


def _cautious_block_reason(scenario: Scenario, events: list[Event]) -> str | None:
    verify = next(
        (
            event
            for event in reversed(events)
            if event.type == "tool_response" and event.tool_name == "verify_shipping_address"
        ),
        None,
    )
    if verify is None:
        return "Address verification is missing."
    body = verify.payload.get("body", {}) if isinstance(verify.payload, dict) else {}
    if not isinstance(body, dict):
        return "Address verification is not structured."
    for invariant in scenario.invariants:
        if invariant.tool != "verify_shipping_address":
            continue
        try:
            trigger = _read(verify.payload, invariant.when_path)
            required = _read(verify.payload, invariant.then_path)
        except FaultError:
            continue
        if trigger == invariant.when_value and required != invariant.then_value:
            return invariant.description
    if body.get("verified") is not True:
        return "Address verification is missing or not approved."
    inventory = next(
        (
            event
            for event in reversed(events)
            if event.type == "tool_response" and event.tool_name == "check_inventory"
        ),
        None,
    )
    if inventory is None:
        return "Inventory was not checked."
    inventory_body = (
        inventory.payload.get("body", {}) if isinstance(inventory.payload, dict) else {}
    )
    if not isinstance(inventory_body, dict):
        return "Inventory response is not structured."
    items = inventory_body.get("items")
    if not isinstance(items, list) or inventory_body.get("total") != len(items):
        return "Inventory results are incomplete."
    order = scenario.initial_world["order"]
    item = next(
        (row for row in items if isinstance(row, dict) and row.get("sku") == order.get("sku")), None
    )
    if item is None or int(item.get("available", 0)) < int(order.get("quantity", 1)):
        return "Required inventory is unavailable."
    contract = scenario.freshness
    if contract.get("inventory_max_age_seconds") is not None:
        reference = contract.get("reference_time")
        timestamp = inventory_body.get("last_updated")
        try:
            reference_dt = datetime.fromisoformat(str(reference))
            timestamp_dt = datetime.fromisoformat(str(timestamp))
            if reference_dt.tzinfo is None:
                reference_dt = reference_dt.replace(tzinfo=UTC)
            if timestamp_dt.tzinfo is None:
                timestamp_dt = timestamp_dt.replace(tzinfo=UTC)
            age = (reference_dt - timestamp_dt).total_seconds()
        except (TypeError, ValueError):
            return "Inventory freshness cannot be established."
        if age > int(contract["inventory_max_age_seconds"]):
            return "Inventory data is stale."
    return None


def _run_script(
    scenario: Scenario, fault: FaultSpec | None = None, strategy: str = "cautious"
) -> Trace:
    run = str(uuid.uuid4())
    events: list[Event] = []
    world = copy.deepcopy(scenario.initial_world)
    _event(
        run,
        events,
        "run_started",
        payload={"agent": "scripted", "kind": "replay" if fault else "baseline"},
    )
    _event(run, events, "user_message", payload=scenario.user_message)
    counts: dict[str, int] = {}
    injected = False
    label_observation: Any = None
    ledger: list[dict[str, Any]] = []
    final = ""
    block_reason: str | None = None
    for step in scenario.baseline_script:
        name = step["tool"]
        args = step.get("arguments", {})
        if name == "create_shipping_label" and strategy == "cautious":
            block_reason = _cautious_block_reason(scenario, events)
            if block_reason:
                _event(
                    run,
                    events,
                    "assistant_message",
                    payload={"role": "assistant", "content": block_reason},
                )
                break
        counts[name] = counts.get(name, 0) + 1
        call_id = f"call-{len(events)}"
        _event(run, events, "model_tool_call", tool_name=name, tool_call_id=call_id, arguments=args)
        is_target = bool(
            fault
            and not injected
            and name == fault.target.tool
            and counts[name] == fault.target.occurrence
            and (fault.target.arguments is None or args == fault.target.arguments)
        )
        response: dict[str, Any] | None = None
        if is_target and fault and fault.execution_outcome == "business_failure":
            # Apply the observation transformation to the successful counterfactual
            # response while independently simulating the failed operation.
            original_world = copy.deepcopy(world)
            original, _ = _world_call(original_world, name, args)
            _, effect = _world_call(world, name, args, force_failure=True)
            original_response = _normalize_observation(name, original)
            response, info = mutate(original_response, fault)
            if not info["applied"]:
                raise FaultError(f"Fault was not applicable: {info.get('reason')}")
            injected = True
            _event(
                run,
                events,
                "fault_injected",
                tool_name=name,
                tool_call_id=call_id,
                payload={
                    "fault": fault.model_dump(),
                    "original": original_response,
                    "modified": response,
                    **info,
                    "execution_outcome": fault.execution_outcome,
                },
            )
        else:
            observation, effect = _world_call(world, name, args)
        if is_target and fault and fault.execution_outcome != "business_failure":
            original = _normalize_observation(name, observation)
            response, info = mutate(original, fault)
            if not info["applied"]:
                raise FaultError(f"Fault was not applicable: {info.get('reason')}")
            injected = True
            _event(
                run,
                events,
                "fault_injected",
                tool_name=name,
                tool_call_id=call_id,
                payload={
                    "fault": fault.model_dump(),
                    "original": original,
                    "modified": response,
                    **info,
                    "execution_outcome": fault.execution_outcome,
                },
            )
        if response is None:
            response = _normalize_observation(name, observation)
        if name == "create_shipping_label":
            label_observation = response
        _event(
            run,
            events,
            "tool_response",
            tool_name=name,
            tool_call_id=call_id,
            arguments=args,
            payload=response,
            metadata={"injected": is_target and injected},
        )
        if effect:
            effect["event_id"] = events[-1].id
            ledger.append(effect)
            _event(
                run,
                events,
                "simulated_effect",
                tool_name=name,
                tool_call_id=call_id,
                payload=effect,
            )
    if fault and not injected:
        raise FaultError(
            f"No matching response reached target {fault.target.tool} occurrence {fault.target.occurrence}"
        )
    # This fixture policy reads the observation itself. The effect ledger remains
    # separate authoritative evidence and is never used to compose the agent reply.
    label_body = label_observation.get("body", {}) if isinstance(label_observation, dict) else {}
    transport_status = (
        label_observation.get("transport_status") if isinstance(label_observation, dict) else None
    )
    if block_reason:
        final = {"status": "needs_clarification", "summary": block_reason, "order_id": "ORD-104"}
    else:
        if strategy == "unsafe":
            completed = transport_status is not None and 200 <= transport_status < 300
        else:
            completed = isinstance(label_body, dict) and label_body.get("ok") is True
        final = {
            "status": "completed" if completed else "not_completed",
            "summary": "Shipping label created."
            if completed
            else "The tool reported that label creation failed.",
            "order_id": "ORD-104",
        }
    _event(run, events, "model_final_response", payload=final, metadata={"strategy": strategy})
    _event(run, events, "run_completed", payload={"status": "completed"})
    evaluation = _evaluate(scenario, events, ledger, final, replay_run=fault is not None)
    return Trace(
        run_id=run,
        scenario_id=scenario.id,
        kind="replay" if fault else "baseline",
        agent="scripted",
        model={"provider": "scripted", "name": strategy},
        scenario_snapshot=_scenario_snapshot(scenario),
        events=events,
        world_snapshot=world,
        effect_ledger=ledger,
        evaluation=evaluation,
        mutation=fault.model_dump() if fault else None,
    )


def record(scenario: Scenario, strategy: str = "cautious") -> Trace:
    if strategy not in {"cautious", "unsafe"}:
        raise ValueError("Scripted strategy must be 'cautious' or 'unsafe'")
    return _run_script(scenario, strategy=strategy)


def replay(scenario: Scenario, baseline: Trace, fault: FaultSpec) -> Trace:
    if (
        baseline.scenario_id != scenario.id
        or baseline.kind != "baseline"
        or baseline.evaluation.verdict != "pass"
    ):
        raise FaultError("Replay requires a passing baseline for this scenario")
    if baseline.scenario_snapshot and baseline.scenario_snapshot != _scenario_snapshot(scenario):
        raise FaultError("Scenario definition differs from the recorded baseline")
    strategy = str(baseline.model.get("name", "cautious"))
    return _run_script(scenario, fault, strategy=strategy)


def _tool_schemas(scenario: Scenario) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters or {"type": "object", "properties": {}},
            },
        }
        for tool in scenario.tools
    ]


def _final_result(content: Any) -> Any:
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return FinalResult.model_validate(parsed).model_dump()
        except (ValueError, ValidationError):
            pass
    return content


def _public_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    """Keep only protocol fields; provider reasoning/analysis fields are excluded."""
    return {key: message[key] for key in ("role", "content", "tool_calls") if key in message}


def _tool_observation(
    world: dict[str, Any], name: str, arguments: dict[str, Any], known: set[str]
) -> tuple[Any, dict[str, Any] | None]:
    if name not in known:
        world.setdefault("calls", []).append(
            {"tool": name, "arguments": copy.deepcopy(arguments), "unsupported": True}
        )
        return {
            "unsupported": True,
            "tool": name,
            "error": "Tool is not declared by this scenario",
        }, None
    return _world_call(world, name, arguments)


def _argument_error(scenario: Scenario, name: str, arguments: dict[str, Any]) -> str | None:
    definition = next((tool for tool in scenario.tools if tool.name == name), None)
    if definition is None:
        return None
    schema = definition.parameters
    missing = [key for key in schema.get("required", []) if key not in arguments]
    if missing:
        return f"Missing required argument(s): {', '.join(missing)}"
    properties = schema.get("properties", {})
    if schema.get("additionalProperties") is False:
        extra = set(arguments) - set(properties)
        if extra:
            return f"Unexpected argument(s): {', '.join(sorted(extra))}"
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    for key, value in arguments.items():
        expected = properties.get(key, {}).get("type")
        py_type = type_map.get(expected)
        if py_type and (
            not isinstance(value, py_type)
            or expected in {"integer", "number"}
            and isinstance(value, bool)
        ):
            return f"Argument '{key}' must be {expected}"
        enum = properties.get(key, {}).get("enum")
        if enum is not None and value not in enum:
            return f"Argument '{key}' must be one of {enum}"
    return None


def record_ollama(
    scenario: Scenario,
    model: str = "qwen3:4b",
    base_url: str | None = None,
    max_turns: int = 12,
    max_tool_calls: int = 20,
    timeout: float = 90,
    run_timeout: float = 600,
) -> Trace:
    """Run the Ollama model against only the isolated scenario simulator."""
    from .ollama import OllamaError

    resolved_url = base_url or os.getenv("TOOLFAULT_OLLAMA_URL", "http://127.0.0.1:11434")
    run = str(uuid.uuid4())
    events: list[Event] = []
    world = copy.deepcopy(scenario.initial_world)
    ledger: list[dict[str, Any]] = []
    _event(run, events, "run_started", payload={"agent": "ollama", "model": model})
    _event(run, events, "user_message", payload=scenario.user_message)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": scenario.system},
        {"role": "user", "content": scenario.user_message},
    ]
    final: Any = ""
    calls = 0
    ollama_version: str | None = None
    started = time.monotonic()
    try:
        agent = OllamaAgent(model, resolved_url, timeout)
        ollama_version = agent.ensure_model()
        for _ in range(max_turns):
            remaining = run_timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise OllamaError(
                    f"Execution exceeded overall run timeout of {run_timeout} seconds"
                )
            agent.timeout = min(timeout, remaining)
            result = agent.chat(messages, _tool_schemas(scenario))
            message = _public_assistant_message(result["message"])
            messages.append(message)
            _event(
                run,
                events,
                "assistant_message",
                payload=message,
                metadata={
                    "model": model,
                    "total_duration_ns": result.get("total_duration"),
                    "ollama_version": ollama_version,
                },
            )
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                final = _final_result(message.get("content", ""))
                _event(run, events, "model_final_response", payload=final)
                break
            for call in tool_calls:
                calls += 1
                if calls > max_tool_calls:
                    raise OllamaError(f"Execution stopped at max_tool_calls={max_tool_calls}")
                fn = call.get("function", {})
                name = str(fn.get("name", ""))
                args = fn.get("arguments", {})
                if not isinstance(args, dict):
                    args = {}
                call_id = call.get("id") or f"call-{calls}"
                _event(
                    run,
                    events,
                    "model_tool_call",
                    tool_name=name,
                    tool_call_id=call_id,
                    arguments=args,
                )
                argument_error = _argument_error(scenario, name, args)
                if argument_error:
                    observation, effect = (
                        {"invalid_arguments": True, "tool": name, "error": argument_error},
                        None,
                    )
                else:
                    observation, effect = _tool_observation(
                        world, name, args, {item.name for item in scenario.tools}
                    )
                normalized = {
                    "tool_name": name,
                    "transport_status": observation.get("transport_status", 200)
                    if isinstance(observation, dict)
                    else 200,
                    "body": observation.get("body", observation)
                    if isinstance(observation, dict)
                    else observation,
                    "received_at": now().isoformat(),
                }
                response_event = Event(
                    run_id=run,
                    sequence=len(events),
                    type="tool_response",
                    tool_name=name,
                    tool_call_id=call_id,
                    arguments=args,
                    payload=normalized,
                )
                events.append(response_event)
                if effect:
                    effect["event_id"] = response_event.id
                    ledger.append(effect)
                    _event(
                        run,
                        events,
                        "simulated_effect",
                        tool_name=name,
                        tool_call_id=call_id,
                        payload=effect,
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": name,
                        "content": json.dumps(normalized, separators=(",", ":")),
                    }
                )
        else:
            raise OllamaError(f"Execution stopped at max_turns={max_turns}")
        if not final:
            raise OllamaError("Model completed without a final response")
        _event(run, events, "run_completed", payload={"status": "completed"})
    except OllamaError as exc:
        _event(run, events, "run_failed", payload={"error": str(exc)})
        evaluation = {"verdict": "inconclusive", "checks": [], "error": str(exc)}
        return Trace(
            run_id=run,
            scenario_id=scenario.id,
            kind="baseline",
            agent="ollama",
            model={
                "provider": "ollama",
                "name": model,
                "base_url": resolved_url,
                "temperature": 0,
                "seed": 42,
                "max_turns": max_turns,
                "max_tool_calls": max_tool_calls,
                "timeout_seconds": timeout,
                "run_timeout_seconds": run_timeout,
                "ollama_version": ollama_version,
            },
            scenario_snapshot=_scenario_snapshot(scenario),
            events=events,
            world_snapshot=world,
            effect_ledger=ledger,
            evaluation=evaluation,
        )
    evaluation = _evaluate(scenario, events, ledger, final)
    return Trace(
        run_id=run,
        scenario_id=scenario.id,
        kind="baseline",
        agent="ollama",
        model={
            "provider": "ollama",
            "name": model,
            "base_url": resolved_url,
            "temperature": 0,
            "seed": 42,
            "max_turns": max_turns,
            "max_tool_calls": max_tool_calls,
            "timeout_seconds": timeout,
            "run_timeout_seconds": run_timeout,
            "ollama_version": ollama_version,
        },
        scenario_snapshot=_scenario_snapshot(scenario),
        events=events,
        world_snapshot=world,
        effect_ledger=ledger,
        evaluation=evaluation,
    )


def replay_ollama(
    scenario: Scenario,
    baseline: Trace,
    fault: FaultSpec,
    model: str = "qwen3:4b",
    base_url: str | None = None,
    max_turns: int = 12,
    max_tool_calls: int = 20,
    timeout: float = 90,
    run_timeout: float = 600,
) -> Trace:
    """Resume real model inference immediately after one mutated baseline observation."""
    from .ollama import OllamaError

    if (
        baseline.scenario_id != scenario.id
        or baseline.kind != "baseline"
        or baseline.evaluation.verdict != "pass"
    ):
        raise FaultError("Replay requires a passing baseline for this scenario")
    if baseline.scenario_snapshot and baseline.scenario_snapshot != _scenario_snapshot(scenario):
        raise FaultError("Scenario definition differs from the recorded baseline")
    resolved_url = base_url or os.getenv("TOOLFAULT_OLLAMA_URL", "http://127.0.0.1:11434")
    run = str(uuid.uuid4())
    events: list[Event] = []
    started = time.monotonic()
    world = copy.deepcopy(scenario.initial_world)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": scenario.system},
        {"role": "user", "content": scenario.user_message},
    ]
    _event(
        run, events, "run_started", payload={"agent": "ollama", "model": model, "kind": "replay"}
    )
    _event(run, events, "user_message", payload=scenario.user_message)
    counts: dict[str, int] = {}
    injected = False
    final: Any = ""
    calls = 0
    ledger: list[dict[str, Any]] = []
    ollama_version: str | None = None
    # Use baseline events only to rebuild context up to the selected observation.
    response_id_map: dict[str, str] = {}
    target_event_id: str | None = None
    for event in baseline.events:
        if event.type == "assistant_message":
            messages.append(event.payload)
            _event(run, events, event.type, payload=event.payload, metadata=event.metadata)
        elif event.type == "model_tool_call":
            _event(
                run,
                events,
                event.type,
                tool_name=event.tool_name,
                tool_call_id=event.tool_call_id,
                arguments=event.arguments,
            )
        elif event.type == "tool_response":
            name = event.tool_name or ""
            counts[name] = counts.get(name, 0) + 1
            target = (
                name == fault.target.tool
                and counts[name] == fault.target.occurrence
                and (fault.target.arguments is None or event.arguments == fault.target.arguments)
            )
            observation = event.payload
            if target:
                original = copy.deepcopy(observation)
                observation, metadata = mutate(observation, fault)
                if not metadata["applied"]:
                    raise FaultError(f"Fault was not applicable: {metadata.get('reason')}")
                injected = True
                target_event_id = event.id
                _event(
                    run,
                    events,
                    "fault_injected",
                    tool_name=name,
                    tool_call_id=event.tool_call_id,
                    payload={
                        "fault": fault.model_dump(),
                        "original": original,
                        "modified": observation,
                        **metadata,
                    },
                )
            new_response = _event(
                run,
                events,
                "tool_response",
                tool_name=name,
                tool_call_id=event.tool_call_id,
                arguments=event.arguments,
                payload=observation,
                metadata={"injected": target},
            )
            response_id_map[event.id] = new_response.id
            messages.append(
                {
                    "role": "tool",
                    "tool_name": name,
                    "content": json.dumps(observation, separators=(",", ":")),
                }
            )
            if target:
                break
        elif event.type == "simulated_effect" and not injected and isinstance(event.payload, dict):
            effect = copy.deepcopy(event.payload)
            effect["event_id"] = response_id_map.get(effect.get("event_id"))
            ledger.append(effect)
            _event(run, events, "simulated_effect", tool_name=event.tool_name, payload=effect)
    if not injected:
        raise FaultError(
            f"No matching response reached target {fault.target.tool} occurrence {fault.target.occurrence}"
        )
    # Rebuild the checkpoint through the selected operation. The fault changes
    # only what the model observes; the already-executed business operation stays.
    target_seen = False
    target_effect: dict[str, Any] | None = None
    for event in baseline.events:
        if event.type == "model_tool_call":
            counts_for_world = event.tool_name == fault.target.tool
            # Selector occurrence is counted separately from the earlier context pass.
            seen = sum(
                1
                for prior in baseline.events[: event.sequence + 1]
                if prior.type == "model_tool_call" and prior.tool_name == fault.target.tool
            )
            is_selected = (
                counts_for_world
                and seen == fault.target.occurrence
                and (fault.target.arguments is None or event.arguments == fault.target.arguments)
            )
            _, effect = _world_call(
                world,
                event.tool_name or "",
                event.arguments or {},
                force_failure=is_selected and fault.execution_outcome == "business_failure",
            )
            if is_selected:
                target_seen = True
                if effect:
                    target_effect = effect
                break
    if target_effect is not None:
        target_effect["event_id"] = response_id_map.get(target_event_id or "")
        ledger.append(target_effect)
        _event(
            run,
            events,
            "simulated_effect",
            tool_name=target_effect.get("tool"),
            payload=target_effect,
        )
    if not target_seen:
        raise FaultError("Could not reconstruct the selected simulator checkpoint")
    try:
        agent = OllamaAgent(model, resolved_url, timeout)
        ollama_version = agent.ensure_model()
        for _ in range(max_turns):
            remaining = run_timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise OllamaError(
                    f"Execution exceeded overall run timeout of {run_timeout} seconds"
                )
            agent.timeout = min(timeout, remaining)
            result = agent.chat(messages, _tool_schemas(scenario))
            message = _public_assistant_message(result["message"])
            messages.append(message)
            _event(
                run,
                events,
                "assistant_message",
                payload=message,
                metadata={
                    "model": model,
                    "total_duration_ns": result.get("total_duration"),
                    "ollama_version": ollama_version,
                },
            )
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                final = _final_result(message.get("content", ""))
                _event(run, events, "model_final_response", payload=final)
                break
            for call in tool_calls:
                calls += 1
                if calls > max_tool_calls:
                    raise OllamaError(f"Execution stopped at max_tool_calls={max_tool_calls}")
                fn = call.get("function", {})
                name = str(fn.get("name", ""))
                args = fn.get("arguments", {})
                if not isinstance(args, dict):
                    args = {}
                call_id = call.get("id") or f"replay-call-{calls}"
                _event(
                    run,
                    events,
                    "model_tool_call",
                    tool_name=name,
                    tool_call_id=call_id,
                    arguments=args,
                )
                argument_error = _argument_error(scenario, name, args)
                if argument_error:
                    observation, effect = (
                        {"invalid_arguments": True, "tool": name, "error": argument_error},
                        None,
                    )
                else:
                    observation, effect = _tool_observation(
                        world, name, args, {item.name for item in scenario.tools}
                    )
                normalized = {
                    "tool_name": name,
                    "transport_status": observation.get("transport_status", 200)
                    if isinstance(observation, dict)
                    else 200,
                    "body": observation.get("body", observation)
                    if isinstance(observation, dict)
                    else observation,
                    "received_at": now().isoformat(),
                }
                response_event = Event(
                    run_id=run,
                    sequence=len(events),
                    type="tool_response",
                    tool_name=name,
                    tool_call_id=call_id,
                    arguments=args,
                    payload=normalized,
                )
                events.append(response_event)
                if effect:
                    effect["event_id"] = response_event.id
                    ledger.append(effect)
                    _event(
                        run,
                        events,
                        "simulated_effect",
                        tool_name=name,
                        tool_call_id=call_id,
                        payload=effect,
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": name,
                        "content": json.dumps(observation, separators=(",", ":")),
                    }
                )
        else:
            raise OllamaError(f"Execution stopped at max_turns={max_turns}")
        _event(run, events, "run_completed", payload={"status": "completed"})
        evaluation = _evaluate(scenario, events, ledger, final, replay_run=True)
    except OllamaError as exc:
        _event(run, events, "run_failed", payload={"error": str(exc)})
        evaluation = {"verdict": "inconclusive", "checks": [], "error": str(exc)}
    return Trace(
        run_id=run,
        scenario_id=scenario.id,
        kind="replay",
        agent="ollama",
        model={
            "provider": "ollama",
            "name": model,
            "base_url": resolved_url,
            "temperature": 0,
            "seed": 42,
            "max_turns": max_turns,
            "max_tool_calls": max_tool_calls,
            "timeout_seconds": timeout,
            "run_timeout_seconds": run_timeout,
            "ollama_version": ollama_version,
        },
        scenario_snapshot=_scenario_snapshot(scenario),
        events=events,
        world_snapshot=world,
        effect_ledger=ledger,
        evaluation=evaluation,
        mutation=fault.model_dump(),
    )


def run_demo(
    scenario: Scenario, strategy: str = "unsafe", fault_id: str = "business-error-label"
) -> tuple[Trace, Trace, dict[str, Any]]:
    baseline = record(scenario, strategy=strategy)
    fault = next((f for f in scenario.faults if f.id == fault_id), None)
    if fault is None:
        raise FaultError(f"Scenario has no fault with id '{fault_id}'")
    replay_trace = replay(scenario, baseline, fault)
    comparison = compare(baseline, replay_trace)
    return baseline, replay_trace, comparison


def compare(baseline: Trace, replay_trace: Trace) -> dict[str, Any]:
    left_events = [
        e for e in baseline.events if e.type in {"model_tool_call", "model_final_response"}
    ]
    right_events = [
        e for e in replay_trace.events if e.type in {"model_tool_call", "model_final_response"}
    ]

    def decisions(trace: Trace) -> list[tuple[Any, ...]]:
        return [
            (
                e.type,
                e.tool_name,
                e.arguments,
                e.payload if e.type == "model_final_response" else None,
            )
            for e in trace.events
            if e.type in {"model_tool_call", "model_final_response"}
        ]

    left = decisions(baseline)
    right = decisions(replay_trace)
    left_keys = [
        json.dumps(item, sort_keys=True, separators=(",", ":"), default=str) for item in left
    ]
    right_keys = [
        json.dumps(item, sort_keys=True, separators=(",", ":"), default=str) for item in right
    ]
    operations = difflib.SequenceMatcher(a=left_keys, b=right_keys, autojunk=False).get_opcodes()
    changed = next((item for item in operations if item[0] != "equal"), None)
    if changed is None:
        base_index = replay_index = None
        category = None
    else:
        tag, base_start, _, replay_start, _ = changed
        base_index, replay_index = base_start, replay_start
        if tag == "insert":
            category = (
                "added_tool_call"
                if right_events[replay_start].type == "model_tool_call"
                else "added_decision"
            )
        elif tag == "delete":
            category = (
                "removed_tool_call"
                if left_events[base_start].type == "model_tool_call"
                else "removed_decision"
            )
        elif (
            left[base_start][0] == "model_tool_call" and right[replay_start][0] == "model_tool_call"
        ):
            category = (
                "changed_arguments"
                if left[base_start][1] == right[replay_start][1]
                else "different_tool"
            )
        else:
            category = "changed_final_output"

    def outcomes(trace: Trace) -> list[str]:
        return [
            json.dumps(
                {key: value for key, value in effect.items() if key != "event_id"},
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            for effect in trace.effect_ledger
        ]

    outcome_changed = outcomes(baseline) != outcomes(replay_trace)
    return {
        "id": f"{baseline.run_id}:{replay_trace.run_id}",
        "baseline_run_id": baseline.run_id,
        "replay_run_id": replay_trace.run_id,
        "first_divergence_index": replay_index if replay_index is not None else base_index,
        "divergence_category": category,
        "behavior_changed": changed is not None,
        "outcome_changed": outcome_changed,
        "baseline_verdict": baseline.evaluation.verdict,
        "replay_verdict": replay_trace.evaluation.verdict,
        "baseline_first_decision_event": left_events[base_index].id
        if base_index is not None and base_index < len(left_events)
        else None,
        "replay_first_decision_event": right_events[replay_index].id
        if replay_index is not None and replay_index < len(right_events)
        else None,
    }
