"""Validated portable models for scenarios, faults, and traces."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def now() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FaultTarget(StrictModel):
    tool: str
    occurrence: int = Field(default=1, ge=1)
    arguments: dict[str, Any] | None = None


class Mutation(StrictModel):
    type: Literal["remove_field", "stale_data", "truncate", "conflict", "reorder", "business_error"]
    path: str | None = None
    value: Any = None
    max_age_seconds: int | None = Field(default=None, ge=0)
    reference_time: datetime | None = None
    count: int | None = Field(default=None, ge=0)
    strategy: Literal["reverse", "shuffle"] = "reverse"
    seed: int = 42
    status_path: str | None = None
    error_value: Any = None

    @model_validator(mode="after")
    def validate_parameters(self) -> Mutation:
        needs_path = {"remove_field", "stale_data", "truncate", "conflict", "reorder"}
        if self.type in needs_path and self.path is None:
            raise ValueError(f"{self.type} requires a JSON Pointer path")
        for pointer in (self.path, self.status_path):
            if pointer is not None and (
                not pointer.startswith("/") or re.search(r"~(?![01])", pointer)
            ):
                raise ValueError(f"Invalid JSON Pointer: {pointer}")
        if self.type == "stale_data" and (
            self.max_age_seconds is None or self.reference_time is None
        ):
            raise ValueError("stale_data requires max_age_seconds and authoritative reference_time")
        return self


class FaultSpec(StrictModel):
    id: str
    name: str
    target: FaultTarget
    mutation: Mutation
    description: str = ""
    execution_outcome: Literal["response_only", "business_failure"] = "response_only"

    @model_validator(mode="after")
    def validate_execution_outcome(self) -> FaultSpec:
        if self.execution_outcome == "business_failure" and self.mutation.type != "business_error":
            raise ValueError("business_failure is only valid for business_error mutations")
        return self


class Check(StrictModel):
    id: str
    description: str
    kind: Literal[
        "effect_exists",
        "final_contains",
        "final_not_contains",
        "field_equals",
        "forbid_completion_without_effect",
        "require_status",
    ]
    tool: str | None = None
    text: str | None = None
    path: str | None = None
    value: Any = None
    status: Literal["completed", "not_completed", "needs_clarification", "blocked"] | None = None

    @model_validator(mode="after")
    def validate_parameters(self) -> Check:
        if self.kind in {"final_contains", "final_not_contains"} and self.text is None:
            raise ValueError(f"{self.kind} requires text")
        if self.kind == "effect_exists" and self.tool is None:
            raise ValueError("effect_exists requires a tool")
        if self.kind in {"field_equals"} and self.path is None:
            raise ValueError(f"{self.kind} requires a JSON Pointer path")
        if self.kind == "require_status" and self.status is None:
            raise ValueError("require_status requires status")
        if self.path is not None and (
            not self.path.startswith("/") or re.search(r"~(?![01])", self.path)
        ):
            raise ValueError(f"Invalid JSON Pointer: {self.path}")
        return self


class ToolDefinition(StrictModel):
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class Invariant(StrictModel):
    id: str
    tool: str
    description: str
    when_path: str
    when_value: Any
    then_path: str
    then_value: Any


class FinalResult(StrictModel):
    status: Literal["completed", "not_completed", "needs_clarification", "blocked"]
    summary: str = Field(min_length=1)
    order_id: str = Field(min_length=1)


class EvaluationCheck(StrictModel):
    id: str
    description: str
    passed: bool | None
    evidence_event_ids: list[str] = Field(default_factory=list)


class Evaluation(StrictModel):
    verdict: Literal["pass", "fail", "inconclusive", "invalid"]
    checks: list[EvaluationCheck] = Field(default_factory=list)
    behavior_tags: list[
        Literal[
            "retried",
            "requested_clarification",
            "recovered",
            "wrong_action",
            "false_success",
            "ignored_conflict",
            "used_stale_data",
            "incomplete_response",
        ]
    ] = Field(default_factory=list)
    final_response: Any = None
    error: str | None = None


class Scenario(StrictModel):
    id: str
    name: str
    system: str
    user_message: str
    tools: list[ToolDefinition]
    initial_world: dict[str, Any]
    baseline_script: list[dict[str, Any]]
    checks: list[Check]
    replay_checks: list[Check] = Field(default_factory=list)
    faults: list[FaultSpec] = Field(default_factory=list)
    invariants: list[Invariant] = Field(default_factory=list)
    freshness: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_references(self) -> Scenario:
        tool_names = [tool.name for tool in self.tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("Tool names must be unique")
        unknown_calls = sorted(
            {step.get("tool", "") for step in self.baseline_script} - set(tool_names)
        )
        if unknown_calls:
            raise ValueError(f"Baseline references undeclared tools: {', '.join(unknown_calls)}")
        unknown_faults = sorted({fault.target.tool for fault in self.faults} - set(tool_names))
        if unknown_faults:
            raise ValueError(f"Fault targets undeclared tools: {', '.join(unknown_faults)}")
        world_requirements = {
            "get_order": {"order"},
            "verify_shipping_address": {"address_verified", "verified_at"},
            "check_inventory": {"inventory"},
            "create_shipping_label": {"address_verified", "labels"},
        }
        required_state = set().union(*(world_requirements.get(name, set()) for name in tool_names))
        missing_state = sorted(required_state - set(self.initial_world))
        if missing_state:
            raise ValueError(
                f"Initial world is missing simulator state: {', '.join(missing_state)}"
            )
        all_checks = [*self.checks, *self.replay_checks]
        for label, values in (
            ("fault", [fault.id for fault in self.faults]),
            ("check", [check.id for check in all_checks]),
            ("invariant", [invariant.id for invariant in self.invariants]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label.title()} IDs must be unique")
        return self


class Event(StrictModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    sequence: int = Field(ge=0)
    type: str
    timestamp: datetime = Field(default_factory=now)
    tool_name: str | None = None
    tool_call_id: str | None = None
    arguments: dict[str, Any] | None = None
    payload: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Trace(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    scenario_id: str
    kind: Literal["baseline", "replay"]
    agent: Literal["scripted", "ollama", "imported"]
    model: dict[str, Any] = Field(default_factory=dict)
    scenario_snapshot: dict[str, Any] = Field(default_factory=dict)
    events: list[Event]
    world_snapshot: dict[str, Any]
    effect_ledger: list[dict[str, Any]]
    evaluation: Evaluation
    mutation: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_events(self) -> Trace:
        if [event.sequence for event in self.events] != list(range(len(self.events))):
            raise ValueError(
                "Trace event sequence numbers must be contiguous and ordered from zero"
            )
        if any(event.run_id != self.run_id for event in self.events):
            raise ValueError("Every event run_id must match the trace run_id")
        event_ids = [event.id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("Trace event IDs must be unique")
        return self

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)
