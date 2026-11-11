"""Local read-only scenario/run API and offline demo launcher."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .engine import compare, load_scenario, record, record_ollama, replay, replay_ollama, run_demo
from .models import Trace
from .privacy import redact

SCENARIO_DIR = Path(__file__).parent / "examples"
RUN_DIR = Path(".agent-tool-response-fault-tester/runs")
REPORT_DIR = Path(".agent-tool-response-fault-tester/reports")
app = FastAPI(title="Agent Tool Response Fault Tester", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class DemoRequest(BaseModel):
    scenario_id: str
    fault_id: str | None = None


class RecordRequest(BaseModel):
    scenario_id: str
    agent: str = "scripted"
    model: str = "qwen3:4b"
    ollama_url: str | None = None
    max_turns: int = Field(default=12, ge=1, le=100)
    max_tool_calls: int = Field(default=20, ge=1, le=500)
    timeout: float = Field(default=90, gt=0, le=600)
    run_timeout: float = Field(default=600, gt=0, le=3600)


class ReplayRequest(BaseModel):
    scenario_id: str
    baseline_run_id: str
    fault_id: str
    model: str | None = None
    ollama_url: str | None = None
    max_turns: int = Field(default=12, ge=1, le=100)
    max_tool_calls: int = Field(default=20, ge=1, le=500)
    timeout: float = Field(default=90, gt=0, le=600)
    run_timeout: float = Field(default=600, gt=0, le=3600)


def _find_scenario(scenario_id: str) -> Path:
    for path in SCENARIO_DIR.glob("*/scenario.yaml"):
        if path.parent.name == scenario_id:
            return path
    raise HTTPException(404, "Scenario not found")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/scenarios")
def scenarios() -> list[dict[str, Any]]:
    result = []
    for path in SCENARIO_DIR.glob("*/scenario.yaml"):
        scenario = load_scenario(path)
        result.append(
            {
                "id": scenario.id,
                "name": scenario.name,
                "tools": [tool.model_dump(mode="json") for tool in scenario.tools],
                "faults": [
                    {
                        "id": fault.id,
                        "name": fault.name,
                        "target": fault.target.model_dump(mode="json"),
                        "mutation": fault.mutation.model_dump(mode="json"),
                    }
                    for fault in scenario.faults
                ],
            }
        )
    return result


@app.get("/api/scenarios/{scenario_id}")
def scenario(scenario_id: str) -> dict[str, Any]:
    return load_scenario(_find_scenario(scenario_id)).model_dump(mode="json")


@app.get("/api/runs")
def runs() -> list[dict[str, Any]]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    return [
        {
            "run_id": value["run_id"],
            "scenario_id": value["scenario_id"],
            "kind": value["kind"],
            "verdict": value["evaluation"]["verdict"],
        }
        for path in sorted(RUN_DIR.glob("*.json"))
        if (value := json.loads(path.read_text()))
    ]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    path = RUN_DIR / f"{Path(run_id).name}.json"
    if not path.exists():
        raise HTTPException(404, "Run not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/runs/{run_id}/events")
def events(run_id: str) -> list[dict[str, Any]]:
    return get_run(run_id)["events"]


@app.post("/api/demo")
def demo(request: DemoRequest) -> dict[str, Any]:
    source = _find_scenario(request.scenario_id)
    scenario_value = load_scenario(source)
    fault_id = request.fault_id or "business-error-label"
    if not any(item.id == fault_id for item in scenario_value.faults):
        raise HTTPException(404, "Fault not found")
    baseline, replay_trace, comparison = run_demo(scenario_value, fault_id=fault_id)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for trace in (baseline, replay_trace):
        (RUN_DIR / f"{trace.run_id}.json").write_text(
            json.dumps(redact(trace.model_dump(mode="json")), indent=2), encoding="utf-8"
        )
    report = {
        "schema_version": "1.0",
        "scenario": scenario_value.id,
        "fault": replay_trace.mutation,
        "baseline": baseline.model_dump(mode="json"),
        "replay": replay_trace.model_dump(mode="json"),
        "comparison": comparison,
    }
    safe_report = redact(report)
    (REPORT_DIR / f"{replay_trace.run_id}.json").write_text(
        json.dumps(safe_report, indent=2), encoding="utf-8"
    )
    return safe_report


@app.post("/api/record")
def start_record(request: RecordRequest) -> dict[str, Any]:
    scenario_value = load_scenario(_find_scenario(request.scenario_id))
    if request.agent not in {"scripted", "scripted-cautious", "scripted-unsafe", "ollama"}:
        raise HTTPException(422, "Unsupported agent")
    strategy = "unsafe" if request.agent == "scripted-unsafe" else "cautious"
    trace = (
        record(scenario_value, strategy=strategy)
        if request.agent.startswith("scripted")
        else record_ollama(
            scenario_value,
            model=request.model,
            base_url=request.ollama_url,
            max_turns=request.max_turns,
            max_tool_calls=request.max_tool_calls,
            timeout=request.timeout,
            run_timeout=request.run_timeout,
        )
    )
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / f"{trace.run_id}.json").write_text(
        json.dumps(redact(trace.model_dump(mode="json")), indent=2), encoding="utf-8"
    )
    return redact(trace.model_dump(mode="json"))


@app.post("/api/replay")
def start_replay(request: ReplayRequest) -> dict[str, Any]:
    scenario_value = load_scenario(_find_scenario(request.scenario_id))
    baseline_value = get_run(request.baseline_run_id)
    baseline = Trace.model_validate(baseline_value)
    if baseline.agent == "imported":
        raise HTTPException(422, "Imported traces cannot be continued in this MVP")
    replay_model = request.model or str(baseline.model.get("name", "qwen3:4b"))
    fault = next((item for item in scenario_value.faults if item.id == request.fault_id), None)
    if fault is None:
        raise HTTPException(404, "Fault not found")
    trace = (
        replay(scenario_value, baseline, fault)
        if baseline.agent == "scripted"
        else replay_ollama(
            scenario_value,
            baseline,
            fault,
            model=replay_model,
            base_url=request.ollama_url,
            max_turns=request.max_turns,
            max_tool_calls=request.max_tool_calls,
            timeout=request.timeout,
            run_timeout=request.run_timeout,
        )
    )
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / f"{trace.run_id}.json").write_text(
        json.dumps(redact(trace.model_dump(mode="json")), indent=2), encoding="utf-8"
    )
    comparison = compare(baseline, trace)
    report = {
        "schema_version": "1.0",
        "scenario": scenario_value.id,
        "fault": trace.mutation,
        "baseline": baseline.model_dump(mode="json"),
        "replay": trace.model_dump(mode="json"),
        "comparison": comparison,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    safe_report = redact(report)
    (REPORT_DIR / f"{trace.run_id}.json").write_text(
        json.dumps(safe_report, indent=2), encoding="utf-8"
    )
    return safe_report


@app.get("/api/comparisons/{comparison_id}")
def get_comparison(comparison_id: str) -> dict[str, Any]:
    path = REPORT_DIR / f"{Path(comparison_id).name}.json"
    if not path.exists():
        raise HTTPException(404, "Comparison not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/reports/{report_id}")
def get_report(report_id: str) -> dict[str, Any]:
    path = REPORT_DIR / f"{Path(report_id).name}.json"
    if not path.exists():
        raise HTTPException(404, "Report not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/models")
def models() -> dict[str, Any]:
    from .ollama import OllamaAgent, OllamaError

    try:
        return {"provider": "ollama", "available": True, "models": OllamaAgent().models()}
    except OllamaError as exc:
        return {"provider": "ollama", "available": False, "models": [], "note": str(exc)}
