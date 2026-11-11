"""Command-line workflows for local Agent Tool Response Fault Tester experiments."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import uvicorn
from rich import print

from .engine import compare, load_scenario, record, record_ollama, replay, replay_ollama, run_demo
from .models import FaultSpec, Trace
from .privacy import redact

app = typer.Typer(no_args_is_help=True, help="Discover which tool responses make AI agents fail.")
DEFAULT_SCENARIO = Path(__file__).parent / "examples/order-fulfillment/scenario.yaml"


def _write(trace: Trace, path: Path) -> None:
    if path.exists():
        raise typer.BadParameter(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(redact(trace.model_dump(mode="json")), indent=2), encoding="utf-8")


def _artifact_path(out: Path, run_id: str) -> Path:
    return out if out.suffix else out / f"{run_id}.json"


@app.command()
def doctor():
    """Check local runtime dependencies and Ollama availability."""
    from .ollama import OllamaAgent, OllamaError

    try:
        version = OllamaAgent().ensure_model()
        print(f"Ollama: available · qwen3:4b installed · version {version or 'unknown'}")
    except OllamaError as exc:
        print(f"Ollama: {exc}")
    scenario = load_scenario(DEFAULT_SCENARIO)
    print(f"Scenario: valid · {scenario.name}")
    storage = Path(".agent-tool-response-fault-tester")
    storage.mkdir(parents=True, exist_ok=True)
    print(f"Storage: writable · {storage.resolve()}")
    print("Scripted workflows do not require Ollama.")


@app.command()
def validate(scenario: Path = typer.Argument(DEFAULT_SCENARIO, exists=True)):
    """Validate a scenario YAML file."""
    value = load_scenario(scenario)
    print(
        f"[green]Valid[/green]: {value.name} ({len(value.tools)} tools, {len(value.faults)} faults)"
    )


@app.command()
def demo(
    agent: str = typer.Option(
        "scripted", help="Choose scripted-unsafe (default demo) or scripted-cautious."
    ),
    out: Path = typer.Option(Path(".agent-tool-response-fault-tester/runs")),
):
    """Run a baseline and the bundled controlled-fault example."""
    if agent not in {"scripted", "scripted-cautious", "scripted-unsafe"}:
        raise typer.BadParameter("demo supports --agent scripted-cautious or scripted-unsafe")
    if out.suffix:
        raise typer.BadParameter("demo --out must be a directory")
    scenario = load_scenario(DEFAULT_SCENARIO)
    strategy = "cautious" if agent == "scripted-cautious" else "unsafe"
    baseline, replay_trace, comparison = run_demo(scenario, strategy=strategy)
    _write(baseline, _artifact_path(out, baseline.run_id))
    _write(replay_trace, _artifact_path(out, replay_trace.run_id))
    report = {
        "schema_version": "1.0",
        "scenario": scenario.id,
        "fault": replay_trace.mutation,
        "comparison": comparison,
        "baseline": baseline.model_dump(mode="json"),
        "replay": replay_trace.model_dump(mode="json"),
    }
    report_path = Path(".agent-tool-response-fault-tester/reports") / f"{replay_trace.run_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(redact(report), indent=2), encoding="utf-8")
    print(
        f"Scenario: {scenario.name}\nBaseline: {baseline.evaluation.verdict}\nReplay: {replay_trace.evaluation.verdict}\nReport: {report_path}"
    )


@app.command("record")
def record_cmd(
    scenario: Path = typer.Argument(exists=True),
    agent: str = typer.Option("scripted"),
    model: str = typer.Option("qwen3:4b"),
    ollama_url: str | None = typer.Option(None, "--ollama-url"),
    max_turns: int = typer.Option(12),
    max_tool_calls: int = typer.Option(20),
    timeout: float = typer.Option(90),
    run_timeout: float = typer.Option(600),
    out: Path = typer.Option(Path(".agent-tool-response-fault-tester/runs"), "--out"),
):
    """Record a baseline. Scripted mode is offline and deterministic."""
    if agent not in {"scripted", "scripted-cautious", "scripted-unsafe", "ollama"}:
        raise typer.BadParameter("--agent must be scripted-cautious, scripted-unsafe, or ollama.")
    scenario_value = load_scenario(scenario)
    strategy = "unsafe" if agent == "scripted-unsafe" else "cautious"
    trace = (
        record(scenario_value, strategy=strategy)
        if agent.startswith("scripted")
        else record_ollama(
            scenario_value,
            model=model,
            base_url=ollama_url,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            timeout=timeout,
            run_timeout=run_timeout,
        )
    )
    path = _artifact_path(out, trace.run_id)
    _write(trace, path)
    print(f"{trace.evaluation.verdict.upper()}: {path}")
    if trace.evaluation.verdict == "fail":
        raise typer.Exit(code=1)
    if trace.evaluation.verdict == "inconclusive":
        raise typer.Exit(code=2)


@app.command("replay")
def replay_cmd(
    baseline: Path = typer.Argument(exists=True),
    fault: Path = typer.Option(..., "--fault", exists=True),
    scenario: Path = typer.Option(DEFAULT_SCENARIO, "--scenario", exists=True),
    model: str | None = typer.Option(None, "--model"),
    ollama_url: str | None = typer.Option(None, "--ollama-url"),
    max_turns: int = typer.Option(12),
    max_tool_calls: int = typer.Option(20),
    timeout: float = typer.Option(90),
    run_timeout: float = typer.Option(600),
    out: Path = typer.Option(Path(".agent-tool-response-fault-tester/runs"), "--out"),
):
    """Replay a baseline with a YAML fault definition."""
    import yaml

    spec = FaultSpec.model_validate(yaml.safe_load(fault.read_text(encoding="utf-8")))
    baseline_trace = Trace.model_validate_json(baseline.read_text(encoding="utf-8"))
    if baseline_trace.agent == "imported":
        raise typer.BadParameter("Imported traces cannot be continued in this MVP.")
    scenario_value = load_scenario(scenario)
    replay_model = model or str(baseline_trace.model.get("name", "qwen3:4b"))
    trace = (
        replay(scenario_value, baseline_trace, spec)
        if baseline_trace.agent == "scripted"
        else replay_ollama(
            scenario_value,
            baseline_trace,
            spec,
            model=replay_model,
            base_url=ollama_url,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            timeout=timeout,
            run_timeout=run_timeout,
        )
    )
    path = _artifact_path(out, trace.run_id)
    _write(trace, path)
    print(f"{trace.evaluation.verdict.upper()}: {path}")
    if trace.evaluation.verdict == "inconclusive":
        raise typer.Exit(code=2)


@app.command()
def report(
    trace: Path = typer.Argument(exists=True),
    output: Path | None = typer.Option(None, "--out"),
    format: str = typer.Option("html", "--format"),
):
    """Render an offline, escaped HTML report from a trace or demo JSON report."""
    import html

    raw = json.loads(trace.read_text(encoding="utf-8"))
    target = output or Path("report.html" if format == "html" else "report.json")
    if target.exists():
        raise typer.BadParameter(f"Refusing to overwrite existing artifact: {target}")
    if format == "json":
        target.write_text(json.dumps(redact(raw), indent=2), encoding="utf-8")
        print(f"Wrote {target}")
        return
    if format != "html":
        raise typer.BadParameter("--format must be html or json")
    report_value = raw if isinstance(raw, dict) and "baseline" in raw and "replay" in raw else None
    traces = (
        [("Baseline", report_value["baseline"]), ("Faulted replay", report_value["replay"])]
        if report_value
        else [("Trace", Trace.model_validate(raw).model_dump(mode="json"))]
    )
    sections = []
    for label, value in traces:
        rows = "".join(
            f"<li><b>{html.escape(str(e['type']))}</b> {html.escape(str(e.get('tool_name') or ''))}<pre>{html.escape(json.dumps(e.get('payload'), ensure_ascii=False, indent=2))}</pre></li>"
            for e in value["events"]
        )
        checks = "".join(
            f"<li>{html.escape(str(check['description']))}: <b>{html.escape(str(check['passed']))}</b><small> Evidence: {html.escape(', '.join(check.get('evidence_event_ids', [])))}</small></li>"
            for check in value["evaluation"].get("checks", [])
        )
        sections.append(
            f"<section><h2>{html.escape(label)} · {html.escape(value['evaluation']['verdict'].upper())}</h2><h3>Evaluation evidence</h3><ul>{checks}</ul><h3>Timeline</h3><ol>{rows}</ol></section>"
        )
    comparison = report_value.get("comparison") if report_value else None
    summary = (
        f"<p>Scenario: {html.escape(str(report_value['scenario']))}</p>"
        if report_value
        else f"<p>Scenario: {html.escape(traces[0][1]['scenario_id'])}</p>"
    )
    if comparison:
        summary += (
            f"<p>Agent decisions changed: {html.escape(str(comparison['behavior_changed']))} · "
            f"Simulator outcome changed: {html.escape(str(comparison.get('outcome_changed', False)))} · "
            f"First decision divergence: {html.escape(str(comparison['first_divergence_index']))}</p>"
        )
    document = f"<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Agent Tool Response Fault Tester report</title><style>body{{font:16px system-ui;max-width:1100px;margin:3rem auto;padding:0 1rem;color:#17212b}}main{{display:grid;grid-template-columns:1fr 1fr;gap:2rem}}pre{{background:#f2f5f7;padding:1rem;overflow:auto;max-height:20rem}}li{{margin:1rem 0}}small{{display:block;color:#60706a}}@media(max-width:700px){{main{{grid-template-columns:1fr}}}}</style><h1>Agent Tool Response Fault Tester report</h1>{summary}<main>{''.join(sections)}</main>"
    target.write_text(document, encoding="utf-8")
    print(f"Wrote {target}")


@app.command()
def serve(host: str = typer.Option("127.0.0.1"), port: int = typer.Option(8000)):
    """Start the local dashboard API."""
    if host not in {"127.0.0.1", "localhost"}:
        raise typer.BadParameter("Local service must bind to loopback.")
    print(
        f"Agent Tool Response Fault Tester API: http://{host}:{port}/api/health\nDashboard: run `cd web && pnpm install --frozen-lockfile && pnpm dev`"
    )
    uvicorn.run("agent_tool_response_fault_tester.api:app", host=host, port=port)


@app.command("compare")
def compare_cmd(
    baseline: Path = typer.Argument(exists=True), replay_trace: Path = typer.Argument(exists=True)
):
    """Compare two portable traces."""
    left = Trace.model_validate_json(baseline.read_text(encoding="utf-8"))
    right = Trace.model_validate_json(replay_trace.read_text(encoding="utf-8"))
    print_json = json.dumps(compare(left, right), indent=2)
    print(print_json)


if __name__ == "__main__":
    app()
