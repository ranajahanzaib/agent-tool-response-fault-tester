import {
  Activity,
  ArrowDown,
  ArrowUpRight,
  Check,
  ChevronDown,
  CircleAlert,
  Download,
  FlaskConical,
  Play,
  ShieldCheck,
  Workflow,
} from "lucide-react";
import { useEffect, useState } from "react";

type CheckResult = {
  id: string;
  description: string;
  passed: boolean | null;
  evidence_event_ids: string[];
};
type Event = {
  id: string;
  sequence: number;
  timestamp: string;
  type: string;
  tool_name?: string;
  payload?: unknown;
  arguments?: unknown;
  metadata?: Record<string, unknown>;
};
type Trace = {
  run_id: string;
  agent: string;
  model: Record<string, unknown>;
  evaluation: {
    verdict: string;
    checks: CheckResult[];
    behavior_tags?: string[];
    error?: string;
  };
  events: Event[];
};
type Scenario = {
  id: string;
  name: string;
  faults: {
    id: string;
    name: string;
    target: { tool: string };
    mutation: { type: string };
  }[];
};
type SavedRun = {
  run_id: string;
  scenario_id: string;
  kind: string;
  verdict: string;
};
type Comparison = {
  behavior_changed: boolean;
  outcome_changed?: boolean;
  first_divergence_index: number | null;
  divergence_category?: string | null;
  replay_first_decision_event?: string | null;
};
type Result = {
  scenario: string;
  baseline: Trace;
  replay: Trace;
  comparison: Comparison;
};
type AgentMode = "scripted-cautious" | "scripted-unsafe" | "ollama";

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    signal: AbortSignal.timeout(630_000),
    ...init,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

function Timeline({
  title,
  trace,
  injection,
  divergenceId,
}: {
  title: string;
  trace: Trace;
  injection?: boolean;
  divergenceId?: string | null;
}) {
  return (
    <div className="timeline">
      <div className="panel-heading">
        <div>
          <small>{title}</small>
          <b>
            {trace.agent === "ollama"
              ? String(trace.model.name)
              : `SCRIPTED · ${String(trace.model.name ?? "cautious").toUpperCase()}`}
          </b>
        </div>
        <span>{trace.events.length} EVENTS</span>
      </div>
      {trace.events.map((event) => (
        <div
          className={`event ${event.type === "fault_injected" ? "injected" : ""} ${event.id === divergenceId ? "diverged" : ""}`}
          id={event.id}
          key={event.id}
        >
          <div className="event-line">
            <i />
          </div>
          <div className="event-content">
            <div className="event-title">
              <b>{event.type.replaceAll("_", " ")}</b>
              <span>#{String(event.sequence + 1).padStart(2, "0")}</span>
            </div>
            {event.tool_name && (
              <div className="tool-name">
                {event.tool_name}
                {event.arguments !== undefined
                  ? ` · ${JSON.stringify(event.arguments)}`
                  : ""}
              </div>
            )}
            {event.type === "fault_injected" && (
              <div className="fault-note">
                <CircleAlert size={14} /> INJECTION APPLIED
              </div>
            )}
            {event.id === divergenceId && (
              <div className="divergence-note">
                FIRST OBSERVED DECISION DIVERGENCE
              </div>
            )}
            {event.payload !== undefined && (
              <details>
                <summary>
                  Inspect payload <ArrowUpRight size={12} />
                </summary>
                <pre>{JSON.stringify(event.payload, null, 2)}</pre>
              </details>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [scenarioId, setScenarioId] = useState("");
  const [faultId, setFaultId] = useState("");
  const [runs, setRuns] = useState<SavedRun[]>([]);
  const [baselineId, setBaselineId] = useState("");
  const [agent, setAgent] = useState<AgentMode>("scripted-unsafe");
  const [model, setModel] = useState("qwen3:4b");
  const [ollamaUrl, setOllamaUrl] = useState("http://127.0.0.1:11434");
  const [maxTurns, setMaxTurns] = useState(12);
  const [maxToolCalls, setMaxToolCalls] = useState(20);
  const [timeout, setTimeoutSeconds] = useState(90);
  const [runTimeout, setRunTimeoutSeconds] = useState(600);
  const [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const scenario = scenarios.find((item) => item.id === scenarioId) ?? null;
  const fault =
    scenario?.faults.find((item) => item.id === faultId) ??
    scenario?.faults[0] ??
    null;

  async function refreshRuns() {
    setRuns(await api<SavedRun[]>("/api/runs"));
  }
  useEffect(() => {
    Promise.all([
      api<Scenario[]>("/api/scenarios"),
      api<SavedRun[]>("/api/runs"),
    ])
      .then(([available, existing]) => {
        setScenarios(available);
        setScenarioId(available[0]?.id ?? "");
        setFaultId(available[0]?.faults[0]?.id ?? "");
        setRuns(existing);
      })
      .catch(() =>
        setError(
          "Start the local API with `uv run agent-tool-response-fault-tester serve`.",
        ),
      )
      .finally(() => setLoading(false));
  }, []);
  useEffect(() => {
    if (scenario) setFaultId(scenario.faults[0]?.id ?? "");
  }, [scenarioId]);

  async function runExperiment() {
    if (!scenario || !fault) return;
    setBusy(true);
    setError("");
    try {
      let selectedBaseline = baselineId;
      if (!selectedBaseline) {
        const baseline = await api<Trace>("/api/record", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            scenario_id: scenario.id,
            agent,
            model,
            ollama_url: ollamaUrl,
            max_turns: maxTurns,
            max_tool_calls: maxToolCalls,
            timeout,
            run_timeout: runTimeout,
          }),
        });
        if (baseline.evaluation.verdict !== "pass")
          throw new Error(
            baseline.evaluation.error ??
              `Baseline is ${baseline.evaluation.verdict}; this scenario is not ready for counterfactual replay.`,
          );
        selectedBaseline = baseline.run_id;
      }
      const report = await api<Result>("/api/replay", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scenario_id: scenario.id,
          baseline_run_id: selectedBaseline,
          fault_id: fault.id,
          model,
          ollama_url: ollamaUrl,
          max_turns: maxTurns,
          max_tool_calls: maxToolCalls,
          timeout,
          run_timeout: runTimeout,
        }),
      });
      setResult(report);
      await refreshRuns();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Experiment failed");
    } finally {
      setBusy(false);
    }
  }
  function download() {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `agent-tool-response-fault-tester-${result.replay.run_id}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }
  function openEvidence(eventId: string) {
    document
      .getElementById(eventId)
      ?.scrollIntoView({ behavior: "smooth", block: "center" });
  }
  const injection = result?.replay.events.find(
    (event) => event.type === "fault_injected",
  );
  const original =
    injection &&
    typeof injection.payload === "object" &&
    injection.payload !== null
      ? (injection.payload as Record<string, unknown>).original
      : undefined;
  const modified =
    injection &&
    typeof injection.payload === "object" &&
    injection.payload !== null
      ? (injection.payload as Record<string, unknown>).modified
      : undefined;
  const behaviorChanged = result?.comparison.behavior_changed;

  return (
    <main className="shell">
      <aside className="rail">
        <div className="mark">
          <Workflow size={19} />
        </div>
        <div className="rail-rule" />
        <button aria-label="Experiments" className="rail-icon active">
          <FlaskConical size={18} />
        </button>
        <button aria-label="Activity" className="rail-icon">
          <Activity size={18} />
        </button>
        <div className="rail-bottom">TF</div>
      </aside>
      <section className="content">
        <header className="top">
          <div className="crumb">
            WORKSPACE <span>/</span> EXPERIMENTS
          </div>
          <div className="local">
            <i /> LOCAL ENVIRONMENT <ChevronDown size={13} />
          </div>
        </header>
        <div className="intro">
          <div>
            <div className="eyebrow">
              <span className="pulse" /> TOOL RESPONSE RELIABILITY
            </div>
            <h1>
              Find the failure
              <br />
              <em>behind the answer.</em>
            </h1>
            <p>Change one observation. See what the agent does next.</p>
          </div>
          <div className="intro-mark">
            <Workflow size={50} strokeWidth={1} />
            <span>{String(scenarios.length).padStart(2, "0")} SCENARIOS</span>
          </div>
        </div>
        <div className="controls">
          <label className="field">
            <small>SCENARIO</small>
            <select
              value={scenarioId}
              onChange={(e) => {
                setScenarioId(e.target.value);
                setBaselineId("");
                setResult(null);
              }}
            >
              {scenarios.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <small>BASELINE</small>
            <select
              value={baselineId}
              onChange={(e) => {
                setBaselineId(e.target.value);
                setResult(null);
              }}
            >
              <option value="">Record a new baseline</option>
              {runs
                .filter(
                  (item) =>
                    item.scenario_id === scenarioId &&
                    item.kind === "baseline" &&
                    item.verdict === "pass",
                )
                .map((item) => (
                  <option key={item.run_id} value={item.run_id}>
                    {item.run_id.slice(0, 8)} · {item.verdict}
                  </option>
                ))}
            </select>
          </label>
          <label className="field fault-field">
            <small>FAULT INJECTION</small>
            <select
              value={fault?.id ?? ""}
              onChange={(e) => {
                setFaultId(e.target.value);
                setResult(null);
              }}
            >
              {scenario?.faults.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name}
                </option>
              ))}
            </select>
          </label>
          <label className="field agent-field">
            <small>AGENT</small>
            <select
              value={agent}
              onChange={(e) => setAgent(e.target.value as AgentMode)}
            >
              <option value="scripted-unsafe">Scripted · unsafe</option>
              <option value="scripted-cautious">Scripted · cautious</option>
              <option value="ollama">Ollama · local model</option>
            </select>
          </label>
          {agent === "ollama" && (
            <>
              <label className="field model-field">
                <small>MODEL</small>
                <input
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  aria-label="Ollama model name"
                />
              </label>
              <label className="field model-field">
                <small>OLLAMA URL</small>
                <input
                  value={ollamaUrl}
                  onChange={(e) => setOllamaUrl(e.target.value)}
                  aria-label="Ollama URL"
                />
              </label>
            </>
          )}
          <button
            className="run"
            onClick={runExperiment}
            disabled={!scenario || !fault || busy || loading}
          >
            <Play size={15} fill="currentColor" />
            {busy ? "RUNNING…" : "RUN REPLAY"}
          </button>
        </div>
        <details className="limits">
          <summary>Execution limits</summary>
          <div>
            <label>
              Model timeout (s)
              <input
                type="number"
                min="1"
                max="600"
                value={timeout}
                onChange={(e) => setTimeoutSeconds(Number(e.target.value))}
              />
            </label>
            <label>
              Run timeout (s)
              <input
                type="number"
                min="1"
                max="3600"
                value={runTimeout}
                onChange={(e) => setRunTimeoutSeconds(Number(e.target.value))}
              />
            </label>
            <label>
              Model turns
              <input
                type="number"
                min="1"
                max="100"
                value={maxTurns}
                onChange={(e) => setMaxTurns(Number(e.target.value))}
              />
            </label>
            <label>
              Tool calls
              <input
                type="number"
                min="1"
                max="500"
                value={maxToolCalls}
                onChange={(e) => setMaxToolCalls(Number(e.target.value))}
              />
            </label>
          </div>
        </details>
        {error && (
          <div className="error" role="alert">
            <CircleAlert size={16} />
            {error}
          </div>
        )}
        <div className="section-head">
          <div>
            <span className="section-num">01</span>
            <h2>Execution comparison</h2>
            <span className="muted">
              Recorded evidence from isolated simulator runs
            </span>
          </div>
          {result && (
            <button className="download" onClick={download}>
              <Download size={14} /> Export JSON
            </button>
          )}
        </div>
        {result ? (
          <>
            <div className="stats">
              <div>
                <small>BASELINE · {result.baseline.agent.toUpperCase()}</small>
                <strong
                  className={
                    result.baseline.evaluation.verdict === "pass"
                      ? "pass"
                      : "fail"
                  }
                >
                  {result.baseline.evaluation.verdict === "pass" ? (
                    <Check size={16} />
                  ) : (
                    <CircleAlert size={16} />
                  )}{" "}
                  {result.baseline.evaluation.verdict.toUpperCase()}
                </strong>
                <span>
                  {
                    result.baseline.events.filter(
                      (event) => event.type === "tool_response",
                    ).length
                  }{" "}
                  observations
                </span>
              </div>
              <div className="stat-arrow">
                <ArrowDown size={16} />
              </div>
              <div>
                <small>FAULTED RUN · {result.replay.agent.toUpperCase()}</small>
                <strong
                  className={
                    result.replay.evaluation.verdict === "pass"
                      ? "pass"
                      : "fail"
                  }
                >
                  {result.replay.evaluation.verdict === "pass" ? (
                    <Check size={16} />
                  ) : (
                    <CircleAlert size={16} />
                  )}{" "}
                  {result.replay.evaluation.verdict.toUpperCase()}
                </strong>
                <span>{fault?.name ?? "Selected fault"}</span>
              </div>
              <div>
                <small>
                  FIRST DIVERGENCE ·{" "}
                  {result.comparison.divergence_category ?? "NONE"}
                </small>
                <strong>
                  {result.comparison.first_divergence_index === null
                    ? "None observed"
                    : `Decision ${result.comparison.first_divergence_index + 1}`}
                </strong>
                <span>
                  {behaviorChanged
                    ? "Continuation differs"
                    : result.comparison.outcome_changed
                      ? "Simulator outcome differs"
                      : "Same decisions observed"}
                </span>
              </div>
            </div>
            <div className="workbench">
              <div className="timeline-pair">
                <Timeline title="BASELINE EXECUTION" trace={result.baseline} />
                <Timeline
                  title="FAULTED EXECUTION"
                  trace={result.replay}
                  injection
                  divergenceId={result.comparison.replay_first_decision_event}
                />
              </div>
              <div className="findings">
                <div className="panel-heading">
                  <div>
                    <small>OUTCOME EVIDENCE</small>
                    <b>Evaluation checks</b>
                  </div>
                  <ShieldCheck size={17} />
                </div>
                {result.replay.evaluation.behavior_tags?.includes(
                  "false_success",
                ) && (
                  <div className="false-success">
                    <strong>FALSE SUCCESS</strong>
                    <span>
                      The agent reported completion while the simulator recorded
                      no committed label effect.
                    </span>
                  </div>
                )}
                {result.replay.evaluation.checks.map((check) => (
                  <div
                    className={`check ${check.passed === true ? "ok" : check.passed === false ? "bad" : "unknown"}`}
                    key={check.id}
                  >
                    <div className="check-mark">
                      {check.passed === true ? (
                        <Check size={14} />
                      ) : check.passed === false ? (
                        <CircleAlert size={14} />
                      ) : (
                        <span>?</span>
                      )}
                    </div>
                    <div>
                      <b>{check.description}</b>
                      <span>
                        {check.passed === true
                          ? "Evidence satisfied"
                          : check.passed === false
                            ? "Contract violated"
                            : "Insufficient evidence"}
                      </span>
                      {check.evidence_event_ids.map((id) => (
                        <button
                          className="evidence-link"
                          key={id}
                          onClick={() => openEvidence(id)}
                        >
                          Open evidence ↗
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
                <div className="note">
                  <b>Interpretation</b>
                  <p>
                    {result.replay.agent === "ollama"
                      ? "Local model output is a fresh generation; divergence is observable evidence, not proof of causation."
                      : "This is a deterministic scripted strategy result, not an LLM measurement."}
                  </p>
                </div>
              </div>
              <div className="response-diff">
                <div className="panel-heading">
                  <div>
                    <small>MODEL-VISIBLE OBSERVATION</small>
                    <b>{injection?.tool_name ?? "Injected response"}</b>
                  </div>
                  <span>ORIGINAL → MUTATED</span>
                </div>
                <div className="json-columns">
                  <div>
                    <small>ORIGINAL</small>
                    <pre>
                      {JSON.stringify(original, null, 2) ??
                        "No injection recorded"}
                    </pre>
                  </div>
                  <div>
                    <small>INJECTED</small>
                    <pre>
                      {JSON.stringify(modified, null, 2) ??
                        "No injection recorded"}
                    </pre>
                  </div>
                </div>
              </div>
            </div>
          </>
        ) : (
          <div className="empty">
            <div className="empty-icon">
              <ArrowDown size={18} />
            </div>
            <b>{loading ? "Loading scenarios" : "Ready when you are"}</b>
            <span>
              {runs.length
                ? "Select a saved baseline or record one, then launch a faulted continuation."
                : "Run the bundled order workflow to create a baseline and isolated replay."}
            </span>
            <button
              onClick={runExperiment}
              disabled={!scenario || !fault || busy || loading}
            >
              <Play size={13} fill="currentColor" /> Run experiment
            </button>
          </div>
        )}
        <footer>
          <span>
            TOOLFAULT <i>·</i> LOCAL-FIRST AGENT DEBUGGING
          </span>
          <span>
            SIMULATOR ONLY <i>·</i> <span className="online">●</span> NO
            PRODUCTION FALLBACK
          </span>
        </footer>
      </section>
    </main>
  );
}
