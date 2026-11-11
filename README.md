# Agent Tool Response Fault Tester

**Discover which tool responses make AI agents fail.**

Agent Tool Response Fault Tester is a local-first developer tool for recording tool-using workflows,
injecting a controlled response fault, and inspecting the evidence around an
agent's behavior. It focuses on structured response semantics and business outcomes:
HTTP success does not prove an operation succeeded, and plausible JSON may still be
incomplete or contradictory.

> Scripted results are deterministic fixture evidence, not measured model behavior.
> Ollama runs use genuine local inference and are labeled separately in traces.

## Quick start

Requirements: Python 3.11+, `uv`, Node.js 22+, and pnpm 12. The scripted
workflow needs no model download.

```bash
uv sync --locked --extra dev
uv run agent-tool-response-fault-tester validate
uv run agent-tool-response-fault-tester demo --agent scripted
uv run agent-tool-response-fault-tester report .agent-tool-response-fault-tester/reports/<replay-id>.json --format html --out report.html
```

The demo writes portable run JSON under `.agent-tool-response-fault-tester/runs/` and a paired JSON report
under `.agent-tool-response-fault-tester/reports/`. Render that baseline/replay report as standalone HTML:

```bash
ollama pull qwen3:4b
uv run agent-tool-response-fault-tester doctor
uv run agent-tool-response-fault-tester record examples/order-fulfillment/scenario.yaml --agent ollama --model qwen3:4b --out .agent-tool-response-fault-tester/runs/baseline.json
uv run agent-tool-response-fault-tester replay .agent-tool-response-fault-tester/runs/baseline.json --fault examples/order-fulfillment/faults/business-error.yaml --model qwen3:4b --out .agent-tool-response-fault-tester/runs/business-error.json
```

The replay command accepts a standalone fault YAML; the bundled fault definition is
also included in the scenario. See the scenario file for a copyable fault object.
Local inference has conservative turn and tool-call limits. If Ollama is unavailable,
the command records an explicit inconclusive trace rather than pretending a run passed.

## Dashboard

Start the API and frontend in separate terminals:

```bash
uv run agent-tool-response-fault-tester serve
cd web && pnpm install --frozen-lockfile && pnpm dev
```

Open the local Vite URL. The dashboard fetches scenario and experiment data from the
API; it does not substitute a fabricated result if the API is unavailable. It can
launch the bundled scripted example and download its report JSON.

## Scenario and fault

The bundled example is in [`examples/order-fulfillment/scenario.yaml`](examples/order-fulfillment/scenario.yaml).
It declares the initial world, tool schemas, scripted call sequence, outcome rules,
an explicit address invariant, and all six fault configurations. The flagship fault
is:

```yaml
id: business-error-label
name: HTTP 200 with business error
target:
  tool: create_shipping_label
  occurrence: 1
  arguments: { order_id: ORD-104 }
mutation:
  type: business_error
  error_value:
    { code: ADDRESS_REJECTED, message: Shipping address validation failed }
execution_outcome: business_failure
```

Other implemented transformations are `remove_field`, `stale_data`, `truncate`,
`conflict`, and `reorder`. See [fault types](docs/fault-types.md) for requirements
and limits.

## CLI

```text
agent-tool-response-fault-tester doctor
agent-tool-response-fault-tester validate [SCENARIO]
agent-tool-response-fault-tester demo --agent scripted
agent-tool-response-fault-tester record SCENARIO --agent scripted
agent-tool-response-fault-tester replay BASELINE --fault FAULT [--scenario SCENARIO]
agent-tool-response-fault-tester report TRACE_OR_REPORT --format html --out report.html
agent-tool-response-fault-tester serve --host 127.0.0.1 --port 8000
```

`doctor` checks the packaged scenario, local storage, Ollama availability, and the
default model. `record` and `replay` accept conservative model turn/call and timeout
limits. `--agent scripted` records the cautious strategy; use `scripted-unsafe` to
demonstrate false success when the business operation itself fails.

`record` exits 0 for a passing baseline, 1 for a failed baseline, and 2 when the
baseline is inconclusive. `replay` exits 0 when it completed with either a pass or a
verified failure; it exits 2 for an inconclusive run. Invalid inputs and infrastructure
errors return nonzero. `demo` exits 0 when both runs and the report were produced,
even though its intentionally unsafe replay has a failing verdict.

## Replay semantics and safety

Each scripted run receives a deep copy of the initial world. Mutations are applied
to the response payload at the observation boundary and preserve the original in
the trace. Tool implementations are local simulator functions; unsupported tools
never execute. Details and current gaps are documented in
[architecture](docs/architecture.md), [trace format](docs/trace-format.md),
[replay semantics](docs/replay-semantics.md), and [limitations](docs/limitations.md).

Trace payloads may contain sensitive values. Exports redact common sensitive keys,
but review them before sharing because key-based filtering cannot catch every value.
No data is uploaded automatically. Optional settings such as `TOOLFAULT_OLLAMA_URL`
and `TOOLFAULT_REDACT_KEYS` can be exported in the shell or sourced from
`.env.example`; Agent Tool Response Fault Tester does not load `.env` files automatically.

## Development

```bash
uv sync --locked --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pytest
cd web && pnpm install --frozen-lockfile && pnpm typecheck && pnpm test && pnpm build
```

GitHub Actions runs the Python and frontend checks without Ollama. Contributions
should include focused tests and preserve the explicit distinction between simulator
fixtures and actual model executions; see [CONTRIBUTING.md](CONTRIBUTING.md).
The optional local-model test can be run with `TOOLFAULT_TEST_OLLAMA=1 uv run pytest tests/test_ollama_integration.py` after pulling the configured model.

## Community

- [Contributing](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md)
- [Changelog](CHANGELOG.md)

## License

MIT. See [LICENSE](LICENSE).
