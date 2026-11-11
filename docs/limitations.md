# Limitations

- Ollama tool calling and continuation support the bundled simulator; arbitrary
  production tools and agent frameworks are outside scope.
- The scripted fixture uses a fixed ordered tool-call sequence with cautious or unsafe
  decision policies; it does not generate alternate tool-call sequences.
- Only the included order-fulfillment simulator is supported.
- Scenario outcome checks are deliberately small and deterministic.
- No production tool calls, arbitrary Python adapters, or cloud inference calls.
- Export redacts common sensitive keys by default; inspect artifacts before sharing,
  since key-based redaction cannot identify every sensitive value.
- The dashboard can choose a saved passing baseline, scenario fault, scripted strategy,
  or Ollama model and renders paired timelines and observation diffs.
- Dashboard record and replay requests are synchronous, with configured model and run
  timeouts; there is no background job queue, progress endpoint, or cancellation.
- The HTML report includes paired timelines and checks, but is less interactive than
  the dashboard.
- A changed observation alone does not establish that it caused a decision change.
- A passing fixture does not establish general agent reliability.
