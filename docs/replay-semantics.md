# Replay semantics

Runs start from deep-copied initial simulator state. Model-backed replay rebuilds
Ollama chat history from recorded assistant and tool messages up to the selected
response, injects the modified observation, then requests a fresh model decision.
New tool calls are resolved only by the scenario simulator; unknown tools return an
unsupported payload. No production API fallback exists. The scripted demo walks a
deterministic policy fixture and works without a model download.

Response-only faults preserve the simulated operation's actual committed effect;
changing the observation alone does not undo an operation. A fault can instead set
`execution_outcome: business_failure`, which simulates a rejected operation with no
committed effect while injecting the corresponding faulty observation. The flagship
scenario uses this mode. Generations can vary between runs even with matching model
settings. Comparisons report agent decision divergence separately from simulator
outcome changes. An agent can make the same decision after injection while the
simulated operation has a different result. Scripted results are fixture evidence,
not model measurements.
