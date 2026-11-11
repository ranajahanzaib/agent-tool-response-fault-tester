# Trace format

Traces are JSON objects at schema version `1.0`. They carry a run and scenario ID,
agent kind, model metadata, a scenario snapshot, ordered events, final simulator
snapshot, effect ledger, and deterministic evaluation checks. The scenario snapshot
preserves instructions, tool schemas, starting world, and checks for replay review.
Event IDs are UUIDs and sequence numbers are zero-based. Tool observations retain a
normalized transport/body envelope. The current implementation rejects versions
other than `1.0` through Pydantic. Ollama traces also retain assistant messages and
tool-call correlation needed to rebuild continuation context.

Do not treat free-form final text as authoritative evidence of simulator effects.
The effect ledger is the authoritative record for simulated operations. Exported
traces redact common sensitive keys, but that may alter replay inputs; inspect and
retain private originals if exact reproduction matters.
