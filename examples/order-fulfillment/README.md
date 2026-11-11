# Order fulfillment example

This isolated world contains order `ORD-104`, a verified shipping address, two
inventory records, and an empty label ledger. The baseline checks that a label was
committed and that the agent's structured final result reports completion.

The six files in `faults/` are usable with `agent-tool-response-fault-tester replay --fault`. The scenario
also declares matching fault definitions for dashboard selection:

- `missing-verification.yaml` removes the verification result.
- `stale-inventory.yaml` moves the inventory timestamp outside the explicit freshness contract.
- `truncated-inventory.yaml` removes an item while preserving the `total` count.
- `conflicting-address.yaml` violates the declared approved-address invariant.
- `reordered-inventory.yaml` reverses the same inventory items.
- `business-error.yaml` returns an HTTP 200 business rejection and commits no label effect.

Run the safe scripted strategy with `uv run agent-tool-response-fault-tester record scenario.yaml
--agent scripted-cautious`. Run the deterministic unsafe demonstration with
`uv run agent-tool-response-fault-tester demo --agent scripted-unsafe`. Scripted output is fixture evidence,
not a language-model measurement.
