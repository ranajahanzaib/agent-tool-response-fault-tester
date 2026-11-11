# Fault types

All faults are applied to a copied observation at the model-visible boundary. JSON
Pointers use RFC 6901 escaping. Supported types are `remove_field`, `stale_data`,
`truncate`, `conflict`, `reorder` (reverse or seeded shuffle), and `business_error`.
Missing paths and wrong value types raise a visible error. One-element ordering is
reported as an inapplicable no-op. Staleness requires an explicit maximum age, a
parseable timestamp, and an authoritative `reference_time`. The mutation sets the
field older than that explicit threshold.

Truncation currently supports JSON arrays; incomplete text-body truncation is not
implemented. Seeded shuffles return a no-op result if equal items make the order
indistinguishable.

The bundled scenario declares an inventory freshness contract, completeness via a
`total` count, and an address invariant. These are simple explicit checks rather than
a general contract language.
