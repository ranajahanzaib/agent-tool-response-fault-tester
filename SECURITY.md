# Security

Agent Tool Response Fault Tester is a local developer tool. The API binds to loopback by default and the
simulator never calls production services. Unknown tools are represented as
unsupported results. Do not place secrets in scenarios or share traces without
reviewing them: payloads can contain sensitive data. Export redaction handles common
keys and supports additional comma-separated names with `TOOLFAULT_REDACT_KEYS`, but
key matching cannot identify every sensitive value.

## Reporting a vulnerability

Report suspected vulnerabilities privately through the repository's
[GitHub Security Advisories](https://github.com/ranajahanzaib/agent-tool-response-fault-tester/security/advisories/new)
page when private vulnerability reporting is available. If it is unavailable,
use the contact options on the [project owner's GitHub profile](https://github.com/ranajahanzaib).
Include reproduction steps and impact. Avoid posting exploit details in public
issues before a fix is available.
