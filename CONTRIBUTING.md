# Contributing

Bug reports, documentation improvements, and focused code changes are welcome.
For behavior changes, include a test that captures the expected result. Keep
simulator evidence distinct from results produced by a local model.

## Development setup

Install [uv](https://docs.astral.sh/uv/), Node.js 22 or newer, and pnpm 12,
then clone the repository and install its dependencies:

```bash
git clone https://github.com/ranajahanzaib/agent-tool-response-fault-tester.git
cd agent-tool-response-fault-tester
uv sync --locked --extra dev
(cd web && pnpm install --frozen-lockfile)
```

Run the checks before opening a pull request:

```bash
cd ..
uv run ruff check .
uv run ruff format --check .
uv run pytest
(cd web && pnpm typecheck && pnpm test && pnpm build)
```

Ollama tests are opt-in and require a local Ollama server and model. See the
README for the `TOOLFAULT_TEST_OLLAMA=1` command.

## Pull requests

Use a short branch name that describes the change. In the pull request, explain
the user-visible behavior, note any compatibility or security impact, and list
the checks you ran. Update relevant documentation when commands, trace formats,
scenario fields, or replay behavior change. Avoid including real credentials or
private trace payloads in commits, tests, screenshots, or issue reports.

For larger changes, open an issue first to discuss the scope. A review does not
guarantee acceptance or a particular merge timeline.
