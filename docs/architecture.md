# Architecture

The Python package separates Pydantic domain models (`models.py`), deterministic
observation transformations (`faults.py`), and the isolated scenario runner
(`engine.py`). Typer exposes local workflows; FastAPI serves scenario and run
artifacts to the React dashboard. JSON is the portable trace and report format.

The bundled scenario owns its initial world, tool descriptions, scripted calls,
fault contracts, and outcome checks. The runner deep-copies that world for each
execution. Both a deterministic scripted fixture and Ollama native tool calling are
available. The model adapter can resume inference after an injected observation.
