"""Deterministic JSON-observation mutations."""

from __future__ import annotations

import copy
import random
from datetime import UTC, datetime, timedelta
from typing import Any

from .models import FaultSpec


class FaultError(ValueError):
    pass


def _parts(pointer: str | None) -> list[str]:
    if not pointer or not pointer.startswith("/"):
        raise FaultError("JSON Pointer must start with '/'.")
    for segment in pointer[1:].split("/"):
        for index, character in enumerate(segment):
            if character == "~" and (index + 1 == len(segment) or segment[index + 1] not in "01"):
                raise FaultError(f"Invalid JSON Pointer escape in {pointer}")
    return [p.replace("~1", "/").replace("~0", "~") for p in pointer[1:].split("/")]


def _parent(value: Any, pointer: str | None) -> tuple[Any, str]:
    parts = _parts(pointer)
    current = value
    for part in parts[:-1]:
        try:
            current = current[int(part)] if isinstance(current, list) else current[part]
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise FaultError(f"JSON Pointer does not exist: {pointer}") from exc
    return current, parts[-1]


def _read(value: Any, pointer: str | None) -> Any:
    parent, key = _parent(value, pointer)
    try:
        return parent[int(key)] if isinstance(parent, list) else parent[key]
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise FaultError(f"JSON Pointer does not exist: {pointer}") from exc


def mutate(observation: Any, fault: FaultSpec) -> tuple[Any, dict[str, Any]]:
    """Return a copied mutated observation plus evidence of whether a change applied."""
    result = copy.deepcopy(observation)
    mutation = fault.mutation
    if mutation.type == "remove_field":
        parent, key = _parent(result, mutation.path)
        try:
            if isinstance(parent, list):
                del parent[int(key)]
            else:
                del parent[key]
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise FaultError(f"JSON Pointer does not exist: {mutation.path}") from exc
    elif mutation.type == "stale_data":
        stamp = _read(result, mutation.path)
        try:
            parsed = datetime.fromisoformat(str(stamp))
        except ValueError as exc:
            raise FaultError(f"Invalid timestamp at {mutation.path}") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        if mutation.max_age_seconds is None or mutation.reference_time is None:
            raise FaultError(
                "stale_data requires max_age_seconds and an authoritative reference_time"
            )
        reference = mutation.reference_time
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=UTC)
        stale = reference.astimezone(UTC) - timedelta(seconds=mutation.max_age_seconds + 1)
        parent, key = _parent(result, mutation.path)
        if isinstance(parent, list):
            parent[int(key)] = stale.isoformat()
        else:
            parent[key] = stale.isoformat()
    elif mutation.type == "truncate":
        collection = _read(result, mutation.path)
        if not isinstance(collection, list):
            raise FaultError("truncate target must point to an array")
        count = mutation.count if mutation.count is not None else max(0, len(collection) - 1)
        if count >= len(collection):
            return result, {"applied": False, "reason": "truncate count does not omit any items"}
        parent, key = _parent(result, mutation.path)
        truncated = collection[:count]
        if isinstance(parent, list):
            parent[int(key)] = truncated
        else:
            parent[key] = truncated
    elif mutation.type == "conflict":
        _read(result, mutation.path)
        parent, key = _parent(result, mutation.path)
        if isinstance(parent, list):
            parent[int(key)] = mutation.value
        else:
            parent[key] = mutation.value
    elif mutation.type == "reorder":
        collection = _read(result, mutation.path)
        if not isinstance(collection, list):
            raise FaultError("reorder target must point to an array")
        if len(collection) < 2:
            return result, {"applied": False, "reason": "ordering requires at least two items"}
        items = list(collection)
        if mutation.strategy == "reverse":
            items.reverse()
        else:
            random.Random(mutation.seed).shuffle(items)
            if items == collection:
                items = items[1:] + items[:1]
        if items == collection:
            return result, {
                "applied": False,
                "reason": "ordering is indistinguishable for these items",
            }
        parent, key = _parent(result, mutation.path)
        if isinstance(parent, list):
            parent[int(key)] = items
        else:
            parent[key] = items
    elif mutation.type == "business_error":
        if not isinstance(result, dict):
            raise FaultError("business_error target must be an object")
        result.setdefault("transport_status", 200)
        body = result.setdefault("body", {})
        if not isinstance(body, dict):
            raise FaultError("business_error body must be an object")
        body["ok"] = False
        body["error"] = mutation.error_value or {
            "code": "SIMULATED_FAILURE",
            "message": "Operation failed",
        }
        if mutation.status_path:
            _read(body, mutation.status_path)
            parent, key = _parent(body, mutation.status_path)
            if isinstance(parent, list):
                parent[int(key)] = False
            else:
                parent[key] = False
    return result, {"applied": True, "type": mutation.type}
