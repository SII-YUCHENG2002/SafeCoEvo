"""Raw trajectory storage with a verifiable hash chain."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Protocol

from .events import HarnessEvent


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def event_snapshot(event: HarnessEvent) -> dict[str, Any]:
    """Capture public runtime state without importing a private evaluator view."""

    return {
        "task_id": event.task.task_id,
        "hook": event.hook.value,
        "sequence": event.sequence,
        "agent_messages": copy.deepcopy(event.state.messages),
        "formatted_messages": copy.deepcopy(event.state.formatted_messages),
        "proposed_action": _json_value(event.proposed_action or event.action),
        "effective_action": _json_value(event.action),
        "execution_status": event.execution_status,
        "synthetic_tool_result": event.synthetic_tool_result,
        "intercepted_by": list(event.intercepted_by),
        "tool_result": copy.deepcopy(event.tool_result),
        "guard_requests": _json_value(event.state.guard_requests),
        "guard_verdicts": _json_value(event.state.guard_verdicts),
        "outcome": _json_value(event.outcome),
        "model_messages": copy.deepcopy(event.model_messages),
        "available_tools": copy.deepcopy(event.available_tools),
        "annotations": copy.deepcopy(event.state.annotations),
        "processor_observations": copy.deepcopy(event.state.processor_observations),
        "evolution_signals": copy.deepcopy(event.state.evolution_signals),
        "evolution_evidence": copy.deepcopy(event.state.evolution_evidence),
    }


class TraceStore(Protocol):
    """Stores immutable trace entries and returns their content hashes."""

    def append(self, record: dict[str, Any]) -> str: ...

    def verify(self) -> bool: ...


class InMemoryTraceStore:
    """Trace store used by tests and in-process replay callers."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._previous_hash = ""

    def append(self, record: dict[str, Any]) -> str:
        payload = copy.deepcopy(record)
        payload["previous_hash"] = self._previous_hash
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        entry_hash = hashlib.sha256(encoded).hexdigest()
        payload["entry_hash"] = entry_hash
        self.records.append(payload)
        self._previous_hash = entry_hash
        return entry_hash

    def verify(self) -> bool:
        previous = ""
        for entry in self.records:
            copied = copy.deepcopy(entry)
            expected = copied.pop("entry_hash", None)
            if copied.get("previous_hash") != previous:
                return False
            encoded = json.dumps(copied, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
            if hashlib.sha256(encoded).hexdigest() != expected:
                return False
            previous = str(expected)
        return True


def load_trace_records(path: Path) -> list[dict[str, Any]]:
    """Load a JSON array of trace records."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(record, dict) for record in value):
        raise ValueError(f"Trace must contain a JSON array of objects: {path}")
    return value


class JsonTraceStore(InMemoryTraceStore):
    """In-memory index persisted atomically as a standard JSON array."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.path.is_file():
            self.records = load_trace_records(self.path)
            if not self.verify():
                raise ValueError(f"Existing trace has an invalid hash chain: {self.path}")
            if self.records:
                self._previous_hash = str(self.records[-1]["entry_hash"])
        else:
            self._persist()

    def append(self, record: dict[str, Any]) -> str:
        entry_hash = super().append(record)
        self._persist()
        return entry_hash

    def _persist(self) -> None:
        fd, temporary_name = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.records, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.chmod(0o600)
            os.replace(temporary_path, self.path)
            if os.name != "nt":
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
