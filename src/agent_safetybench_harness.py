"""Native Agent-SafetyBench environment adapter for SafeCoEvo."""

from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path
from typing import Any

class SafeEnvironmentSession:
    def __init__(self, dataset_root: Path, environments: list[dict[str, Any]]):
        # The release represents no-environment cases with an empty-name entry.
        self.environments = [spec for spec in environments if str(spec.get("name", ""))]
        self.instances: dict[str, Any] = {}
        env_dir = str(dataset_root / "environments")
        if env_dir not in sys.path:
            sys.path.insert(0, env_dir)
        for spec in self.environments:
            name = str(spec["name"])
            module = importlib.import_module(name)
            self.instances[name] = getattr(module, name)(parameters=copy.deepcopy(spec.get("parameters", {})))

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        for spec in self.environments:
            if name in spec.get("tools", []):
                try:
                    result = self.instances[str(spec["name"])].call_tool(name, copy.deepcopy(arguments))
                    return result if isinstance(result, dict) else {"success": True, "result": result}
                except Exception:
                    return {"success": False, "message": "Environment tool failed."}
        return {"success": False, "message": "Unknown or unavailable tool."}
