"""Load repository-local API routing settings for SafeCoEvo components."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs/runtime_api_config.json"
REQUIRED_ENDPOINT_FIELDS = ("model", "base_url", "api_key")
_ACTIVE_CONFIG_PATH: Path | None = None


def _endpoint(data: Any, *, name: str) -> dict[str, str]:
    if not isinstance(data, dict):
        raise ValueError(f"runtime API config {name} must be an object")
    value = {field: str(data.get(field, "")).strip() for field in REQUIRED_ENDPOINT_FIELDS}
    missing = [field for field, item in value.items() if not item or item.startswith("REPLACE_WITH_")]
    if missing:
        raise ValueError(f"runtime API config {name} has missing fields: {', '.join(missing)}")
    return value


def _user_agent(data: Any, *, name: str) -> str:
    if not isinstance(data, dict):
        return ""
    value = data.get("user_agent", "")
    if not isinstance(value, str):
        raise ValueError(f"runtime API config {name}.user_agent must be a string")
    return value.strip()


def _transport(data: Any, *, name: str) -> dict[str, object] | None:
    if not isinstance(data, dict):
        return None
    nested = data.get("transport")
    direct = {key: data[key] for key in ("host_header", "verify_tls", "trust_env") if key in data}
    if nested is not None and direct:
        raise ValueError(f"runtime API config {name} must use either transport or direct transport fields")
    raw = nested if nested is not None else direct
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"runtime API config {name}.transport must be an object")
    unknown = set(raw) - {"host_header", "verify_tls", "trust_env"}
    if unknown:
        raise ValueError(f"runtime API config {name}.transport has unsupported fields: {sorted(unknown)}")
    host_header = raw.get("host_header", "")
    verify_tls = raw.get("verify_tls", True)
    trust_env = raw.get("trust_env", True)
    if not isinstance(host_header, str) or not isinstance(verify_tls, bool) or not isinstance(trust_env, bool):
        raise ValueError(f"runtime API config {name}.transport has invalid field types")
    if not verify_tls and not host_header.strip():
        raise ValueError(f"runtime API config {name}.transport disables TLS verification without host_header")
    return {"host_header": host_header.strip(), "verify_tls": verify_tls, "trust_env": trust_env}


def load_runtime_api_config(path: Path | None = None, *, override: bool = True) -> Path | None:
    """Load a runtime API config; the selected file overrides inherited routing."""
    global _ACTIVE_CONFIG_PATH
    config_path = Path(path or _ACTIVE_CONFIG_PATH or DEFAULT_CONFIG_PATH)
    if not config_path.is_file():
        return None
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("runtime API config must be a schema_version=1 JSON object")

    target_primary_raw = raw.get("target", {}).get("primary")
    target_fallback_raw = raw.get("target", {}).get("fallback")
    evolution_primary_raw = raw.get("evolution", {}).get("primary")
    evolution_fallback_raw = raw.get("evolution", {}).get("fallback")
    target_primary = _endpoint(target_primary_raw, name="target.primary")
    target_fallback = _endpoint(target_fallback_raw, name="target.fallback")
    evolution_primary = _endpoint(evolution_primary_raw, name="evolution.primary")
    evolution_fallback = _endpoint(evolution_fallback_raw, name="evolution.fallback")
    judge_primary = _endpoint(raw.get("judge", {}).get("primary"), name="judge.primary")
    judge_fallback = _endpoint(raw.get("judge", {}).get("fallback"), name="judge.fallback")
    judge_primary_user_agent = _user_agent(raw.get("judge", {}).get("primary"), name="judge.primary")
    judge_fallback_user_agent = _user_agent(raw.get("judge", {}).get("fallback"), name="judge.fallback")
    guard = _endpoint(raw.get("guard"), name="guard")
    embedding_raw = raw.get("embedding")
    embedding = _endpoint(embedding_raw, name="embedding") if isinstance(embedding_raw, dict) else None

    values = {
        "INF_API_KEY": target_primary["api_key"],
        "SAFECOEVO_TARGET_MODEL": target_primary["model"],
        "SAFECOEVO_TARGET_BASE_URL": target_primary["base_url"],
        "SAFECOEVO_TARGET_FALLBACK_MODEL": target_fallback["model"],
        "SAFECOEVO_TARGET_FALLBACK_BASE_URL": target_fallback["base_url"],
        "SAFECOEVO_TARGET_FALLBACK_API_KEY": target_fallback["api_key"],
        "SAFECOEVO_EVOLUTION_MODEL": evolution_primary["model"],
        "SAFECOEVO_EVOLUTION_BASE_URL": evolution_primary["base_url"],
        "SAFECOEVO_EVOLUTION_API_KEY": evolution_primary["api_key"],
        "SAFECOEVO_EVOLUTION_FALLBACK_MODEL": evolution_fallback["model"],
        "SAFECOEVO_EVOLUTION_FALLBACK_BASE_URL": evolution_fallback["base_url"],
        "SAFECOEVO_EVOLUTION_FALLBACK_API_KEY": evolution_fallback["api_key"],
        "SAFECOEVO_AGENT_SAFETYBENCH_JUDGE_MODEL": judge_primary["model"],
        "SAFECOEVO_AGENT_SAFETYBENCH_JUDGE_BASE_URL": judge_primary["base_url"],
        "SAFECOEVO_AGENT_SAFETYBENCH_JUDGE_API_KEY_ENV": "SAFECOEVO_JUDGE_API_KEY",
        "SAFECOEVO_JUDGE_API_KEY": judge_primary["api_key"],
        "SAFECOEVO_AGENT_SAFETYBENCH_JUDGE_FALLBACK_MODEL": judge_fallback["model"],
        "SAFECOEVO_AGENT_SAFETYBENCH_JUDGE_FALLBACK_BASE_URL": judge_fallback["base_url"],
        "SAFECOEVO_AGENT_SAFETYBENCH_JUDGE_FALLBACK_API_KEY_ENV": "SAFECOEVO_JUDGE_FALLBACK_API_KEY",
        "SAFECOEVO_JUDGE_FALLBACK_API_KEY": judge_fallback["api_key"],
        "AGENTDOG_MODEL": guard["model"],
        "AGENTDOG_BASE_URL": guard["base_url"],
        "AGENTDOG_API_KEY": guard["api_key"],
    }
    # This endpoint belongs to the native ASB memory resolver. SafeCoEvo's
    # Memory and Skill retrieval use local lexical matching.
    if embedding is not None:
        values.update({
            "ASB_EMBEDDING_API_KEY": embedding["api_key"],
            "ASB_EMBEDDING_BASE_URL": embedding["base_url"],
            "ASB_EMBEDDING_MODEL": embedding["model"],
        })
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    for key, value in (
        ("SAFECOEVO_AGENT_SAFETYBENCH_JUDGE_USER_AGENT", judge_primary_user_agent),
        ("SAFECOEVO_AGENT_SAFETYBENCH_JUDGE_FALLBACK_USER_AGENT", judge_fallback_user_agent),
    ):
        if value:
            if override or key not in os.environ:
                os.environ[key] = value
        elif override:
            os.environ.pop(key, None)
    for key, value in (
        ("SAFECOEVO_TARGET_PRIMARY_TRANSPORT_JSON", _transport(target_primary_raw, name="target.primary")),
        ("SAFECOEVO_TARGET_FALLBACK_TRANSPORT_JSON", _transport(target_fallback_raw, name="target.fallback")),
        ("SAFECOEVO_EVOLUTION_PRIMARY_TRANSPORT_JSON", _transport(evolution_primary_raw, name="evolution.primary")),
        ("SAFECOEVO_EVOLUTION_FALLBACK_TRANSPORT_JSON", _transport(evolution_fallback_raw, name="evolution.fallback")),
        ("SAFECOEVO_JUDGE_PRIMARY_TRANSPORT_JSON", _transport(raw.get("judge", {}).get("primary"), name="judge.primary")),
        ("SAFECOEVO_JUDGE_FALLBACK_TRANSPORT_JSON", _transport(raw.get("judge", {}).get("fallback"), name="judge.fallback")),
    ):
        if value is None:
            if override:
                os.environ.pop(key, None)
        elif override or key not in os.environ:
            os.environ[key] = json.dumps(value, sort_keys=True)
    _ACTIVE_CONFIG_PATH = config_path
    return config_path


def require_runtime_api_config(path: Path | None = None) -> Path:
    """Load a validated JSON file, refusing credential-free execution."""
    config_path = Path(path or DEFAULT_CONFIG_PATH)
    if not config_path.is_file():
        raise ValueError(f"runtime API config file does not exist: {config_path}")
    loaded = load_runtime_api_config(config_path)
    assert loaded is not None
    return loaded


def load_embedding_settings(path: Path | None = None) -> dict[str, str] | None:
    config_path = Path(path or _ACTIVE_CONFIG_PATH or DEFAULT_CONFIG_PATH)
    if not config_path.is_file():
        return None
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    value = raw.get("embedding")
    return _endpoint(value, name="embedding") if isinstance(value, dict) else None


def redacted_runtime_api_config_summary(path: Path | None = None) -> dict[str, Any]:
    config_path = load_runtime_api_config(path)
    return {"path": str(config_path) if config_path else None, "loaded": config_path is not None, "asb_memory_embedding_configured": load_embedding_settings() is not None}
