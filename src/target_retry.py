"""Auditable target-model retry and fallback calls for benchmark runners."""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

# The main entry point loads endpoint configuration before importing this module.
DEFAULT_FALLBACK_MODEL = os.environ.get("SAFECOEVO_TARGET_FALLBACK_MODEL", "")
DEFAULT_FALLBACK_BASE_URL = os.environ.get("SAFECOEVO_TARGET_FALLBACK_BASE_URL", "")
DEFAULT_FALLBACK_API_KEY = os.environ.get("SAFECOEVO_TARGET_FALLBACK_API_KEY", "")


def _redact_secrets(text: str, secrets: tuple[str, ...]) -> str:
    for secret in dict.fromkeys(secrets):
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


class TargetModelRequestTimeout(TimeoutError):
    """Raised when the target endpoint exceeds the runner's wall-clock limit."""


def _target_endpoint_transport(label: str, role: str = "target") -> dict[str, object] | None:
    """Return the optional JSON transport profile for one configured Target endpoint."""

    if role not in {"target", "evolution", "judge"}:
        raise ValueError("Unsupported completion transport role")
    raw = os.environ.get(f"SAFECOEVO_{role.upper()}_{label.upper()}_TRANSPORT_JSON", "").strip()
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid target transport JSON for {label}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Target transport for {label} must be an object")
    host_header = value.get("host_header", "")
    verify_tls = value.get("verify_tls", True)
    trust_env = value.get("trust_env", True)
    if not isinstance(host_header, str) or not isinstance(verify_tls, bool) or not isinstance(trust_env, bool):
        raise ValueError(f"Invalid target transport fields for {label}")
    if not verify_tls and not host_header.strip():
        raise ValueError(f"Target transport for {label} disables TLS verification without a host_header")
    return {"host_header": host_header.strip(), "verify_tls": verify_tls, "trust_env": trust_env}


@dataclass(frozen=True)
class TargetEndpoint:
    """One OpenAI-compatible target endpoint."""

    label: str
    model: str
    base_url: str
    api_key: str
    retries: int
    transport: dict[str, object] | None = None
    user_agent: str = ""


def _target_completion_worker(
    endpoint: TargetEndpoint,
    request: dict[str, Any],
    timeout_seconds: float,
    send_connection: Any,
) -> None:
    """Run one target request in a disposable child process."""

    http_client: Any | None = None
    try:
        from openai import OpenAI

        payload = dict(request)
        payload["model"] = endpoint.model
        if endpoint.model.lower().startswith("gpt-") and "max_tokens" in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")
        # Qwen3.8 deployments run with thinking enabled by default; disable it
        # so reasoning text never consumes latency or the max_tokens budget.
        if "qwen" in str(endpoint.model).lower():
            payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        if not payload.get("tools"):
            # Several OpenAI-compatible endpoints reject an explicit empty
            # tools array; no-tool tasks must omit tool fields entirely.
            payload.pop("tools", None)
            payload.pop("parallel_tool_calls", None)
        client_kwargs: dict[str, Any] = {
            "base_url": endpoint.base_url,
            "api_key": endpoint.api_key,
            "timeout": timeout_seconds,
            "max_retries": 0,
        }
        if endpoint.user_agent:
            client_kwargs["default_headers"] = {"User-Agent": endpoint.user_agent}
        if endpoint.transport is not None:
            import httpx

            http_client = httpx.Client(
                verify=bool(endpoint.transport["verify_tls"]),
                trust_env=bool(endpoint.transport["trust_env"]),
                timeout=timeout_seconds,
            )
            client_kwargs["http_client"] = http_client
            host_header = str(endpoint.transport["host_header"])
            if host_header:
                client_kwargs.setdefault("default_headers", {})["Host"] = host_header
        client = OpenAI(
            **client_kwargs,
        )
        message = client.chat.completions.create(**payload).choices[0].message
        tool_calls = [
            {
                "id": call.id,
                "name": call.function.name,
                "arguments": call.function.arguments,
            }
            for call in (message.tool_calls or [])
        ]
        send_connection.send(
            {
                "ok": True,
                "content": message.content,
                "tool_calls": tool_calls,
                "model": endpoint.model,
                "endpoint": endpoint.label,
            }
        )
    except BaseException as exc:  # noqa: BLE001 - child must serialize all failures.
        send_connection.send(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": _redact_secrets(str(exc), (endpoint.api_key,)),
                "model": endpoint.model,
                "endpoint": endpoint.label,
            }
        )
    finally:
        if http_client is not None:
            http_client.close()
        send_connection.close()


def _single_completion_with_deadline(
    endpoint: TargetEndpoint,
    request: dict[str, Any],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Call one endpoint once with a hard wall-clock deadline."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    try:
        context = multiprocessing.get_context("fork")
    except ValueError:  # pragma: no cover - all supported runners use Linux.
        raise RuntimeError("Target retry helper requires a fork-capable runtime")

    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_target_completion_worker,
        args=(endpoint, request, timeout_seconds, send_connection),
        daemon=True,
    )
    process.start()
    send_connection.close()
    deadline = time.monotonic() + timeout_seconds
    try:
        payload: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if receive_connection.poll(min(0.2, max(0.0, remaining))):
                payload = receive_connection.recv()
                break
            if not process.is_alive():
                break
        if payload is None:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():  # pragma: no cover - terminate normally succeeds.
                    process.kill()
                    process.join(timeout=5)
                raise TargetModelRequestTimeout(
                    f"Target model request exceeded {timeout_seconds:.1f} seconds"
                )
            raise RuntimeError("Target model worker exited without a result payload")
        if not payload.get("ok"):
            raise RuntimeError(
                f"Target model worker failed: {payload.get('error_type')}: {payload.get('error')}"
            )
        return payload
    finally:
        receive_connection.close()
        if process.is_alive():
            process.join(timeout=5)


def _message_from_payload(payload: dict[str, Any], attempts: list[dict[str, Any]]) -> Any:
    tool_calls = [
        SimpleNamespace(
            id=item.get("id"),
            function=SimpleNamespace(
                name=item.get("name"),
                arguments=item.get("arguments"),
            ),
        )
        for item in payload.get("tool_calls", [])
    ]
    return SimpleNamespace(
        content=payload.get("content"),
        tool_calls=tool_calls,
        model=payload.get("model"),
        target_endpoint=payload.get("endpoint"),
        target_retry_attempts=attempts,
    )


def completion_with_retry_and_fallback(
    *,
    request: dict[str, Any],
    primary_model: str,
    primary_base_url: str,
    primary_api_key: str,
    timeout_seconds: float,
    primary_retries: int = 3,
    fallback_model: str = DEFAULT_FALLBACK_MODEL,
    fallback_base_url: str = DEFAULT_FALLBACK_BASE_URL,
    fallback_api_key: str = DEFAULT_FALLBACK_API_KEY,
    fallback_retries: int = 2,
    retry_sleep_seconds: float = 5.0,
    transport_role: str = "target",
    primary_user_agent: str = "",
    fallback_user_agent: str = "",
) -> Any:
    """Call primary target first, then the fixed fallback endpoint if needed.

    ``primary_retries`` and ``fallback_retries`` are retry counts after the
    first attempt, so the default performs up to 4 primary attempts and then up
    to 3 fallback attempts. No benchmark state changes happen until a response
    is returned, so these request-level retries do not retest a completed task.
    """

    if primary_retries < 0 or fallback_retries < 0:
        raise ValueError("Target model retries must be non-negative")
    if retry_sleep_seconds < 0:
        raise ValueError("Target model retry sleep must be non-negative")

    endpoints = [
        TargetEndpoint(
            "primary", primary_model, primary_base_url, primary_api_key, primary_retries,
            _target_endpoint_transport("primary", transport_role), primary_user_agent,
        ),
        TargetEndpoint(
            "fallback", fallback_model, fallback_base_url, fallback_api_key, fallback_retries,
            _target_endpoint_transport("fallback", transport_role), fallback_user_agent,
        ),
    ]
    attempts: list[dict[str, Any]] = []
    for endpoint in endpoints:
        if not endpoint.api_key:
            attempts.append(
                {
                    "endpoint": endpoint.label,
                    "model": endpoint.model,
                    "base_url": endpoint.base_url,
                    "ok": False,
                    "error_type": "MissingAPIKey",
                    "error": "empty API key",
                }
            )
            continue
        for attempt_index in range(endpoint.retries + 1):
            try:
                payload = _single_completion_with_deadline(
                    endpoint,
                    request,
                    timeout_seconds=timeout_seconds,
                )
                attempts.append(
                    {
                        "endpoint": endpoint.label,
                        "model": endpoint.model,
                        "base_url": endpoint.base_url,
                        "attempt": attempt_index + 1,
                        "ok": True,
                    }
                )
                return _message_from_payload(payload, attempts)
            except Exception as exc:  # noqa: BLE001 - failures are audited and retried.
                attempts.append(
                    {
                        "endpoint": endpoint.label,
                        "model": endpoint.model,
                        "base_url": endpoint.base_url,
                        "attempt": attempt_index + 1,
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": _redact_secrets(
                            str(exc),
                            (primary_api_key, fallback_api_key),
                        )[:1000],
                    }
                )
                if attempt_index < endpoint.retries and retry_sleep_seconds:
                    time.sleep(retry_sleep_seconds * (attempt_index + 1))
    raise RuntimeError(
        "Target model request failed on primary and fallback endpoints: "
        + json.dumps(attempts, ensure_ascii=False)
    )
