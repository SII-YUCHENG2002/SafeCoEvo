"""Optional HTTP settings published by the selected runtime JSON config."""
from contextlib import contextmanager
import json
import os

from runtime_api_config import _transport


@contextmanager
def endpoint_transport_options(role, label, timeout_seconds):
    raw = os.environ.get(f"SAFECOEVO_{role.upper()}_{label.upper()}_TRANSPORT_JSON")
    if not raw:
        yield {}
        return
    profile = _transport(json.loads(raw), name=f"{role}.{label}")
    if profile is None:
        yield {}
        return
    import httpx

    with httpx.Client(verify=profile['verify_tls'], trust_env=profile['trust_env'],
                      timeout=timeout_seconds) as client:
        options = {"http_client": client}
        if profile['host_header']:
            options['default_headers'] = {'Host': profile['host_header']}
        yield options
