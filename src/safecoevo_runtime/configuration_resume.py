"""Validate journal configuration when an API endpoint changes."""

API_FIELDS = frozenset({
    'runtime_api_config', 'api_config_file_sha256',
    'model', 'base_url',
    'target_fallback_model', 'target_fallback_base_url', 'target_fallback_api_key_sha256',
    'evolution_model', 'evolution_base_url',
    'evolution_fallback_model', 'evolution_fallback_base_url', 'evolution_fallback_api_key_sha256',
    # Identical Guard replicas may use distinct endpoints; model and behavior stay frozen.
    'guard_base_url',
    # Judge fallback routing may change without changing its scoring contract.
    'agent_safetybench_judge_fallback_base_url',
    # Memory review uses the same transport identity without role prefixes.
    'fallback_model', 'fallback_base_url', 'fallback_request_retries',
})


def api_only_configuration_change(before, after, *, identity_key='execution_contract'):
    """Return true only when a configuration differs exclusively by API identity.

    The surrounding feature contract (prompt digest, budgets, retrieval and
    review/consolidation semantics) must remain byte-for-byte equivalent.
    """
    if not isinstance(before, dict) or not isinstance(after, dict):
        return False
    if {k: v for k, v in before.items() if k != identity_key} != \
       {k: v for k, v in after.items() if k != identity_key}:
        return False
    old_identity, new_identity = before.get(identity_key), after.get(identity_key)
    if not isinstance(old_identity, dict) or not isinstance(new_identity, dict):
        return False
    changed = {key for key in old_identity.keys() | new_identity.keys()
               if old_identity.get(key) != new_identity.get(key)}
    return bool(changed) and changed <= API_FIELDS


def recorded_configuration_matches(recorded, current, ordinal, history=()):
    if recorded == current:
        return True
    if type(ordinal) is not int or ordinal < 1:
        return False
    for transition in history:
        boundary = transition.get('through_ordinal')
        old = transition.get('configuration')
        if type(boundary) is not int or ordinal > boundary or not isinstance(old, dict):
            continue
        if old != recorded or not isinstance(current, dict):
            continue
        if api_only_configuration_change(old, current):
            return True
    return False
