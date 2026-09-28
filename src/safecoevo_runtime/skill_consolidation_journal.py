"""Durable post-case barrier: prepared responses survive checkpoint interruptions."""
from __future__ import annotations

import json
import hashlib
import os
import tempfile
from copy import deepcopy
from pathlib import Path

from .skill_consolidation import SKILL_CONSOLIDATION_SYSTEM_PROMPT, build_consolidation_messages, parse_consolidation_response
from .skills import SkillRegistry


def _prefix_digest(rows):
    from .skill_consolidation import digest
    keys = ('ordinal', 'task_id', 'status', 'official_feedback', 'retrieved_memory_ids', 'skill_usage')
    return digest([{k: r.get(k) for k in keys} for r in rows])


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix='.consolidation-', delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(handle.name, path)
    if os.name != 'nt':
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class ConsolidationRuntime:
    def __init__(self, out_dir, *, interval, max_input_chars=250000, execution_contract=None):
        from .skill_consolidation import digest
        if type(interval) is not int or interval < 1 or type(max_input_chars) is not int or max_input_chars < 1:
            raise ValueError('evidence_aware interval/input budget must be positive integers')
        self.root = Path(out_dir)
        self.path = self.root / 'skill_consolidation_evidence_aware' / 'state.json'
        self.configuration = dict(implementation='evidence_aware', schema_version=2, interval=interval,
            trigger='strict_periodic', max_input_chars=max_input_chars,
            prompt_sha256=hashlib.sha256(SKILL_CONSOLIDATION_SYSTEM_PROMPT.encode()).hexdigest(),
            per_episode_skill_updates='unchanged', execution_contract=deepcopy(execution_contract))
        self.data = dict(configuration=self.configuration, rounds=[], pending=None)
        if self.path.exists():
            envelope = json.loads(self.path.read_text())
            if digest(envelope['data']) != envelope['sha256']:
                raise ValueError('Consolidation journal integrity mismatch')
            self.data = envelope['data']
            if self.data['configuration'] != self.configuration:
                raise ValueError('Consolidation configuration differs; use a fresh output directory')
        self.last_result = self.data['rounds'][-1]['result'] if self.data['rounds'] else None
        self._audits_checked = set()

    def _save(self):
        from .skill_consolidation import digest
        atomic_json(self.path, dict(data=self.data, sha256=digest(self.data)))

    def due(self, ordinal):
        return ordinal > 0 and ordinal % self.configuration['interval'] == 0

    def advance(self, *, rows, memory_items, registry, agent, persist_registry):
        from .skill_consolidation import digest, build_consolidation_pack_evidence_aware, apply_consolidation_candidate_evidence_aware
        interval = self.configuration['interval']
        ordinals = [r['ordinal'] for r in rows]
        if ordinals != list(range(1, len(rows) + 1)):
            raise ValueError('Consolidation requires an exact committed row prefix')
        binding = dict(transport={key: getattr(agent, key, None) for key in (
            'model', 'base_url', 'max_tokens', 'timeout_seconds', 'request_retries',
            'fallback_model', 'fallback_base_url', 'fallback_request_retries', 'retry_sleep_seconds')},
            skill_retrieval=dict(mode=registry.retrieval_mode))
        if 'binding' not in self.data:
            self.data['binding'] = binding
            self._save()
        elif self.data['binding'] != binding:
            raise ValueError('Consolidation model/transport or Skill retrieval configuration changed')
        committed = [r['ordinal'] for r in self.data['rounds']]
        for record in self.data['rounds']:
            if record['ordinal'] not in self._audits_checked:
                audit = json.loads(Path(record['input_path']).read_text())
                if digest(audit) != record['audit_sha256']:
                    raise ValueError('Committed consolidation audit integrity mismatch')
                self._audits_checked.add(record['ordinal'])
        if committed != list(range(interval, (len(committed) + 1) * interval, interval)):
            raise ValueError('Non-contiguous consolidation journal')
        last = committed[-1] if committed else 0
        if last > len(rows):
            raise ValueError('Consolidation journal is ahead of cases')
        if last and self.data['rounds'][-1]['prefix_hash'] != _prefix_digest(rows[:last]):
            raise ValueError('Committed consolidation evidence prefix changed')
        pending = self.data['pending']
        if len(rows) < last + interval:
            if pending:
                raise ValueError('Pending consolidation is ahead of cases')
            if len(rows) == last and last:
                expected = self.data['rounds'][-1]['active_hash']
                if registry.registry_hash != expected:
                    raise ValueError('Committed consolidation differs from active checkpoint')
            if not self.path.exists():
                self._save()
            return registry
        if len(rows) != last + interval:
            raise ValueError('Missed consolidation boundary; refuse retrospective update')
        ordinal = len(rows)
        prefix_hash = _prefix_digest(rows)
        if pending is None:
            pack = build_consolidation_pack_evidence_aware(registry=registry, memory_items=memory_items, rows=rows, since_ordinal=last)
            pending = dict(ordinal=ordinal, prefix_hash=prefix_hash, memory_hash=digest(memory_items),
                           parent=registry.to_dict(), pack=pack, phase='pending')
            self.data['pending'] = pending
            self._save()  # Durable BEFORE the model request.
        if (pending['ordinal'] != ordinal or pending['prefix_hash'] != prefix_hash
                or pending['memory_hash'] != digest(memory_items)
                or pending.get('review_context_hash') is not None):
            raise ValueError('Pending consolidation input changed')
        parent = SkillRegistry.from_dict(pending['parent'])
        if pending['phase'] == 'pending':
            if registry.registry_hash != parent.registry_hash:
                raise ValueError('Pending consolidation parent differs from active checkpoint')
            messages = build_consolidation_messages(SKILL_CONSOLIDATION_SYSTEM_PROMPT, pending['pack'])
            result = dict(status='skipped', reason='input_budget_exceeded')
            if sum(len(m['content']) for m in messages) <= self.configuration['max_input_chars']:
                try:
                    from .evolution_graph import _thinking_disabled_extra_body
                    request = dict(model=agent.model, messages=messages, temperature=0.0, max_tokens=agent.max_tokens)
                    extra = _thinking_disabled_extra_body(str(agent.model), str(agent.base_url))
                    if extra is not None:
                        request['extra_body'] = extra
                    response = agent._chat_completion_with_retry_and_fallback(request)
                    message = response.choices[0].message
                except Exception as exc:
                    # Never persist provider exception strings or credentials.
                    result = dict(status='planner_error', error_type=type(exc).__name__)
                else:
                    pending['raw_response'] = message.content or ''
                    if getattr(message, 'tool_calls', None):
                        result = dict(status='rejected', error_type='UnexpectedToolCalls')
                    else:
                        pending['phase'] = 'responded'
                        # IO failure must stop the stream, never masquerade as model failure.
                        self._save()
            if pending['phase'] == 'pending':
                pending.update(phase='prepared', result=result, successor=None)
                self._save()
        if pending['phase'] == 'responded':
            try:
                proposal = parse_consolidation_response(pending['raw_response'])
                applied = apply_consolidation_candidate_evidence_aware(root=self.root / 'evolution', ordinal=ordinal,
                    registry=parent, response=proposal, memory_items=memory_items)
                pending['result'] = {k: v for k, v in applied.items() if k != 'registry'}
                pending['successor'] = applied['registry'].to_dict() if applied.get('registry') else None
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                text = str(exc)
                pending['repair'] = dict(validation_error_code=text.split(':', 1)[0],
                    validation_error=text[:500])
                pending['phase'] = 'repair_pending'
            else:
                pending['phase'] = 'prepared'
            self._save()  # Publication must never precede durable proposal/result.
        if pending['phase'] == 'repair_pending':
            messages = build_consolidation_messages(SKILL_CONSOLIDATION_SYSTEM_PROMPT, pending['pack'])
            messages.extend([
                {'role': 'assistant', 'content': pending['raw_response']},
                {'role': 'user', 'content': json.dumps({
                    'instruction': 'Return one corrected complete JSON proposal. Change only what is needed to satisfy the validation error. If evidence is insufficient, omit the invalid operation. Do not add unsupported source IDs.',
                    'validation_error_code': pending['repair']['validation_error_code'],
                    'validation_error': pending['repair']['validation_error'],
                }, ensure_ascii=False)},
            ])
            if sum(len(m['content']) for m in messages) > self.configuration['max_input_chars']:
                pending.update(phase='prepared', result=dict(status='rejected',
                    error_type='RepairInputBudgetExceeded'), successor=None)
            else:
                try:
                    from .evolution_graph import _thinking_disabled_extra_body
                    request = dict(model=agent.model, messages=messages, temperature=0.0, max_tokens=agent.max_tokens)
                    extra = _thinking_disabled_extra_body(str(agent.model), str(agent.base_url))
                    if extra is not None:
                        request['extra_body'] = extra
                    response = agent._chat_completion_with_retry_and_fallback(request)
                    message = response.choices[0].message
                    if getattr(message, 'tool_calls', None):
                        raise ValueError('UnexpectedToolCalls')
                    pending['repair']['raw_response'] = message.content or ''
                    pending['phase'] = 'repair_responded'
                except Exception as exc:
                    pending.update(phase='prepared', result=dict(status='rejected',
                        error_type=type(exc).__name__, reason='repair_request_failed'), successor=None)
            self._save()
        if pending['phase'] == 'repair_responded':
            try:
                proposal = parse_consolidation_response(pending['repair']['raw_response'])
                applied = apply_consolidation_candidate_evidence_aware(root=self.root / 'evolution', ordinal=ordinal,
                    registry=parent, response=proposal, memory_items=memory_items)
                pending['result'] = {k: v for k, v in applied.items() if k != 'registry'}
                pending['successor'] = applied['registry'].to_dict() if applied.get('registry') else None
                pending['repair']['status'] = 'accepted'
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                pending.update(result=dict(status='rejected', error_type=type(exc).__name__,
                    reason=str(exc).split(':', 1)[0]), successor=None)
                pending['repair']['status'] = 'rejected'
            pending['phase'] = 'prepared'
            self._save()
        if pending['phase'] != 'prepared':
            raise ValueError('Unknown consolidation journal phase')
        target = SkillRegistry.from_dict(pending['successor']) if pending['successor'] else parent
        if registry.registry_hash not in {parent.registry_hash, target.registry_hash}:
            raise ValueError('Consolidation checkpoint is neither parent nor successor')
        if registry.registry_hash != target.registry_hash:
            # Callback writes version file and checkpoint. A failure propagates: no next task.
            persist_registry(target)
        result = deepcopy(pending['result'])
        record = dict(ordinal=ordinal, result=result, active_hash=target.registry_hash,
                      prefix_hash=prefix_hash, input_path=str(self.path.parent / f'{ordinal:04d}.json'))
        audit = dict(configuration=self.configuration, binding=self.data['binding'], **pending)
        record['audit_sha256'] = digest(audit)
        atomic_json(Path(record['input_path']), audit)
        self.data['rounds'].append(record)
        self.data['pending'] = None
        self._save()  # If this fails, the durable prepared record remains recoverable.
        self.last_result = result
        return target


def configure_consolidation_evidence_aware(args, rows, *, execution_contract=None):
    """Read-only contract checks before the runner restores or writes any state."""
    journal = args.out_dir / 'skill_consolidation_evidence_aware' / 'state.json'
    if args.skill_consolidation_implementation != 'evidence_aware':
        if journal.exists() or any('skill_consolidation_evidence_aware' in r for r in rows):
            raise ValueError('Cannot disable evidence_aware consolidation midstream; use a fresh output directory')
        return None
    if args.evolution_mode != 'model' or not args.skill_runtime or args.retry_failed:
        raise ValueError('evidence_aware requires model evolution, Skill Runtime and no --retry-failed')
    runtime = ConsolidationRuntime(args.out_dir, interval=args.skill_consolidation_interval,
                                   max_input_chars=args.skill_consolidation_max_input_chars,
                                   execution_contract=execution_contract)
    from .configuration_resume import recorded_configuration_matches
    if rows and (not journal.exists() or any(not recorded_configuration_matches(
            r.get('skill_consolidation_evidence_aware'), runtime.configuration, r.get('ordinal'),
            runtime.data.get('api_configuration_history', [])) for r in rows)):
        raise ValueError('Missing evidence_aware journal or changed consolidation configuration; use a fresh output directory')
    return runtime


def execution_parameters(args):
    """Freeze normalized behavior flags; persist digests, never credential values."""
    from .skill_consolidation import digest
    values = {}
    for key, value in vars(args).items():
        if key in {
            'agent_safetybench_judge_user_agent',
            'agent_safetybench_judge_fallback_user_agent',
        } and value == '':
            continue  # Empty endpoint headers do not affect execution identity.
        if (args.memory_review_mode == 'off' and key in {
                'memory_review_implementation', 'memory_review_repair_attempts',
                'memory_review_max_read_rounds', 'memory_review_max_seconds',
                'memory_review_max_provider_requests'}):
            continue  # Inactive review controls do not affect execution identity.
        # Extending the case limit is supported, subject to the runner's exact prefix checks.
        if key in {'out_dir', 'resume', 'execute', 'max_cases'}:
            continue
        if 'api_key' in key and not key.endswith('_env'):
            values[key + '_sha256'] = digest(value)
        else:
            values[key] = str(value.resolve()) if isinstance(value, Path) else value
    if args.memory_review_mode != 'off':
        # Record the review implementation in the execution contract.
        values['memory_review_implementation'] = 'standard'
    config = getattr(args, 'runtime_api_config', None)
    if config is not None:
        values['api_config_file_sha256'] = hashlib.sha256(Path(config).read_bytes()).hexdigest()
    return values
