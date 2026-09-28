"""Opt-in bounded reviews. Archived model responses are proposals, not commits."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from .memory_review import apply_revisions, memory_snapshot, _resolve_pointer
from .memory_review_evidence import EvidencePack
from .memory_review_protocol import validate_memory_decision, ReviewValidationError
from .prompt_templates import load_prompt_template


def system_prompt():
    return load_prompt_template('memory_review/system_prompt.md')


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def messages_for(payload):
    return [{'role': 'system', 'content': system_prompt()},
            {'role': 'user', 'content': encoded(payload)}]


def input_chars(payload):
    """Measure serialized messages, including escaped content and repair feedback."""
    return len(encoded(messages_for(payload)))


def add_memory_review_arguments(parser):
    parser.add_argument('--memory-review-mode', choices=('off', 'audit', 'apply'), default='off',
                        help='Independent post-episode memory review')
    parser.add_argument('--memory-review-max-input-chars', type=int, default=250000)
    parser.add_argument('--memory-review-repair-attempts', type=int, default=2)
    parser.add_argument('--memory-review-max-read-rounds', type=int, default=4)
    parser.add_argument('--memory-review-max-seconds', type=float, default=600)
    parser.add_argument('--memory-review-max-provider-requests', type=int, default=12)


def review_episode_id(row):
    return json.dumps([row['ordinal'], row['task_id']], ensure_ascii=False)


def check_review_resume(configuration, rows):
    for row in rows:
        event = row.get('memory_review')
        if configuration['mode'] == 'off':
            if event is not None:
                raise ValueError('Cannot disable memory review midstream; use a new output directory')
        elif not isinstance(event, dict) or event.get('episode_id') != review_episode_id(row):
            raise ValueError('Memory review configuration/episode mismatch; use a new output directory')
        elif event.get('configuration') != configuration:
            raise ValueError('Memory review configuration/episode mismatch; use a new output directory')


def attach_memory_review(runtime, row, *, selected, store, rows, case_dir,
                         case=None, session=None, feedback=None, trace_records=None):
    if runtime.mode == 'off':
        return
    episode_id = review_episode_id(row)
    if row.get('status') == 'failed' or feedback is None:
        row['memory_review'] = runtime.skip(episode_id, 'runtime_failure')
        return
    try:
        from .online_evolution import build_online_evidence
        evidence = build_online_evidence(case=case, session=session, feedback=feedback,
                                         trace_records=trace_records or [])
        available = (feedback.evaluation_complete is True
                     and feedback.availability.get('safety') is True
                     and feedback.availability.get('goal') is True)
        row['memory_review'] = runtime.finish(
            episode_id=episode_id, selected=selected, evidence=evidence, store=store,
            rows=rows, case_dir=case_dir, evaluation_available=available)
    except Exception as exc:
        row['memory_review'] = {**runtime.skip(episode_id, 'review_processing_failed'),
                                'status': 'review_error', 'error_type': type(exc).__name__}


def _write_record(path, record):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    text = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n', dir=path.parent,
                                     prefix='.review-', suffix='.tmp', delete=False) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(handle.name, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def committed_history(rows, selected, *, max_chars=40000):
    """Only committed, integrity-checked source observations; no reviewer essays."""
    wanted = {m.memory_id for m in selected}
    history, seen = [], set()
    for row in reversed(rows):
        event = row.get('memory_review') or {}
        if row.get('task_id') in seen or not event.get('record_path'):
            continue
        try:
            raw = Path(event['record_path']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != event.get('record_sha256'):
                continue
            payload = json.loads(raw)['payload']
            memories = [m for m in payload['selected_memories'] if m['memory_id'] in wanted]
            if not memories:
                continue
            ev = payload['evidence']
            observations = []
            for d in event.get('decisions', []):
                if d.get('memory_id') not in wanted:
                    continue
                for ref in d.get('evidence', []):
                    value = _resolve_pointer(ev, ref['pointer'])
                    if isinstance(value, str) and ref['quote'] in value:
                        observations.append(deepcopy(ref))
            item = {'episode_id': event['episode_id'], 'selected_memories': memories,
                    'official_feedback': ev.get('official_feedback', {}),
                    'observations': observations,
                    'context': 'Selected exact observations, not the full historical trajectory or independent causal trials.'}
            if len(encoded([*history, item])) > max_chars:
                continue
            history.append(item)
            seen.add(row['task_id'])
            if len(history) == 6:
                break
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return list(reversed(history))


class MemoryReviewRuntime:
    def __init__(self, mode='off', *, reviewer=None, max_input_chars=250000,
                 repair_attempts=2, max_read_rounds=4, max_seconds=600,
                 max_provider_requests=12, execution_identity=None):
        if mode not in {'off', 'audit', 'apply'} or max_input_chars < 1:
            raise ValueError('Invalid memory review mode/input budget')
        self.mode = mode
        self.reviewer = reviewer
        self.max_input_chars = max_input_chars
        if (isinstance(repair_attempts, bool) or not isinstance(repair_attempts, int)
                or not 0 <= repair_attempts <= 10 or isinstance(max_read_rounds, bool)
                or not isinstance(max_read_rounds, int) or not 0 <= max_read_rounds <= 32
                or not math.isfinite(max_seconds) or max_seconds <= 0
                or not isinstance(max_provider_requests, int) or max_provider_requests < 1):
            raise ValueError('Invalid memory review repair/read/request/time budget')
        self.repair_attempts = repair_attempts
        self.max_read_rounds = max_read_rounds
        self.max_seconds = max_seconds
        self.max_provider_requests = max_provider_requests
        self.execution_identity = deepcopy(execution_identity)

    @property
    def configuration(self):
        configuration = {'schema_version': 2, 'implementation': 'standard', 'mode': self.mode,
                'max_input_chars': self.max_input_chars,
                'prompt_sha256': hashlib.sha256(system_prompt().encode()).hexdigest() if self.mode != 'off' else None,
                'citation_protocol': 'delivered_evidence_ids',
                'repair_attempts': self.repair_attempts, 'max_read_rounds': self.max_read_rounds,
                'max_seconds': self.max_seconds, 'max_provider_requests': self.max_provider_requests,
                'execution_identity': self.execution_identity,
                'history_max_episodes': 6, 'history_max_chars': 40000,
                'evidence_chunk_chars': 4000, 'repair_reserve_chars': min(65536, self.max_input_chars // 4)}
        configuration['citation_feedback'] = 'detailed'
        return configuration

    def skip(self, episode_id, reason):
        return {'episode_id': episode_id, 'configuration': self.configuration,
                'status': 'skipped', 'reason': reason, 'decisions': [], 'changes': [],
                'review_complete': False}

    def _set_citation_context(self, payload, pack, delivered):
        from .memory_review_citations import citation_context
        payload['citation_context'] = citation_context(
            pack, delivered, max_chars=max(1024, min(6000, self.max_input_chars // 16)))

    def _initial_payload(self, pack, selected, rows):
        payload = {'selected_memories': [memory_snapshot(m) for m in selected],
                   'evidence': pack.full_payload(), 'history': [],
                   'delivered_ids': [],
                   'limits': {'remaining_read_rounds': self.max_read_rounds,
                              'remaining_repairs': self.repair_attempts}}
        payload['delivered_ids'] = payload['evidence'].get('delivered_ids', list(pack.entries))
        self._set_citation_context(payload, pack, set(payload['delivered_ids']))
        cap = self.max_input_chars - self.configuration['repair_reserve_chars']
        full = input_chars(payload) <= cap
        if not full:
            # Keep space for exact evidence reads, metadata pages and error repair.
            payload['evidence'] = pack.overview_payload(max_chars=max(128, min(24000, cap // 3)))
            payload['delivered_ids'] = payload['evidence'].get('delivered_ids', [])
            self._set_citation_context(payload, pack, set(payload['delivered_ids']))
        if input_chars(payload) > cap:
            raise ValueError('input_budget_exceeded_after_dedup')
        for item in reversed(committed_history(rows, selected)):
            candidate = deepcopy(payload)
            candidate['history'].insert(0, item)
            if input_chars(candidate) <= cap:
                payload = candidate
        return payload, full

    def _fit(self, payload):
        """Never shorten source evidence silently. Repair text is explicitly marked."""
        directory = payload.get('citation_context')
        while directory and directory['entries'] and input_chars(payload) > self.max_input_chars:
            directory['entries'].pop()
            directory['omitted'] += 1
        while payload.get('history') and input_chars(payload) > self.max_input_chars:
            payload['history'].pop(0)
        repair = payload.get('repair')
        if repair and input_chars(payload) > self.max_input_chars:
            raw = encoded(repair.get('previous_response'))
            repair['previous_response_truncated'] = True
            # The exact output remains archived. Only a labeled repair preview shrinks.
            repair['previous_response'] = ''
            room = max(0, (self.max_input_chars - input_chars(payload)) // 2)
            repair['previous_response'] = raw[:room]
        if repair and repair.get('errors') and input_chars(payload) > self.max_input_chars:
            from .memory_review_citations import fit_repair_errors
            # Full diagnostics remain in attempt.validation_errors. Only this
            # request's preview shrinks, retaining a first error for each memory.
            original_errors = repair['errors']
            base = deepcopy(payload)
            base['repair']['errors'] = []
            budget = max(1, (self.max_input_chars - input_chars(base) - 512) // 2)
            while budget > 0:
                try:
                    repair.update(fit_repair_errors(original_errors, max_chars=budget))
                except ValueError:
                    return False
                if input_chars(payload) <= self.max_input_chars:
                    break
                budget //= 2
        return input_chars(payload) <= self.max_input_chars

    def finish(self, *, episode_id, selected, evidence, store, rows, case_dir, evaluation_available):
        if self.mode == 'off' or not selected or not evaluation_available:
            return self.skip(episode_id, 'disabled' if self.mode == 'off' else
                             'no_selected_memory' if not selected else 'feedback_unavailable')
        review_dir = Path(case_dir).resolve() / 'memory_review'
        path = review_dir / 'review.json'
        identity = digest({'configuration': self.configuration, 'episode_id': episode_id,
                           'selected_memories': [memory_snapshot(m) for m in selected], 'evidence': evidence})
        record = {'schema_version': 2, 'identity': identity, 'configuration': self.configuration,
                  'episode_id': episode_id, 'system_prompt': system_prompt(),
                  'payload': {'evidence': deepcopy(evidence),
                              'selected_memories': [memory_snapshot(m) for m in selected]},
                  'case_commit_required': True, 'attempts': [], 'provider_reservations': [],
                  'deadline_unix': time.time() + self.max_seconds,
                  'validation': {'citation_integrity_passed': False, 'semantic_correctness_proven': False}}
        try:
            if path.exists():
                record = json.loads(path.read_text())
                if record.get('identity') != identity:
                    raise ValueError('Saved memory review input/configuration drift')
                if 'result' in record:
                    event = deepcopy(record['result'])
                    # Journal alone does not authorize replay into a different live version.
                    if event['status'] == 'applied':
                        live = {m.memory_id: memory_snapshot(m) for m in store.items}
                        pending_ids = set()
                        for change in event['changes']:
                            ident = change['before']['memory_id']
                            if digest(live.get(ident)) == digest(change['after']):
                                continue
                            if digest(live.get(ident)) != digest(change['before']):
                                raise ValueError('Live memory diverges from saved review')
                            pending_ids.add(ident)
                        apply_revisions(store, selected, [d for d in event['decisions']
                            if d['memory_id'] in pending_ids], tuple(evidence.get('trace_hashes', ())))
                    return {**event, 'record_path': str(path),
                            'record_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            else:
                _write_record(path, record)
        except OSError:
            return {**self.skip(episode_id, 'review_record_write_failed'), 'status': 'review_error'}
        except (ValueError, KeyError, TypeError):
            return {**self.skip(episode_id, 'saved_review_input_or_live_version_mismatch'), 'status': 'review_error'}

        event = {**self.skip(episode_id, 'pending'), 'status': 'rejected', 'unresolved': []}
        started = time.monotonic()
        previous_elapsed = sum(a.get('elapsed_seconds', 0) for a in record['attempts'])
        deadline = started + max(0, record['deadline_unix'] - time.time())
        if self.reviewer is not None and hasattr(self.reviewer, 'begin_episode'):
            requests_used = len(record['provider_reservations'])
            def reserve_request(metadata):
                record['provider_reservations'].append({**metadata, 'reserved_at': time.time()})
                _write_record(path, record)  # An interrupted in-flight request still consumes its slot.
            self.reviewer.begin_episode(deadline, max(0, self.max_provider_requests - requests_used),
                                        reserve_request=reserve_request)
        try:
            pack = EvidencePack(evidence)
            payload, full = self._initial_payload(pack, selected, rows)
            record['packing'] = {'original_chars': len(encoded(evidence)),
                                 'full_pack_chars': len(encoded(pack.full_payload())),
                                 'initial_request_chars': input_chars(payload), 'full_evidence': full}
            event = self._review(record, path, payload, pack, selected, store, evidence, deadline, event)
        except OSError:
            return {**self.skip(episode_id, 'review_record_write_failed'), 'status': 'review_error'}
        except ValueError as exc:
            # Packing and local contract errors only; no provider exception bodies.
            event.update(status='skipped', reason='input_budget_exceeded_after_dedup', error_type=type(exc).__name__)
        except Exception as exc:
            event.update(status='review_error', reason='independent_review_failed', error_type=type(exc).__name__)
        event['elapsed_seconds'] = previous_elapsed + time.monotonic() - started
        record['result'] = deepcopy(event)
        try:
            event['record_sha256'] = _write_record(path, record)
            event['record_path'] = str(path)
        except OSError:
            return {**self.skip(episode_id, 'review_record_write_failed'), 'status': 'review_error'}
        if event['status'] == 'applied':
            apply_revisions(store, selected, event['decisions'], tuple(evidence.get('trace_hashes', ())))
        return event

    def _review(self, record, path, payload, pack, selected, store, evidence, deadline, event):
        pending = {m.memory_id: m for m in selected}
        accepted, errors = {}, []
        delivered = set(payload['delivered_ids'])
        reads = repairs = index = 0
        while pending:
            payload['selected_memories'] = [memory_snapshot(m) for m in pending.values()]
            payload['limits'] = {'remaining_read_rounds': self.max_read_rounds - reads,
                                 'remaining_repairs': self.repair_attempts - repairs}
            payload['delivered_ids'] = sorted(delivered)
            self._set_citation_context(payload, pack, delivered)
            if not self._fit(payload):
                event['reason'] = 'repair_or_read_input_budget_exceeded'
                break
            request_hash = digest(messages_for(payload))
            if index < len(record['attempts']):
                attempt = record['attempts'][index]
                if attempt['request_sha256'] != request_hash:
                    raise RuntimeError('Saved review request drift')
            else:
                if time.monotonic() >= deadline:
                    event['reason'] = 'review_time_budget_exceeded'
                    break
                if self.reviewer is None:
                    raise RuntimeError('Memory review model is not configured')
                begin = time.monotonic()
                attempt = {'request_sha256': request_hash, 'payload': deepcopy(payload)}
                try:
                    attempt['proposal'] = self.reviewer.review(payload)
                except (ValueError, json.JSONDecodeError) as exc:
                    attempt['error'] = {'code': 'invalid_json', 'field': 'response',
                                        'message': 'Return one complete JSON object with reviews or read/catalog_offset.'}
                except Exception as exc:
                    attempt['error'] = {'code': 'provider_error', 'field': 'request',
                                        'message': 'Independent model call failed', 'error_type': type(exc).__name__}
                attempt['raw_response'] = getattr(self.reviewer, 'last_response', None)
                attempt['elapsed_seconds'] = time.monotonic() - begin
                attempt['request_attempts'] = deepcopy(getattr(self.reviewer, 'request_attempts', []))
                record['attempts'].append(attempt)
                _write_record(path, record)  # Durable BEFORE interpretation or mutation.
            index += 1
            proposal = attempt.get('proposal')
            errors = []
            if attempt.get('error'):
                errors = [attempt['error']]
                if errors[0]['code'] == 'provider_error':
                    event['reason'] = 'provider_request_budget_or_failure'
                    break
            elif isinstance(proposal, dict) and set(proposal) in (
                    {'read'}, {'catalog_offset'}, {'event_offset'}, {'structure_offset'}):
                if reads >= self.max_read_rounds:
                    event['reason'] = 'evidence_read_budget_exhausted'
                    break
                reads += 1
                # Keep delivered bodies/structure and repair context together. New
                # reads must fit the shared bound; never discard supporting context.
                previous_payload = deepcopy(payload)
                payload = deepcopy(payload)
                payload['selected_memories'] = [memory_snapshot(m) for m in pending.values()]
                # Drop optional history before admitting requested current evidence.
                payload['history'] = []
                available = max(0, self.max_input_chars - input_chars(payload)
                                - self.configuration['repair_reserve_chars'])
                try:
                    if 'read' in proposal:
                        result = pack.read(proposal['read'], max_chars=max(1, available), max_entries=32)
                        cached = {r['id']: r for r in payload.get('read_results', [])}
                        cached.update({r['id']: r for r in result})
                        payload['read_results'] = list(cached.values())
                        new_ids = {r['id'] for r in result}
                        # IDs count as delivered only once an actual model call receives this request.
                        if input_chars({**payload, 'delivered_ids': sorted(delivered | new_ids)}) > self.max_input_chars:
                            raise ValueError('Requested evidence exceeds available input budget')
                        delivered |= new_ids
                    else:
                        offset = next(iter(proposal.values()))
                        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
                            raise ValueError('catalog_offset must be a nonnegative integer')
                        key, page = ('event_catalog', pack.event_catalog) if 'event_offset' in proposal else (
                            ('structure_catalog', pack.structure_catalog) if 'structure_offset' in proposal else ('catalog', pack.catalog))
                        if key in payload:
                            payload.setdefault('earlier_pages', []).append({'kind': key, 'page': payload.pop(key)})
                        payload[key] = page(offset=offset, limit=64, max_chars=max(1, available))
                        if input_chars(payload) > self.max_input_chars:
                            raise ValueError('Requested metadata page exceeds available input budget')
                    continue
                except (ValueError, KeyError, TypeError) as exc:
                    payload = previous_payload
                    errors = [{'code': 'invalid_evidence_read', 'field': 'read',
                               'message': str(exc)}]
            elif not isinstance(proposal, dict) or set(proposal) != {'reviews'} or not isinstance(proposal['reviews'], list):
                errors = [{'code': 'invalid_envelope', 'field': 'reviews',
                           'message': 'Return exactly reviews:[complete memory objects], read:[IDs], or catalog_offset:integer.'}]
            else:
                grouped = {}
                for d in proposal['reviews']:
                    ident = d.get('memory_id') if isinstance(d, dict) else None
                    if not isinstance(ident, str) or ident not in pending:
                        errors.append({'code': 'unexpected_memory', 'field': 'memory_id',
                                       'message': 'Only return each requested memory once.'})
                        continue
                    grouped.setdefault(ident, []).append(d)
                for ident, memory in list(pending.items()):
                    candidates = grouped.get(ident, [])
                    if len(candidates) != 1:
                        errors.append({'memory_id': ident, 'code': 'missing_or_duplicate_memory',
                                       'field': 'memory_id', 'message': 'Return exactly one full object for this memory.'})
                        continue
                    from .memory_review_citations import citation_issues
                    issues = citation_issues(candidates[0], pack, delivered)
                    if issues:
                        errors.extend(issues)
                        continue
                    try:
                        validated = validate_memory_decision(candidates[0], memory, evidence,
                            resolve_citation=pack.citation, delivered_ids=delivered)
                        # Live versions are checked per memory, then all accepted edits are committed together.
                        stage = SimpleNamespace(items=list(store.items))
                        apply_revisions(stage, [memory], [validated], tuple(evidence.get('trace_hashes', ())))
                        accepted[ident] = validated
                        pending.pop(ident)
                    except ReviewValidationError as exc:
                        errors.append(exc.details)
                    except ValueError:
                        errors.append({'memory_id': ident, 'code': 'stale_live_memory', 'field': 'memory_id',
                                       'message': 'Injected memory version no longer matches live memory; cannot overwrite.'})
                if not pending:
                    attempt['validation_errors'] = deepcopy(errors)
                    break
            attempt['validation_errors'] = deepcopy(errors)
            _write_record(path, record)
            event['unresolved'] = errors
            # Version conflicts need a new episode snapshot, not a model rewrite.
            if any(e.get('code') == 'stale_live_memory' for e in errors):
                event['reason'] = 'stale_live_memory'
                break
            if repairs >= self.repair_attempts:
                event['reason'] = 'validation_retries_exhausted'
                break
            repairs += 1
            previous = proposal if proposal is not None else attempt.get('raw_response')
            if isinstance(previous, dict) and isinstance(previous.get('reviews'), list):
                previous = {'reviews': [d for d in previous['reviews'] if not isinstance(d, dict)
                                       or d.get('memory_id') in pending]}
            payload['repair'] = {'errors': errors, 'previous_response': previous,
                                 'previous_response_truncated': False}
        event['unresolved'] = errors or ([{'memory_id': ident, 'code': event.get('reason', 'incomplete'),
                                           'field': 'review', 'message': 'Memory preserved without a validated decision.'}
                                          for ident in pending] if pending else [])
        decisions = [accepted[m.memory_id] for m in selected if m.memory_id in accepted]
        stage = SimpleNamespace(items=list(store.items))
        changes = apply_revisions(stage, selected, decisions, tuple(evidence.get('trace_hashes', ())))
        record['proposed_changes'] = changes
        record['validation']['citation_integrity_passed'] = bool(decisions)
        event.update(decisions=decisions, changes=changes if self.mode == 'apply' else [],
                     review_calls=index, repair_rounds=repairs, read_rounds=reads,
                     delivered_evidence_count=len(delivered), review_complete=not pending)
        if changes:
            event['status'] = 'audited' if self.mode == 'audit' else 'applied' if changes else 'no_change'
            if not pending:
                event.pop('reason', None)
        elif decisions and not pending:
            event['status'] = 'audited' if self.mode == 'audit' else 'no_change'
            event.pop('reason', None)
        return event


def configure_memory_review(args):
    identity = {key: getattr(args, 'evolution_' + key, None) for key in (
        'model', 'base_url', 'max_tokens', 'timeout_seconds', 'request_retries',
        'retry_sleep_seconds', 'fallback_model', 'fallback_base_url', 'fallback_request_retries')}
    identity['temperature'] = 0.0
    config = getattr(args, 'runtime_api_config', None)
    if config is not None:
        identity['api_config_file_sha256'] = hashlib.sha256(Path(config).read_bytes()).hexdigest()
    return MemoryReviewRuntime(args.memory_review_mode, execution_identity=identity,
        max_input_chars=args.memory_review_max_input_chars,
        repair_attempts=args.memory_review_repair_attempts, max_read_rounds=args.memory_review_max_read_rounds,
        max_seconds=args.memory_review_max_seconds, max_provider_requests=args.memory_review_max_provider_requests)
