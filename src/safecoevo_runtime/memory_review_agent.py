"""Tool-free reviewer with an episode-wide physical-request/deadline budget."""
from __future__ import annotations

import os
import time

from .evolution_graph import OpenAICompatibleEvolutionAgent, _thinking_disabled_extra_body
from .structured_output import parse_single_json_object


class EvolutionMemoryReviewer:
    def __init__(self, transport):
        # Transport was constructed from the runner's already loaded JSON config.
        self.transport = transport
        self.last_response = None
        self.request_attempts = []
        self.deadline = 0
        self.remaining_requests = 0
        self.reserve_request = None

    def begin_episode(self, deadline, max_requests, *, reserve_request=None):
        self.deadline = deadline
        self.remaining_requests = max_requests
        self.reserve_request = reserve_request

    def _physical_request(self, endpoint, request, timeout):
        from openai import OpenAI
        from runtime_endpoint_transport import endpoint_transport_options
        with endpoint_transport_options('evolution', endpoint['endpoint'], timeout) as options:
            with OpenAI(base_url=endpoint['base_url'], api_key=endpoint['api_key'],
                        timeout=timeout, max_retries=0, **options) as client:
                return client.chat.completions.create(**request)

    def _complete(self, request):
        agent = self.transport
        endpoints = [dict(endpoint='primary', model=agent.model, base_url=agent.base_url,
                          api_key=os.environ.get(agent.api_key_env), retries=agent.request_retries),
                     dict(endpoint='fallback', model=agent.fallback_model, base_url=agent.fallback_base_url,
                          api_key=agent.fallback_api_key, retries=agent.fallback_request_retries)]
        for endpoint in endpoints:
            if not endpoint['api_key']:
                continue
            for ordinal in range(int(endpoint['retries']) + 1):
                remaining = self.deadline - time.monotonic()
                if remaining <= 0 or self.remaining_requests <= 0:
                    raise RuntimeError('Memory review time or physical request budget exhausted')
                payload = dict(request, model=endpoint['model'])
                extra = _thinking_disabled_extra_body(str(endpoint['model']), str(endpoint['base_url']))
                if extra is not None:
                    payload['extra_body'] = extra
                audit = {'endpoint': endpoint['endpoint'], 'model': endpoint['model'], 'attempt': ordinal + 1}
                if self.reserve_request is not None:
                    self.reserve_request(audit)
                self.remaining_requests -= 1
                try:
                    result = self._physical_request(endpoint, payload, min(float(agent.timeout_seconds), remaining))
                    self.request_attempts.append({**audit, 'ok': True})
                    return result
                except Exception as exc:
                    # Never persist exception bodies, endpoint credentials or request headers.
                    self.request_attempts.append({**audit, 'ok': False, 'error_type': type(exc).__name__})
                if ordinal < int(endpoint['retries']) and agent.retry_sleep_seconds:
                    delay = float(agent.retry_sleep_seconds) * (ordinal + 1)
                    if time.monotonic() + delay >= self.deadline or self.remaining_requests <= 0:
                        raise RuntimeError('Memory review retry budget exhausted')
                    time.sleep(delay)
        raise RuntimeError('Memory review configured providers failed')

    def review(self, payload):
        from .memory_review_runtime import messages_for
        self.last_response = None
        self.request_attempts = []
        response = self._complete({'model': self.transport.model, 'temperature': 0.0,
                                  'max_tokens': self.transport.max_tokens, 'messages': messages_for(payload)})
        choice = response.choices[0]
        message = choice.message
        self.last_response = message.content
        if getattr(message, 'tool_calls', None):
            raise ValueError('Memory Review cannot execute tools')
        if getattr(choice, 'finish_reason', None) == 'length':
            raise ValueError('Incomplete JSON: model output budget reached')
        return parse_single_json_object(message.content or '')


def make_memory_reviewer(args):
    # A separate model instance has no shared conversation or task tools.
    return EvolutionMemoryReviewer(OpenAICompatibleEvolutionAgent(
        model=args.evolution_model, base_url=args.evolution_base_url,
        api_key_env=args.evolution_api_key_env, max_tokens=args.evolution_max_tokens,
        timeout_seconds=args.evolution_timeout_seconds, request_retries=args.evolution_request_retries,
        fallback_model=args.evolution_fallback_model, fallback_base_url=args.evolution_fallback_base_url,
        fallback_api_key=args.evolution_fallback_api_key,
        fallback_request_retries=args.evolution_fallback_request_retries,
        retry_sleep_seconds=args.evolution_retry_sleep_seconds,
    ))
