"""Composable Processor protocol for the SafeCoEvo runtime.

The public Safety-Harness lifecycle intentionally exposes six fixed Hook
positions. Processors may also use an event-stream
contract: a handler can pass an event through, replace it, emit supplemental
events, or yield nothing to intercept it.
"""

from __future__ import annotations

import inspect
from abc import ABC
from dataclasses import dataclass
from typing import Any, AsyncIterator, FrozenSet

from .contracts import Hook, Resource
from .events import HarnessEvent


@dataclass(frozen=True)
class ProcessorSpec:
    """Static placement and declared resource access for one Processor."""

    name: str
    hooks: FrozenSet[Hook]
    order: int = 50
    reads: FrozenSet[Resource] = frozenset()
    writes: FrozenSet[Resource] = frozenset()
    # Logical processor IDs that must run first when both processors are
    # registered on the same Hook.
    after: FrozenSet[str] = frozenset()


class Processor(ABC):
    """A component in the six-hook Safety Harness event stream.

    Builtin processors may mutate the supplied event synchronously. Generated
    class processors return an async iterator of events. ``HarnessSession``
    supports both contracts.
    """

    spec: ProcessorSpec

    def process(self, event: HarnessEvent) -> Any:
        """Pass an event through by default."""

        return event


class MultiHookProcessor(Processor):
    """Event-stream dispatcher for the six Safety Harness hooks.

    Generated subclasses override one or more ``async def on_<hook>()``
    methods and ``yield`` zero or more :class:`HarnessEvent` values.  The
    base owns ``process`` so source modules do not need to hand-roll a generic
    dispatcher or guess the runtime's event-type logic.
    """

    _VALID_HOOKS = frozenset(hook.value for hook in Hook)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name, value in vars(cls).items():
            if not name.startswith("on_"):
                continue
            hook_name = name.removeprefix("on_")
            if hook_name not in cls._VALID_HOOKS:
                raise TypeError(
                    f"{cls.__qualname__}.{name}: unsupported Safety Harness hook; "
                    f"valid hooks are {sorted(cls._VALID_HOOKS)}"
                )
            if not inspect.isasyncgenfunction(value):
                raise TypeError(
                    f"{cls.__qualname__}.{name}: hook handlers must be async generators that yield events"
                )

    async def process(self, event: HarnessEvent) -> AsyncIterator[HarnessEvent]:
        handler = getattr(self, f"on_{event.hook.value}", None)
        if handler is None:
            yield event
            return
        async for output in handler(event):
            if not isinstance(output, HarnessEvent):
                raise TypeError(
                    f"{type(self).__name__}.on_{event.hook.value} yielded "
                    f"{type(output).__name__}, expected HarnessEvent"
                )
            yield output
