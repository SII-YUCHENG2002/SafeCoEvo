"""Checks for persisted runtime state and artifact loading."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safecoevo_runtime.contracts import Hook, MemoryItem, TaskOutcome
from safecoevo_runtime.artifact_patcher import hydrate_seed_permission_store
from safecoevo_runtime.events import HarnessEvent, HarnessState, PublicTask
from safecoevo_runtime.memory import InMemoryMemoryStore
from safecoevo_runtime.online_feedback import OnlineEvolutionState
from safecoevo_runtime.permission import InMemoryPermissionExperienceStore
from safecoevo_runtime.processors import MemoryWriter
from safecoevo_runtime.skill_consolidation_journal import atomic_json
from safecoevo_runtime.skills import SkillRegistry, load_seed_skill_registry
from safecoevo_runtime.trace import JsonTraceStore, load_trace_records


class RuntimeContractsTest(unittest.TestCase):
    def test_trace_and_journal_persist_on_host_platform(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.json"
            store = JsonTraceStore(trace_path)
            store.append({"hook": "task_start", "task_id": "case"})
            self.assertTrue(JsonTraceStore(trace_path).verify())
            self.assertEqual(len(load_trace_records(trace_path)), 1)

            journal_path = root / "journal.json"
            atomic_json(journal_path, {"complete": True})
            self.assertEqual(json.loads(journal_path.read_text(encoding="utf-8")), {"complete": True})

    def test_only_verified_memory_is_committed_and_checkpointed(self) -> None:
        accepted = MemoryItem("accepted", "verified lesson", (), True)
        rejected = MemoryItem("rejected", "unverified lesson", (), False)
        store = InMemoryMemoryStore()
        task = PublicTask("case", {}, (), (), {})
        state = HarnessState(pending_memory=[accepted, rejected])
        MemoryWriter(store).process(
            HarnessEvent(Hook.TASK_END, task, state, outcome=TaskOutcome("done", verified=True))
        )
        self.assertEqual([item.memory_id for item in store.items], ["accepted"])

        checkpoint = OnlineEvolutionState(memory_store=store).to_dict()
        self.assertEqual(checkpoint["schema_version"], 3)
        self.assertNotIn("confidence", checkpoint["memory_items"][0])
        restored = OnlineEvolutionState.from_dict(checkpoint)
        self.assertEqual(restored.memory_store.items, [accepted])

    def test_memory_without_lexical_match_returns_no_recent_item(self) -> None:
        store = InMemoryMemoryStore(items=[
            MemoryItem("unrelated", "database migration procedure", (), True),
        ])
        self.assertEqual(store.retrieve("calendar scheduling", limit=1), [])

    def test_seed_permission_experience_survives_checkpoint(self) -> None:
        permission_store = InMemoryPermissionExperienceStore()
        count = hydrate_seed_permission_store(ROOT / "artifacts", permission_store)
        self.assertEqual(count, 4)
        self.assertEqual(len(permission_store.retrieve("any task", limit=4)), 4)
        state = OnlineEvolutionState(permission_store=permission_store)
        restored = OnlineEvolutionState.from_dict(state.to_dict())
        self.assertEqual(restored.permission_store.items, permission_store.items)

    def test_seed_skill_registry_has_lexical_schema(self) -> None:
        registry = load_seed_skill_registry(ROOT / "artifacts" / "skills")
        encoded = registry.to_dict()
        self.assertEqual(encoded["schema_version"], 2)
        self.assertEqual(encoded["retrieval_mode"], "lexical")
        self.assertNotIn("min_similarity", encoded)
        self.assertEqual(SkillRegistry.from_dict(encoded).registry_hash, registry.registry_hash)


if __name__ == "__main__":
    unittest.main()
