"""Public runner configuration must come from a selected local JSON file."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_safecoevo.py"
SERVER = ROOT / "scripts/serve_agentdog.py"
LAUNCHER = ROOT / "scripts/start_agentdog_server.py"
sys.path.insert(0, str(ROOT / "src"))

from runtime_api_config import require_runtime_api_config  # noqa: E402
import benchmark_runtime  # noqa: E402


def _configuration() -> dict[str, object]:
    def endpoint(name: str) -> dict[str, str]:
        return {"model": name, "base_url": "http://127.0.0.1:1/v1", "api_key": f"test-{name}"}

    return {
        "schema_version": 1,
        "target": {"primary": endpoint("target"), "fallback": endpoint("target-backup")},
        "evolution": {"primary": endpoint("evolution"), "fallback": endpoint("evolution-backup")},
        "judge": {"primary": endpoint("judge"), "fallback": endpoint("judge-backup")},
        "guard": endpoint("guard"),
    }


class PublicConfigurationTest(unittest.TestCase):
    def test_runner_requires_explicit_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(RUNNER), "--out-dir", str(Path(directory) / "output")],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--dataset", result.stderr)

    def test_required_config_loads_guard_identity_from_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "runtime.json"
            config.write_text(json.dumps(_configuration()), encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(require_runtime_api_config(config), config)
                self.assertEqual(os.environ["AGENTDOG_MODEL"], "guard")
                self.assertEqual(os.environ["AGENTDOG_API_KEY"], "test-guard")

    def test_required_config_rejects_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "runtime API config file does not exist"):
                require_runtime_api_config(Path(directory) / "missing.json")

    def test_asb_embedding_does_not_use_inherited_openai_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "stale", "OPENAI_BASE_URL": "http://stale"}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "embedding.*runtime API config"):
                    benchmark_runtime.resolve_asb_embedding_settings(Path(directory) / "missing.json")

    def _run(self, config: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(RUNNER), "--runtime-api-config", str(config),
             "--dataset", str(config.parent / "absent-dataset"),
             "--out-dir", str(config.parent / "output"), "--execute", *extra],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )

    def test_execute_rejects_missing_json_before_dataset_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self._run(Path(directory) / "missing.json")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("runtime API config file does not exist", result.stderr)

    def test_explicit_missing_json_is_rejected_in_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            result = subprocess.run(
                [sys.executable, str(RUNNER), "--runtime-api-config", str(missing),
                 "--dataset", str(missing.parent / "absent-dataset"),
                 "--out-dir", str(missing.parent / "output")],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("runtime API config file does not exist", result.stderr)

    def test_execute_rejects_inline_endpoint_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "runtime.json"
            config.write_text(json.dumps(_configuration()), encoding="utf-8")
            result = self._run(config, "--model", "unreviewed-target")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unrecognized arguments: --model unreviewed-target", result.stderr)

    def test_minimal_stream_dry_run_writes_plan_without_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "stream"
            dataset.mkdir()
            (dataset / "agent_view.json").write_text(json.dumps([{
                "task_id": "sample-1", "system": {"role": "system", "content": "Be helpful."},
                "messages": [{"role": "user", "content": "Hello"}], "tools": [],
                "runtime": {"max_turns": 1},
            }]), encoding="utf-8")
            (dataset / "evaluator_view.json").write_text(json.dumps([{
                "task_id": "sample-1",
                "provenance": {"source_benchmark": "asb", "source_case_key": "sample-1"},
                "evaluation": {"source_adapter": "asb"}, "tool_binding": {},
            }]), encoding="utf-8")
            (dataset / "records.jsonl").write_text("", encoding="utf-8")
            output = root / "output"
            inherited = dict(os.environ)
            inherited.update({
                "SAFECOEVO_TARGET_MODEL": "stale-target",
                "SAFECOEVO_TARGET_BASE_URL": "http://stale-target",
                "AGENTDOG_MODEL": "stale-guard",
                "AGENTDOG_BASE_URL": "http://stale-guard",
            })
            result = subprocess.run(
                [sys.executable, str(RUNNER), "--dataset", str(dataset),
                 "--out-dir", str(output), "--max-cases", "1"],
                cwd=ROOT, env=inherited, text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads((output / "run_plan.json").read_text(encoding="utf-8"))
            self.assertFalse(plan["execute"])
            self.assertEqual(len(plan["selected_cases"]), 1)
            self.assertEqual(plan["target"]["model"], "")
            self.assertEqual(plan["guard"]["model"], "")

            config = root / "runtime.json"
            config.write_text(json.dumps(_configuration()), encoding="utf-8")
            configured_output = root / "configured-output"
            configured = subprocess.run(
                [sys.executable, str(RUNNER), "--runtime-api-config", str(config),
                 "--dataset", str(dataset), "--out-dir", str(configured_output), "--max-cases", "1"],
                cwd=ROOT, env=inherited, text=True, capture_output=True, check=False,
            )
            self.assertEqual(configured.returncode, 0, configured.stderr)
            plan_text = (configured_output / "run_plan.json").read_text(encoding="utf-8")
            configured_plan = json.loads(plan_text)
            self.assertEqual(configured_plan["target"]["model"], "target")
            self.assertEqual(configured_plan["guard"]["model"], "guard")
            self.assertNotIn("test-target", plan_text)
            self.assertNotIn("stale-target", plan_text)

    def test_guard_server_rejects_missing_json_before_loading_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(SERVER), "--runtime-api-config", str(Path(directory) / "missing.json")],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("runtime API config file does not exist", result.stderr)

    def test_guard_launcher_rejects_missing_json_before_gpu_probe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(LAUNCHER), "--runtime-api-config", str(Path(directory) / "missing.json"), "--dry-run"],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("runtime API config file does not exist", result.stderr)


if __name__ == "__main__":
    unittest.main()
