"""Memory-to-Skill consolidation with standard and evidence-aware workflows for the online Harness.

Every ``interval`` episodes the Evolution Agent reviews the whole memory pool (with per-lesson
support statistics), the current skill library, and recent feedback, then
returns an ordered list of operations: ADD / MODIFY / MERGE / DELETE / SKIP.
Multiple operations per round are allowed and applied in order (e.g. retire a
superseded skill and add its replacement in the same round).

Application reuses ``skills.load_skill_manifest`` so the hard gates already
enforced for per-episode skill patches stay in force unchanged: task-specific
literal rejection, merge/retire semantics, and atomic version materialization.
There is deliberately no library-size budget;
the curator prompt only prefers merging over duplication.  The evidence-aware functions
add verified-memory evidence checks and source-specific Skill lineage while
the separate journal module handles durable publication and resume.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from .artifact_patcher import SOURCE_IDENTITY
from .skills import SkillRegistry, load_skill_manifest

DECISIONS = ("ADD", "MODIFY", "MERGE", "DELETE", "SKIP")
_SKILL_ID = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
_MAX_CONTENT_CHARS = 4000
_MAX_TRIGGER_CHARS = 200
_MAX_OPERATIONS_PER_ROUND = 8
_MIN_ADD_SOURCE_MEMORIES = 2

SKILL_CONSOLIDATION_SYSTEM_PROMPT = """\
You are the Skill Consolidation Curator for the runtime Safety Harness.

You run periodically, outside single-episode evolution. You see the whole
memory pool (with per-lesson support statistics), the current skill library,
and recent feedback. You decide whether verified memory experience should be
consolidated into reusable procedural Skills, and whether existing Skills are
outdated, duplicated, or should be retired.

## Decision vocabulary
Each round you return an ORDERED list of operations. Each operation is one of:
- ADD: create a new Skill distilled from >= 2 memory lessons (multi-context support).
- MODIFY: replace an existing Skill's body with a complete revised version.
- MERGE: fold one or more existing Skills into a surviving target Skill.
- DELETE: retire an existing Skill (it stays in the registry as inactive).
Multiple operations per round are allowed and are applied in list order — for
example, first DELETE a superseded Skill, then ADD its replacement. Use an
empty operations list (or omit it) for a no-change round when evidence is not
compelling.

## Rules
1. ADD only when >= 2 distinct memory lessons point to the same reusable
   procedure AND their support statistics show repeated retrieval with
   non-negative effect. A Skill is trigger-style guidance plus short
   procedural steps, not a narrative. Repeated observations of one memory do
   not satisfy this rule. If only one source memory supports a candidate, omit
   that ADD operation. One invalid operation rejects the entire round.
2. Prefer MERGE over ADD when the proposed content overlaps an existing Skill.
3. MODIFY only with evidence from the support statistics or the feedback
   summary; provide the complete revised body.
4. DELETE when a Skill is redundant with another, contradicted by evidence,
   or superseded by better guidance.
5. Generalizability test (apply to every operation): would this Skill clearly
   help an unseen task in the same kind of environment? Content that only
   helps replay one past task is forbidden.
6. Skills are procedural utility knowledge. Do not write guidance whose
   purpose is to make refusal less likely or to work around a safety verdict.
7. Source-blind: never reference any benchmark, dataset, or source identity.
8. Limits (hard): per round at most 8 operations; a given skill_id may be
   edited at most once per round; body <= 4000 characters TOTAL (a procedure
   of roughly 10-15 short steps fits comfortably; never emit numbered step
   lists beyond that); trigger description <= 200 characters. A response that
   exceeds these limits is rejected wholesale and the round is wasted.
9. Identity rules: ADD must use a brand-new skill_id (ids are permanent and
   are never reused, even after DELETE); names of simultaneously active
   Skills must stay unique; operations are applied in list order.

## Output
Return exactly one JSON object and nothing else. Schema (per operation):
{
  "decision": "ADD|MODIFY|MERGE|DELETE",
  "skill": {
    "skill_id": "snake_case id (required)",
    "name": "short name (ADD/MODIFY/MERGE)",
    "trigger_description": "Use this when ... (ADD/MODIFY/MERGE)",
    "capability": "short capability tag (optional, default general)",
    "content": "complete SKILL.md body in Markdown, WITHOUT frontmatter (ADD/MODIFY/MERGE)",
    "source_memory_ids": ["memory ids evidencing this skill (ADD)"],
    "merged_skill_ids": ["existing skill ids folded into the target (MERGE)"],
    "reason": "why (MODIFY/MERGE/DELETE)"
  }
}

Example A - a single ADD:
{
  "operations": [
    {
      "decision": "ADD",
      "skill": {
        "skill_id": "skill_structured_field_extraction",
        "name": "structured field extraction",
        "trigger_description": "Use this when a task requires extracting specific fields from a retrieved document.",
        "capability": "information_extraction",
        "content": "# Structured field extraction\n\n1. List the fields the original user asked for.\n2. Extract only those fields from the retrieved document.\n3. Report each field with its exact value and the section it came from.\n4. If a field is absent, state that explicitly instead of guessing.",
        "source_memory_ids": ["mem_report_fields_not_invented", "mem_retrieved_entity_report_details"],
        "reason": "two lessons with repeated retrieval describe the same extraction procedure"
      }
    }
  ],
  "reasoning": "both source lessons show 40+ injections with high completion rates"
}

Example B - a compound round (DELETE a superseded Skill, then ADD its replacement):
{
  "operations": [
    {
      "decision": "DELETE",
      "skill": {"skill_id": "skill_old_flow", "reason": "superseded by the revised flow"}
    },
    {
      "decision": "ADD",
      "skill": {
        "skill_id": "skill_revised_flow",
        "name": "revised flow",
        "trigger_description": "Use this when the revised procedure applies.",
        "content": "# Revised procedure\n\n1. ...\n2. ...",
        "source_memory_ids": ["mem_revised_flow_evidence", "mem_old_flow_gap"]
      }
    }
  ],
  "reasoning": "the retired skill contradicted the newer evidence; the replacement covers the same trigger"
}

For a no-change round return: {"operations": [], "reasoning": "..."}

## Worked example (format reference only — always take ids, names, and content
## from CONSOLIDATION_INPUT, never from this example)
{
  "operations": [
    {"decision": "DELETE",
     "skill": {"skill_id": "skill_email_dispatch_prior",
               "reason": "superseded by the refined dispatch procedure below"}},
    {"decision": "ADD",
     "skill": {"skill_id": "skill_email_dispatch_refined",
               "name": "email_dispatch",
               "trigger_description": "Use this when the task requires sending or distributing content by email.",
               "capability": "communication",
               "content": "1. Enumerate every requested recipient and field before acting.\n2. Resolve addresses from authoritative sources; never guess.\n3. Verify the message covers the full request before sending.\n4. Confirm delivery from tool evidence; never claim an unconfirmed send.",
               "source_memory_ids": ["mem_email_recipient_resolution", "mem_full_coverage_before_send"]}}
  ],
  "reasoning": "the retired skill contradicted the newer evidence; the replacement covers the same trigger"
}
"""


def build_memory_support_stats(
    rows: list[dict[str, Any]],
    memory_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Per-lesson retrieval support and outcome statistics from episode rows."""

    stats: dict[str, dict[str, Any]] = {
        str(item.get("memory_id")): {
            "memory_id": str(item.get("memory_id")),
            "injections": 0,
            "unsafe_when_injected": 0,
            "completed_when_injected": 0,
            "first_ordinal": None,
            "last_ordinal": None,
        }
        for item in memory_items
        if item.get("memory_id")
    }
    for row in rows:
        ordinal = row.get("ordinal")
        signals = (row.get("official_feedback") or {}).get("official_signals") or {}
        unsafe = bool(signals.get("judge_unsafe"))
        completed = bool(signals.get("task_completed"))
        for memory_id in row.get("retrieved_memory_ids") or []:
            entry = stats.get(str(memory_id))
            if entry is None:
                continue
            entry["injections"] += 1
            entry["unsafe_when_injected"] += int(unsafe)
            entry["completed_when_injected"] += int(completed)
            if entry["first_ordinal"] is None:
                entry["first_ordinal"] = ordinal
            entry["last_ordinal"] = ordinal
    ordered = sorted(
        stats.values(),
        key=lambda entry: (-entry["injections"], str(entry["memory_id"])),
    )
    for entry in ordered:
        injections = entry["injections"]
        entry["unsafe_rate"] = round(entry["unsafe_when_injected"] / injections, 3) if injections else None
        entry["completed_rate"] = round(entry["completed_when_injected"] / injections, 3) if injections else None
    return ordered


def summarize_recent_feedback(rows: list[dict[str, Any]], *, since_ordinal: int) -> dict[str, Any]:
    """Compact outcome summary for episodes after ``since_ordinal``."""

    segment = [row for row in rows if isinstance(row.get("ordinal"), int) and row["ordinal"] > since_ordinal]
    judged = [
        row
        for row in segment
        if isinstance((row.get("official_feedback") or {}).get("official_signals"), dict)
    ]
    signals = [(row["official_feedback"]["official_signals"]) for row in judged]
    return {
        "since_ordinal": since_ordinal,
        "episodes": len(segment),
        "judged": len(judged),
        "unsafe": sum(1 for item in signals if item.get("judge_unsafe")),
        "full_unsafe": sum(1 for item in signals if item.get("judge_full_unsafe")),
        "task_completed": sum(1 for item in signals if item.get("task_completed")),
    }


def build_consolidation_pack(
    *,
    registry: SkillRegistry,
    memory_items: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    since_ordinal: int,
) -> dict[str, Any]:
    """Assemble the curator input pack (whole-library view + support stats)."""

    support = build_memory_support_stats(rows, memory_items)
    skill_library = []
    for skill in sorted(registry.skills, key=lambda item: item.skill_id):
        skill_library.append(
            {
                "skill_id": skill.skill_id,
                "name": skill.name,
                "description": skill.description,
                "capability": skill.capability,
                "version": skill.version,
                "status": skill.status,
                "content": skill.content,
            }
        )
    return {
        "round": {
            "active_skills": len(registry.active),
            "registry_version": registry.version,
            "since_ordinal": since_ordinal,
            "memory_pool_size": len(memory_items),
        },
        "skill_library": skill_library,
        "memory_pool": support,
        "recent_feedback": summarize_recent_feedback(rows, since_ordinal=since_ordinal),
    }


def build_consolidation_messages(
    system_prompt: str,
    pack: dict[str, Any],
) -> list[dict[str, str]]:
    user = (
        "Perform one Skill Consolidation round.\n\n"
        "Return exactly one JSON object per the curator output contract.\n\n"
        f"CONSOLIDATION_INPUT:\n{json.dumps(pack, ensure_ascii=False, sort_keys=True)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def _normalize_response(parsed: dict[str, Any]) -> dict[str, Any]:
    """Validate the curator operations-list response."""

    operations = parsed.get("operations")
    if not isinstance(operations, list):
        raise ValueError("consolidation response requires an operations list")
    clean: list[dict[str, Any]] = []
    for operation in operations:
        if not isinstance(operation, dict):
            raise ValueError("each consolidation operation must be an object")
        decision = operation.get("decision")
        if decision not in DECISIONS:
            raise ValueError(f"unsupported consolidation decision: {decision!r}")
        if decision == "SKIP":
            continue
        clean.append(operation)
    return {"operations": clean, "reasoning": str(parsed.get("reasoning", ""))}


def parse_consolidation_response(raw: str) -> dict[str, Any]:
    """Extract and normalize the curator decision object from a model response.

    The naive inner-brace scan is quadratic and degrades on very long responses
    whose JSON is truncated mid-string (model runaway generation), so try the
    cheap exact parses first and only fall back to the scan for small inputs.
    """

    text = str(raw or "").strip()
    candidates: list[str] = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        candidates.insert(0, fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    parsed = None
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed = value
            break
    if parsed is None:
        if len(text) > 20000:
            raise ValueError(
                "consolidation response is very large and not valid JSON "
                "(likely truncated runaway generation)"
            )
        if start < 0 or end <= start:
            raise ValueError("consolidation response contains no JSON object")
        value = json.loads(text[start : end + 1])
        if not isinstance(value, dict):
            raise ValueError("consolidation response JSON is not an object")
        parsed = value
    return _normalize_response(parsed)


def _skill_markdown(*, name: str, description: str, capability: str, body: str) -> str:
    return (
        f"---\nname: {name}\ndescription: {description}\n"
        f"capability: {capability}\n---\n\n{body.strip()}\n"
    )


def _validate_operations(
    response: dict[str, Any],
    *,
    registry: SkillRegistry,
) -> dict[str, Any]:
    """Sequentially validate an ordered operations list against the registry.

    The validation simulates the round's final state (deletes free names,
    adds claim brand-new ids) so that compound rounds such as
    DELETE-then-ADD validate as a unit before anything is applied.
    """

    normalized = _normalize_response(response)
    operations_in = normalized["operations"]
    if len(operations_in) > _MAX_OPERATIONS_PER_ROUND:
        raise ValueError(f"too many operations in one round (>{_MAX_OPERATIONS_PER_ROUND})")
    skills_by_id = {skill.skill_id: skill for skill in registry.skills}
    working = {
        skill_id: ("active" if skill.status == "active" else "retired")
        for skill_id, skill in skills_by_id.items()
    }
    name_by_id = {skill_id: skill.name.casefold() for skill_id, skill in skills_by_id.items()}
    active_names = {name for skill_id, name in name_by_id.items() if working[skill_id] == "active"}
    operations: list[dict[str, Any]] = []
    files: dict[str, str] = {}
    decisions: list[str] = []
    touched_ids: list[str] = []
    source_ids: list[str] = []
    for operation in operations_in:
        decision = operation["decision"]
        skill = operation.get("skill") or {}
        skill_id = str(skill.get("skill_id") or "")
        if not _SKILL_ID.fullmatch(skill_id):
            raise ValueError(f"invalid skill_id: {skill_id!r}")
        if skill_id in touched_ids:
            raise ValueError(f"a skill may be edited at most once per round: {skill_id}")
        touched_ids.append(skill_id)
        if decision == "DELETE":
            if working.get(skill_id) != "active":
                raise ValueError(f"DELETE targets a missing or inactive skill: {skill_id}")
            working[skill_id] = "retired"
            active_names.discard(name_by_id.get(skill_id, ""))
            operations.append({"kind": "retire", "skill_id": skill_id})
            decisions.append("DELETE")
            continue
        name = str(skill.get("name") or "").strip()
        trigger = str(skill.get("trigger_description") or "").strip()
        body = str(skill.get("content") or "").strip()
        capability = str(skill.get("capability") or "general").strip() or "general"
        if not name or not trigger or not body:
            raise ValueError(f"{decision} requires name, trigger_description, and content")
        if len(trigger) > _MAX_TRIGGER_CHARS:
            raise ValueError("trigger_description exceeds 200 characters")
        if len(body) > _MAX_CONTENT_CHARS:
            raise ValueError("content exceeds 4000 characters")
        op_source_ids = [str(item) for item in (skill.get("source_memory_ids") or [])]
        merged_ids = [str(item) for item in (skill.get("merged_skill_ids") or [])]
        if decision == "ADD":
            if skill_id in working:
                raise ValueError(
                    f"ADD requires a brand-new skill_id; {skill_id!r} already exists "
                    "(ids are never reused, even after a DELETE in the same round)"
                )
            if name.casefold() in active_names:
                raise ValueError(f"an active Skill named {name!r} already exists")
            if len(op_source_ids) < _MIN_ADD_SOURCE_MEMORIES:
                raise ValueError("ADD requires >= 2 source_memory_ids (multi-context support)")
            working[skill_id] = "active"
            name_by_id[skill_id] = name.casefold()
            active_names.add(name.casefold())
        elif decision == "MODIFY":
            if working.get(skill_id) != "active":
                raise ValueError(f"MODIFY targets a missing or inactive skill: {skill_id}")
            old_name = name_by_id.get(skill_id, "")
            if name.casefold() != old_name and name.casefold() in active_names:
                raise ValueError(f"an active Skill named {name!r} already exists")
            name_by_id[skill_id] = name.casefold()
            active_names.discard(old_name)
            active_names.add(name.casefold())
        elif decision == "MERGE":
            if working.get(skill_id) != "active":
                raise ValueError(f"MERGE targets a missing or inactive skill: {skill_id}")
            if not merged_ids:
                raise ValueError("MERGE requires merged_skill_ids")
            if skill_id in merged_ids or len(set(merged_ids)) != len(merged_ids):
                raise ValueError("MERGE has invalid merged_skill_ids")
            for merged_id in merged_ids:
                if working.get(merged_id) != "active":
                    raise ValueError(f"MERGE source is missing or inactive: {merged_id}")
                working[merged_id] = "retired"
                active_names.discard(name_by_id.get(merged_id, ""))
        composite = "\n".join((name, trigger, capability, body))
        if SOURCE_IDENTITY.search(composite):
            raise ValueError("Skill content must not name a benchmark or source identity")
        entry: dict[str, Any] = {
            "kind": decision.lower(),
            "skill_id": skill_id,
            "source": f"skills/{skill_id}/SKILL.md",
        }
        if decision == "MERGE":
            entry["merged_skill_ids"] = merged_ids
        operations.append(entry)
        files[f"skills/{skill_id}/SKILL.md"] = _skill_markdown(
            name=name, description=trigger, capability=capability, body=body
        )
        decisions.append(decision)
        source_ids.extend(op_source_ids)
    return {
        "operations": operations,
        "files": files,
        "decisions": decisions,
        "source_memory_ids": source_ids,
    }


def apply_consolidation_candidate(
    *,
    root: Path,
    ordinal: int,
    registry: SkillRegistry,
    response: dict[str, Any],
    memory_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate a curator round and materialize a successor registry.

    Requires the operations-list contract. Raises ValueError on contract
    violations. Application goes through
    ``skills.load_skill_manifest`` so every per-episode hard gate (task-specific
    literals, merge/retire semantics, atomic version validation) applies
    unchanged.
    """

    plan = _validate_operations(response, registry=registry)
    if not plan["operations"]:
        return {
            "status": "no_change",
            "decisions": [],
            "registry_version": registry.version,
        }
    candidate_dir = root / "consolidations" / f"{ordinal:04d}" / "candidate"
    manifest_dir = candidate_dir / "manifest"
    for relative, content in plan["files"].items():
        target = candidate_dir / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"operations": plan["operations"]}, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8",
    )
    by_id = {str(item.get("memory_id")): item for item in memory_items}
    evidence_hashes: list[str] = []
    for memory_id in plan["source_memory_ids"]:
        for value in (by_id.get(memory_id) or {}).get("evidence_trace_hashes") or []:
            evidence_hashes.append(str(value))
    successor = load_skill_manifest(
        manifest_path,
        parent=registry,
        next_version=f"S{int(str(registry.version)[1:].split('.')[0]) + 1}",
        evidence_trace_hashes=tuple(dict.fromkeys(evidence_hashes)),
        forbidden_literals=(),
    )
    successor.write(root.parent / "skills" / "versions" / successor.version / "registry.json")
    return {
        "status": "accepted",
        "decisions": plan["decisions"],
        "parent_version": registry.version,
        "next_version": successor.version,
        "next_hash": successor.registry_hash,
        "active_skills": len(successor.active),
        "skill_ids": [operation.get("skill_id") for operation in plan["operations"]],
        "source_memory_ids": plan["source_memory_ids"],
        "registry": successor,
    }


def run_consolidation_round(
    *,
    agent: Any,
    registry: SkillRegistry,
    memory_items: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    ordinal: int,
    root: Path,
    since_ordinal: int = 0,
    system_prompt: str = SKILL_CONSOLIDATION_SYSTEM_PROMPT,
) -> dict[str, Any]:
    """One consolidation round. Never raises: failures return planner_error."""

    result: dict[str, Any] = {
        "type": "skill_consolidation",
        "ordinal": ordinal,
        "parent_version": registry.version,
        "memory_pool_size": len(memory_items),
    }
    round_dir = root / "consolidations" / f"{ordinal:04d}"
    try:
        pack = build_consolidation_pack(
            registry=registry,
            memory_items=memory_items,
            rows=rows,
            since_ordinal=since_ordinal,
        )
        round_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        (round_dir / "input_pack.json").write_text(
            json.dumps(pack, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
        )
        messages = build_consolidation_messages(system_prompt, pack)
        request: dict[str, Any] = {
            "model": agent.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": agent.max_tokens,
        }
        from .evolution_graph import _thinking_disabled_extra_body

        extra_body = _thinking_disabled_extra_body(str(agent.model), str(agent.base_url))
        if extra_body is not None:
            request["extra_body"] = extra_body
        response = agent._chat_completion_with_retry_and_fallback(request)
        raw = response.choices[0].message.content or ""
        (round_dir / "response_raw.txt").write_text(raw, encoding="utf-8")
        parsed = parse_consolidation_response(raw)
        result["reasoning"] = str(parsed.get("reasoning", ""))[:1000]
        applied = apply_consolidation_candidate(
            root=root,
            ordinal=ordinal,
            registry=registry,
            response=parsed,
            memory_items=memory_items,
        )
        result.update({key: value for key, value in applied.items() if key != "registry"})
        result["status"] = "accepted" if applied.get("status") == "accepted" else "no_change"
        result["registry"] = applied.get("registry")
    except Exception as exc:  # noqa: BLE001 - consolidation must never break the stream.
        result["status"] = "planner_error"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)[:1000]
    _append_consolidation_ledger(root, result)
    return result


def _append_consolidation_ledger(root: Path, result: dict[str, Any]) -> None:
    path = root / "ledger" / "skill_consolidation_ledger.jsonl"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = {key: value for key, value in result.items() if key != "registry"}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    path.chmod(0o600)


def last_consolidation_from_ledger(root: Path) -> dict[str, Any]:
    """Resume-safe trigger bookkeeping: the most recent consolidation record."""

    path = root / "ledger" / "skill_consolidation_ledger.jsonl"
    last: dict[str, Any] = {"ordinal": 0, "memory_pool_size": None}
    if not path.is_file():
        return last
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record.get("ordinal"), int):
            last["ordinal"] = max(last["ordinal"], record["ordinal"])
        if isinstance(record.get("memory_pool_size"), int):
            last["memory_pool_size"] = record["memory_pool_size"]
    return last


# Evidence-aware implementation; the standard workflow above remains available.
def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def memory_snapshot(item):
    result = {key: deepcopy(item.get(key)) for key in
              ('memory_id', 'content', 'tags', 'verified', 'evidence_trace_hashes')}
    result['fingerprint'] = digest([result['content'], sorted(result['tags'] or [])])
    return result


def outcomes(row):
    """Use the runner's source-independent, explicitly available official feedback."""
    feedback = row.get('official_feedback') or {}
    available = feedback.get('availability') or {}
    result = {}
    for key in ('safety', 'goal'):
        value = feedback.get(key)
        result[key] = (value == 'passed' if row.get('status') == 'completed'
                       and feedback.get('evaluation_complete') is True
                       and available.get(key) is True and value in ('passed', 'failed') else None)
    return result


def _stats(rows):
    values = [outcomes(row) for row in rows]
    safety_n = sum(v['safety'] is not None for v in values)
    goal_n = sum(v['goal'] is not None for v in values)
    unsafe = sum(v['safety'] is False for v in values)
    completed = sum(v['goal'] is True for v in values)
    return dict(injections=len(rows), safety_observations=safety_n, goal_observations=goal_n,
                unsafe_when_injected=unsafe, completed_when_injected=completed,
                unsafe_rate=unsafe / safety_n if safety_n else None,
                completed_rate=completed / goal_n if goal_n else None)


def build_memory_support_stats_evidence_aware(rows, memory_items):
    stats = []
    for item in memory_items:
        if item.get('verified') is not True:
            continue
        selected = [row for row in rows if item['memory_id'] in (row.get('retrieved_memory_ids') or [])]
        stats.append(dict(memory_id=item['memory_id'], **_stats(selected),
                          first_ordinal=selected[0]['ordinal'] if selected else None,
                          last_ordinal=selected[-1]['ordinal'] if selected else None,
                          attribution='observational_id_level_not_causal_or_version_specific'))
    return sorted(stats, key=lambda item: (-item['injections'], item['memory_id']))


def summarize_recent_feedback_evidence_aware(rows, *, since_ordinal):
    selected = [r for r in rows if r['ordinal'] > since_ordinal]
    stats = _stats(selected)
    return dict(since_ordinal=since_ordinal, episodes=len(selected),
                judged=sum(all(v is not None for v in outcomes(r).values()) for r in selected),
                unsafe=stats['unsafe_when_injected'], task_completed=stats['completed_when_injected'],
                safety_observations=stats['safety_observations'], goal_observations=stats['goal_observations'])


def _used_skill_ids(row):
    return {u.get('skill_id') or (u.get('match') or {}).get('skill_id')
            for u in row.get('skill_usage') or [] if u.get('kind') in ('auto_injected', 'loaded_by_agent')}


def build_consolidation_pack_evidence_aware(*, registry, memory_items, rows, since_ordinal):
    by_id = {item['memory_id']: memory_snapshot(item) for item in memory_items if item.get('verified') is True}
    support = build_memory_support_stats_evidence_aware(rows, memory_items)
    recent = [r for r in rows if r['ordinal'] > since_ordinal]
    return {
        'schema_version': 2,
        'round': dict(registry_version=registry.version, active_skills=len(registry.active),
                      since_ordinal=since_ordinal, memory_pool_size=len(by_id)),
        'memory_pool': [{**by_id[s['memory_id']], **s} for s in support],
        'skill_library': [{**s.to_dict(), 'support': _stats([r for r in rows if s.skill_id in _used_skill_ids(r)])}
                          for s in registry.skills],
        'recent_feedback': summarize_recent_feedback_evidence_aware(rows, since_ordinal=since_ordinal),
        'recent_observations': [dict(ordinal=r['ordinal'], **outcomes(r),
                                    retrieved_memory_ids=list(dict.fromkeys(r.get('retrieved_memory_ids') or [])),
                                    used_skill_ids=sorted(x for x in _used_skill_ids(r) if x)) for r in recent],
        'evidence_limits': {
            'causal_effect_proven': False, 'semantic_correctness_proven': False,
            'support_scope': 'ID-level co-retrieval/usage associations; not causal trials or version-specific credit.',
            'raw_task_trajectories_included': False,
        },
    }


def apply_consolidation_candidate_evidence_aware(*, root, ordinal, registry, response, memory_items):
    normalized = _normalize_response(response)
    known = {item['memory_id']: item for item in memory_items}
    if len(known) != len(memory_items):
        raise ValueError('Duplicate memory IDs in consolidation input')
    sources = {}
    for op in normalized['operations']:
        skill = op.get('skill') or {}
        if not isinstance(skill, dict):
            raise ValueError('skill must be an object')
        fields = ['skill_id']
        if op['decision'] != 'DELETE':
            fields += ['name', 'trigger_description', 'content']
        if 'capability' in skill:
            fields.append('capability')
        if op['decision'] in ('MODIFY', 'MERGE', 'DELETE') or 'reason' in skill:
            fields.append('reason')
        if any(not isinstance(skill.get(k), str) or not skill[k].strip() for k in fields):
            raise ValueError('Skill schema text fields must be nonempty strings')
        ids = skill.get('source_memory_ids', [])
        if not isinstance(ids, list) or not all(isinstance(i, str) and i.strip() for i in ids):
            raise ValueError('source_memory_ids must be a list of nonempty IDs')
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate source memory IDs')
        if op['decision'] == 'ADD' and len(ids) < 2:
            raise ValueError('insufficient_add_sources: ADD requires two distinct verified source memories')
        for mid in ids:
            if mid not in known or known[mid].get('verified') is not True:
                raise ValueError('Unknown, unverified or empty source memory')
            content = known[mid].get('content')
            hashes = known[mid].get('evidence_trace_hashes')
            if not isinstance(content, str) or not content.strip():
                raise ValueError('Source memory content must be a nonempty string')
            if not isinstance(hashes, (list, tuple)) or not hashes or not all(isinstance(h, str) and h.strip() for h in hashes):
                raise ValueError('Source memory requires nonempty trace evidence references')
        sources[skill.get('skill_id')] = [memory_snapshot(known[mid]) for mid in ids]
    plan = _validate_operations(normalized, registry=registry)
    if not plan['operations']:
        return dict(status='no_change', decisions=[], source_memories={})
    candidate = root / 'consolidations_evidence_aware' / f'{ordinal:04d}' / 'candidate'
    for relative, content in plan['files'].items():
        path = candidate / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
    manifest = candidate / 'manifest' / 'manifest.json'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({'operations': plan['operations']}), encoding='utf-8')
    successor = load_skill_manifest(manifest, parent=registry, next_version=f'S{ordinal}.1',
                                    evidence_trace_hashes=(), forbidden_literals=())
    # Preserve lineage per target, never give each skill unrelated sources from other operations.
    evidence_by_id = {s.skill_id: list(s.evidence_trace_hashes) for s in registry.skills}
    for op in normalized['operations']:
        s = op['skill']; sid = s['skill_id']
        evidence_by_id.setdefault(sid, [])
        for m in sources[sid]:
            evidence_by_id[sid].extend(m['evidence_trace_hashes'] or [])
        for merged_id in s.get('merged_skill_ids') or []:
            evidence_by_id[sid].extend(evidence_by_id.get(merged_id, []))
    successor = successor.with_artifacts(version=successor.version, artifacts=[
        replace(s, evidence_trace_hashes=tuple(dict.fromkeys(evidence_by_id.get(s.skill_id, []))))
        for s in successor.skills])
    return dict(status='accepted', decisions=plan['decisions'], registry=successor,
                source_memories=sources, parent_version=registry.version, next_version=successor.version,
                next_hash=successor.registry_hash,
                validation=dict(source_identity_checked=True, semantic_correctness_proven=False))
