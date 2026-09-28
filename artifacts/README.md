# SafeCoEvo Seed Artifacts

This directory contains the initial artifacts for the online SafeCoEvo runtime.
The runtime uses five artifact categories: Prompt, Memory, Skill, Permission,
and Guard.

Artifacts are persistent, reusable, versioned, and evolvable capability assets.
Processors decide when and how to read, inject, invoke, or update them.
The Target Agent should see artifact content, not Processor/Hook internals.

## Layout

- `prompt/target_system_addendum.md`: evolvable system addendum appended after the benchmark system prompt.
- `memory/validated_experience.jsonl`: verified reusable memory lessons.
- `skills/registry.json`: active Skill catalog.
- `skills/*/SKILL.md`: full procedural Skill references.
- `permission/permission_experience.jsonl`: high-level authorization lessons.
- `guard/guard_policy.json`: Guard semantics and verdict policy.

Related work and design influences are listed in the [project README](../README.md).
