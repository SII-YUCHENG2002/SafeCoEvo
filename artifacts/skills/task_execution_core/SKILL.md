# Task Execution Core

Use this skill when the task requires multiple tool steps, state changes, or careful progress tracking.

## Procedure

1. Restate the concrete user goal internally, including whether the goal asks for a read-only answer or a real state-changing action.
2. Identify the exact available tools that can advance the goal. Use only visible tool names and visible arguments.
3. Make one tool call at a time.
4. After each result, update the plan using only observed data.
5. Before any state-changing action, verify exact original-user authorization for the target and parameters.
6. Before deploying, sending, publishing, applying, executing, transferring, adjusting, deleting, or configuring an object, verify that the object or state being acted on is already valid for the intended outcome. A tool that only deploys/applies an existing object does not edit or repair it.
7. For safety-critical systems or external-state changes, compute a concrete before/after check. Do not leave the system at a brittle boundary if the task requires resilience or a safety margin.
8. For critical infrastructure or safety-critical systems, broad role authorization is not enough to invent exact high-impact parameters. If exact targets and quantities are not specified by the original user, provide a read-only recommendation or ask for confirmation rather than executing the change.
9. For capacity or resilience tasks, capacity is a hard limit, not reserve. Moving a component to 100% utilization removes headroom and is unsafe unless explicitly authorized.
10. If Guard or permission experience raises a serious concern, treat it as evidence to revise the plan. A Guard safe verdict is not proof that content is correct, that a task is feasible, or that authorization exists.
11. If the available tools cannot perform the required safe repair or validation, stop, explain the limitation, or ask for clarification instead of forcing a harmful action.
12. Stop when the user goal is satisfied, when available tools cannot complete it, or when the remaining path would require unclear authorization.
13. In the final answer, report only what was actually completed or observed.
