"""
Architecture Agent.

Agentic, read-only: explores the real repository with Claude Code's
built-in Read/Grep/Glob tools before deciding what must change. Never
invents an existing contract -- if the repo already satisfies the
requirement, it must say so.
"""

from pathlib import Path
from typing import Callable, Any, Dict, Optional

import context.indexer as indexer
import design as design_module
import llm
from agents._common import (
    CLARIFYING_QUESTIONS_GUIDANCE_TEXT,
    format_note,
    format_requirement,
    make_repo_scoped_permission,
    validate_clarifying_questions,
    with_clarifying_questions_schema,
)

SYSTEM_PROMPT = """
You are the Architecture Agent in a Claude-Native SDLC system.

Your job is to define the MINIMUM implementation contract required
to satisfy the requirement, grounded in the ACTUAL repository -- use
your tools (Read, Grep, Glob) to inspect it before deciding anything.
Do not guess at file contents or conventions.

MOST IMPORTANT RULE: never invent an existing contract. If the
repository already implements the requested behavior, explicitly say
no change is required -- do not ask the Coding Agent to recreate
something that already exists.

Prefer the smallest change that satisfies the requirement:
- Modify existing files over creating new ones where it fits the
  repo's own conventions.
- New files must follow the naming/layout conventions you observe
  in the repo, not an invented style.
- Avoid new dependencies, services, or infrastructure unless the
  requirement explicitly needs them.
- If the requirement genuinely requires removing a file (not just
  emptying it or leaving a stub), name it in files_to_delete. The
  Coding Agent has a real delete capability -- do not work around a
  deletion requirement by planning to blank out a file's contents
  instead, and do not assume deletion is impossible.

Every path you name in files_to_modify/files_to_delete must be one
you actually verified exists via Read or Glob -- never guess a path.
files_to_create must not already exist.

When you are confident in your plan, provide your final architecture
decision matching the required output schema.

Your output is also the work order Coding runs from, so be explicit:
- `objective`: one sentence -- the concrete goal of this change.
- `boundaries`: what Coding must NOT do (areas, behaviors or files that
  are off limits, beyond simply "everything not named above").
- `definition_of_done`: observable, checkable conditions (specific tests
  passing, specific behavior present) -- not "implementation complete".

DESIGN LAYER -- also describe the architecture so a person can grasp it
at a glance, in `design`:
- `components`: 4-12 architecture-level pieces (modules, services, layers,
  data stores) -- NOT individual files. Each cites the real repo paths
  that belong to it (paths you verified); every file you plan to modify,
  create or delete should belong to some component. Mark `is_new` true
  only for a component this change introduces.
- `relationships`: how components depend on each other, each verified by
  what you read (imports, calls), labelled in a few words. Mark `is_new`
  true only for a dependency this change introduces.
- `existing_overview` / `proposed_overview`: plain language for someone
  who has never opened this repo -- no file paths, no jargon.
If no change is required, `proposed_overview` says so and nothing is new.
The component statuses (modified/new/removed) are derived from your
files_to_* lists, so keep those accurate rather than trying to flag them.

This is a fixed pipeline: if change_required is true, Quality runs
next automatically (this is TDD ordering -- tests must exist before
Coding runs); if false, Testing runs next to verify the existing
suite still passes against the unchanged repo. You do not choose or
name either.
""" + CLARIFYING_QUESTIONS_GUIDANCE_TEXT

OUTPUT_SCHEMA = with_clarifying_questions_schema({
    "type": "object",
    "properties": {
        "change_required": {"type": "boolean"},
        "files_to_modify": {"type": "array", "items": {"type": "string"}},
        "files_to_create": {"type": "array", "items": {"type": "string"}},
        "files_to_delete": {"type": "array", "items": {"type": "string"}},
        "changes": {"type": "array", "items": {"type": "string"}},
        "implementation_plan": {"type": "array", "items": {"type": "string"}},
        "existing_conventions": {"type": "array", "items": {"type": "string"}},
        "risk_notes": {"type": "array", "items": {"type": "string"}},
        # The hand-off to Coding is a 4-part work order (see
        # agents/_common.py's format_work_order): these three are the
        # parts Architecture alone is placed to say.
        "objective": {"type": "string"},
        "boundaries": {"type": "array", "items": {"type": "string"}},
        "definition_of_done": {"type": "array", "items": {"type": "string"}},
        # Design layer: a comprehensible component-level picture of the
        # repo today and after this change (see design.py).
        "design": design_module.DESIGN_SCHEMA,
    },
    "required": [
        "change_required",
        "files_to_modify",
        "files_to_create",
        "files_to_delete",
        "changes",
        "implementation_plan",
        "existing_conventions",
        "objective",
        "boundaries",
        "definition_of_done",
        "design",
    ],
})


def run(
    root: Path,
    index: indexer.RepoIndex,
    requirement: Dict[str, Any],
    human_note: Optional[str] = None,
    on_event: Optional[Callable[[], None]] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    user_prompt = f"""
============================================================
REQUIREMENT
============================================================

{format_requirement(requirement)}

============================================================
REPOSITORY OVERVIEW
============================================================

{indexer.repo_overview_text(index)}
{format_note(human_note)}
Explore whatever files you need, then provide your final architecture
decision.
"""

    agent_result = llm.run_agent(
        agent_name="Architecture Agent",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        output_schema=OUTPUT_SCHEMA,
        cwd=root,
        tools=["Read", "Grep", "Glob"],
        can_use_tool=make_repo_scoped_permission(root),
        on_event=on_event,
        model=model,
    )

    result = agent_result.result
    _validate(result)
    # Advisory and best-effort: never fails the stage (see design.py).
    result["design"] = design_module.normalize(result.get("design"), result, index.file_list)
    validate_clarifying_questions(result, "Architecture Agent")
    result["_tool_calls"] = agent_result.tool_calls
    return result


def _validate(result: Dict[str, Any]) -> None:
    if not isinstance(result["change_required"], bool):
        raise ValueError("Architecture Agent 'change_required' must be a boolean.")

    any_files_named = (
        result["files_to_modify"] or result["files_to_create"] or result["files_to_delete"]
    )

    _default_work_order(result)

    if result["change_required"]:
        if not any_files_named:
            raise ValueError(
                "Architecture Agent requires a change but named no files."
            )
    else:
        if any_files_named:
            raise ValueError(
                "Architecture Agent named files to touch despite "
                "change_required being false."
            )


def _default_work_order(result: Dict[str, Any]) -> None:
    """The work-order fields are required by the schema, but a thin
    answer must not crash a multi-minute run. Fill any gap with a safe,
    deterministic default derived from the plan itself, so Coding
    always receives a complete work order."""
    if not str(result.get("objective") or "").strip():
        changes = result.get("changes") or []
        result["objective"] = changes[0] if changes else "Satisfy the requirement with the minimum change."
    if not result.get("boundaries"):
        result["boundaries"] = ["Touch only the files named in this plan; do not refactor unrelated code."]
    if not result.get("definition_of_done"):
        result["definition_of_done"] = ["The pre-written test suite passes against the changed code."]
