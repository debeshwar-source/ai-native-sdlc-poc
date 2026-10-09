"""Shared prompt-formatting and tool-permission helpers used across agents."""

import json
import re
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

# Tools whose input carries a file path, and the key it's under. Every
# one of them -- read or write -- is kept inside the repo root; Write/
# Edit/NotebookEdit are additionally checked against an allow-list when
# one is given, since where content LANDS is the thing each agent's own
# scope actually restricts (exploration itself is never restricted).
_PATH_KEYED_TOOLS = {
    "Read": "file_path",
    "Write": "file_path",
    "Edit": "file_path",
    "NotebookEdit": "notebook_path",
    "Grep": "path",
    "Glob": "path",
}
_WRITE_TOOLS = {"Write", "Edit", "NotebookEdit"}


def make_repo_scoped_permission(
    root: Path,
    is_write_allowed: Optional[Callable[[str], bool]] = None,
    extra_read_roots: Optional[List[str]] = None,
) -> Callable[[str, Dict[str, Any]], "PermissionResultAllow | PermissionResultDeny"]:
    """A can_use_tool callback. Every path-taking tool call is confined to
    inside root (mirrors the granular repo-file-list scoping the original
    Bedrock tool definitions enforced). If is_write_allowed is given,
    Write/Edit/NotebookEdit are additionally checked against it -- pass
    None for a read-only agent, where any in-root path is fine.
    extra_read_roots are additional directories (the knowledge base)
    that may be READ -- never written -- from outside root."""
    extras = [os.path.realpath(r) for r in (extra_read_roots or [])]

    def _check(tool_name: str, tool_input: Dict[str, Any]):
        path_key = _PATH_KEYED_TOOLS.get(tool_name)
        if path_key is None:
            return PermissionResultAllow()

        raw_path = tool_input.get(path_key) or "."
        try:
            rel_path = os.path.relpath(raw_path, root)
        except ValueError:
            rel_path = raw_path

        if rel_path.startswith("..") or os.path.isabs(rel_path):
            real = os.path.realpath(raw_path)
            if tool_name not in _WRITE_TOOLS and any(
                real == r or real.startswith(r + os.sep) for r in extras
            ):
                return PermissionResultAllow()
            return PermissionResultDeny(
                message=f"{tool_name} to '{raw_path}' is outside the repository."
            )

        if tool_name in _WRITE_TOOLS and is_write_allowed is not None:
            if not is_write_allowed(rel_path):
                return PermissionResultDeny(
                    message=(
                        f"{tool_name} to '{rel_path}' is out of scope for "
                        f"this agent's assigned files."
                    )
                )

        return PermissionResultAllow()

    return _check


def written_paths_from_tool_calls(tool_calls: List[Dict[str, Any]], root: Path) -> List[str]:
    """Unique relative paths written or edited, in first-touched order --
    used to read back final on-disk content after an agent run, since an
    Edit call's own input doesn't carry the file's full resulting
    content (only the changed fragment)."""
    seen: List[str] = []
    for call in tool_calls:
        tool_name = call.get("tool")
        if tool_name not in _WRITE_TOOLS:
            continue
        raw_path = call.get("input", {}).get(_PATH_KEYED_TOOLS[tool_name])
        if not raw_path:
            continue
        try:
            rel_path = os.path.relpath(raw_path, root)
        except ValueError:
            rel_path = raw_path
        if rel_path not in seen:
            seen.append(rel_path)
    return seen


# ============================================================
# DEPENDENCY-MANIFEST FIX PATH: closes a real dead end -- Failure
# Analysis can correctly diagnose an ENVIRONMENT_ERROR (e.g. a package
# missing from the test-execution manifest) and know exactly which
# file fixes it, but Coding is normally restricted to exactly the
# files Architecture named for the ORIGINAL feature -- which never
# includes a dependency manifest, since that isn't part of the
# feature's own plan. Without this, a correctly-diagnosed,
# correctly-scoped fix has no path to actually get applied and the run
# dead-ends at FINALIZE/INCOMPLETE. See failure_analysis.py's
# `dependency_fix_files` and coding.py's use of it below.
#
# Deliberately a narrow, fixed allowlist of known manifest BASENAMES
# (not "whatever Failure Analysis says") -- an LLM's own judgment call
# about what needs fixing is trusted for WHICH manifest, never for
# whether an arbitrary path is safe to widen Coding's write scope to.
# ============================================================

DEPENDENCY_MANIFEST_BASENAMES = {
    "requirements.txt", "requirements-dev.txt", "requirements_dev.txt",
    "requirements-test.txt", "pyproject.toml",
    "package.json", "go.mod", "Cargo.toml",
}


def is_dependency_manifest_path(path: str) -> bool:
    return Path(path).name in DEPENDENCY_MANIFEST_BASENAMES


def format_note(human_note: Optional[str]) -> str:
    """The operator's free-text instructions -- standing orders typed
    into this agent in the workflow builder and/or a human-in-the-loop
    "suggest changes" note -- formatted as a prompt section -- or "" if
    there is none. Every agent splices this in the same spot every
    other optional feedback section goes, so it reads like any other
    revision-feedback signal rather than a bolted-on extra."""
    if not human_note:
        return ""
    return f"""
============================================================
OPERATOR INSTRUCTIONS (follow these; if they respond to your own last output, revise it to address them)
============================================================

{human_note}
"""


def format_requirement(requirement: Dict[str, Any]) -> str:
    lines = [
        f"Feature: {requirement.get('feature', '')}",
        f"Scope: {requirement.get('scope_summary', '')}",
        "Acceptance criteria:",
    ]
    lines += [f"- {c}" for c in requirement.get("acceptance_criteria", [])]
    if requirement.get("out_of_scope"):
        lines.append("Out of scope:")
        lines += [f"- {c}" for c in requirement["out_of_scope"]]
    return "\n".join(lines)


def format_work_order(architecture: Dict[str, Any]) -> str:
    """The Architecture -> Coding hand-off as a 4-part work order:
    objective, output, tools/references, boundaries + definition of
    done. Built from fields Architecture already returns (plus the
    objective/definition_of_done/boundaries it now adds), so Coding
    isn't left to infer what "done" means or what is off limits."""
    def bullets(items):
        return "\n".join(f"- {i}" for i in items) if items else "- (none)"

    output = (
        [f"modify {p}" for p in architecture.get("files_to_modify", [])]
        + [f"create {p}" for p in architecture.get("files_to_create", [])]
        + [f"delete {p}" for p in architecture.get("files_to_delete", [])]
    )
    return f"""
1. OBJECTIVE
{architecture.get('objective', '')}

2. OUTPUT (the exact files you are expected to produce)
{bullets(output)}

3. TOOLS & REFERENCES
- Follow these existing conventions rather than inventing a style:
{bullets(architecture.get('existing_conventions', []))}
- The test suite is already written (see below) -- run it, do not edit it.

4. BOUNDARIES & DEFINITION OF DONE
Boundaries (do NOT):
{bullets(architecture.get('boundaries', []))}
Done means:
{bullets(architecture.get('definition_of_done', []))}
"""


def format_dict(data: Dict[str, Any]) -> str:
    return json.dumps(data or {}, indent=2, default=str)


# ============================================================
# ROUTING AUTONOMY: the pipeline is otherwise a FIXED sequence --
# Requirement -> Architecture -> (Quality, if a change is needed) ->
# Coding -> Testing -- deterministic edges in orchestrator/graph.py,
# no agent choice involved. Only Failure Analysis (after a test
# failure: is the application, the test, or the plan at fault?) and
# Review (does this pass, and if not, who owns the fix?) are genuine
# judgment calls that can't be hardcoded, so those two alone choose
# `next_agent`. Each gets its OWN restricted enum (FAILURE_ANALYSIS_
# NEXT_AGENT_NAMES / REVIEW_NEXT_AGENT_NAMES below) -- Review can't
# even express "send this to Failure Analysis" or vice versa. This
# replaces an earlier design where all 7 pre-release stages chose
# freely among all 9 targets, which could oscillate across many
# stages before ever hitting the overall step cap; restricting BOTH
# the schema (what a choice can even say) and the graph topology
# (only two nodes have a conditional edge at all) closes that off
# structurally rather than just capping how long it takes to happen.
# ============================================================

FAILURE_ANALYSIS_NEXT_AGENT_NAMES = ["CODING", "QUALITY", "ARCHITECTURE", "FINALIZE"]
REVIEW_NEXT_AGENT_NAMES = ["RELEASE_PLANNING", "CODING", "QUALITY", "ARCHITECTURE"]

FAILURE_ANALYSIS_ROUTING_TEXT = """
============================================================
WHAT HAPPENS NEXT
============================================================

This pipeline is otherwise a fixed sequence up to this point:
REQUIREMENT -> ARCHITECTURE -> (QUALITY, if a change is needed) ->
CODING -> TESTING -> you. Your diagnosis is one of only two places in
this whole pipeline where a genuine judgment call decides what
happens next -- choose `next_agent` accordingly:
- CODING: the application itself is at fault (or a fixable
  environment/dependency issue -- see dependency_fix_files above).
- QUALITY: the test itself is wrong or missing coverage.
- ARCHITECTURE: the underlying plan was wrong, not just its
  implementation.
- FINALIZE: only for a genuine infrastructure problem no agent here
  can fix -- never as a way to give up on a diagnosable failure.
Explain your choice in one sentence via `next_agent_reason`. Whichever
stage you send this to picks up the fixed sequence again from there
(e.g. ARCHITECTURE re-plans, then QUALITY/CODING/TESTING all run
again automatically -- you don't need to route through each of those
yourself).

Orchestrator-enforced regardless of what you choose: CODING is
unreachable with nothing for it to act on (no files named by
ARCHITECTURE and no dependency_fix_files) -- redirected to ARCHITECTURE
instead. The run also has a hard overall step limit and a stagnation
limit (repeating the identical test failure); once reached, it ends
regardless of what you recommend.
"""

REVIEW_ROUTING_TEXT = """
============================================================
WHAT HAPPENS NEXT
============================================================

This pipeline is otherwise a fixed sequence up to this point:
REQUIREMENT -> ARCHITECTURE -> (QUALITY, if a change is needed) ->
CODING -> TESTING -> you. Your review is the other of only two places
in this whole pipeline where a genuine judgment call decides what
happens next -- choose `next_agent` accordingly:
- RELEASE_PLANNING: status is PASS -- this change is ready to ship.
- CODING: an implementation defect.
- QUALITY: the test itself is wrong or missing coverage.
- ARCHITECTURE: the underlying plan was wrong, not just its
  implementation.
Explain your choice in one sentence via `next_agent_reason`. Whichever
stage you send this to picks up the fixed sequence again from there.

Orchestrator-enforced regardless of what you choose: RELEASE_PLANNING
is unreachable unless status is actually PASS (including a
deterministic Semgrep/pip-audit/scope-creep gate you don't control) --
redirected back to you instead. The run also has a hard overall step
limit and a stagnation limit; once reached, it ends regardless of what
you recommend.
"""


QA_REVIEW_ROUTING_TEXT = """
============================================================
WHAT HAPPENS NEXT
============================================================

You are the second of two reviewers. Code Review already passed
(correctness, security, scope). Your verdict decides whether this
change is released -- choose `next_agent` accordingly:
- RELEASE_PLANNING: status is PASS -- every acceptance criterion is
  demonstrably covered.
- QUALITY: a criterion has no test, or the tests are too weak to prove
  it (the usual owner of a coverage gap).
- CODING: a test exercises the criterion and the behavior is actually
  wrong or missing.
- ARCHITECTURE: the plan itself cannot satisfy the requirement.
Explain your choice in one sentence via `next_agent_reason`.

Orchestrator-enforced regardless of what you choose: RELEASE_PLANNING
is unreachable unless status is actually PASS, and any acceptance
criterion you mark UNCOVERED (or fail to assess) is appended as a
BLOCKING finding automatically. The run also has a hard overall step
limit; once reached, it ends regardless of what you recommend.
"""


def with_next_agent_schema(schema: Dict[str, Any], allowed_next_agents: List[str]) -> Dict[str, Any]:
    """Merges next_agent/next_agent_reason into an agent's own
    OUTPUT_SCHEMA. Only failure_analysis.py and review.py call this now
    -- allowed_next_agents scopes the enum to exactly what THAT agent
    can sanely choose (see the module-level comment above)."""
    merged = dict(schema)
    properties = dict(merged.get("properties", {}))
    properties["next_agent"] = {
        "type": "string",
        "enum": allowed_next_agents,
        "description": (
            "Which stage should run next -- your own judgment call. "
            "See WHAT HAPPENS NEXT for what each choice means and the "
            "invariants the orchestrator still enforces regardless of "
            "this choice."
        ),
    }
    properties["next_agent_reason"] = {
        "type": "string",
        "description": "One sentence: why that stage, specifically.",
    }
    merged["properties"] = properties
    merged["required"] = list(merged.get("required", [])) + ["next_agent", "next_agent_reason"]
    return merged


# ============================================================
# CLARIFICATORY LAYER: every human-gated agent may flag a genuine
# decision it can't make for itself -- surfaced ONLY in Human approval
# mode (see streamlit_app.py's approval gate), as options the reviewer
# picks from before the next agent is ever invoked. In Agentic mode
# nobody reads this field at all; the agent still had to make its own
# best call regardless (see the guidance text below), so behavior
# there is unchanged.
# ============================================================

MAX_CLARIFYING_QUESTIONS = 2

# A question is read on a card, not in a terminal: it must be answerable
# without reading the investigation behind it. validate_clarifying_questions
# enforces these limits so a long report can't masquerade as a question.
MAX_QUESTION_CHARS = 280
MAX_OPTION_CHARS = 120

CLARIFYING_QUESTIONS_GUIDANCE_TEXT = f"""
============================================================
CLARIFYING QUESTIONS (optional -- shown to a human reviewer ONLY in
Human approval mode; ignored entirely in Agentic mode)
============================================================

`clarifying_questions` is for genuine, material ambiguity ONLY -- one
where a wrong guess would change what you actually build or how this
run proceeds, not an implementation detail you'd reasonably settle on
your own. At most {MAX_CLARIFYING_QUESTIONS} -- do not manufacture
questions to seem thorough; an empty list is the normal, expected
outcome when nothing clears that bar. This is not a substitute for
making a defensible call yourself: produce your actual output using
your own best judgment regardless of whether a human ever answers --
the question just captures the alternative you didn't take, for a
human to resolve if they want to. Each entry needs 2-4 concrete,
mutually exclusive options (never a generic yes/no) plus which one you
recommend in `recommended_option`. If a human already answered one of
these on a prior revision pass (see the revision-feedback section
above), that ambiguity is now RESOLVED -- fold their answer into your
actual output and do not ask it again.

Shape each question so it is answerable on its own, without reading
anything else: ONE sentence (at most {MAX_QUESTION_CHARS} characters)
stating exactly what decision you need, then short options (at most
{MAX_OPTION_CHARS} characters each). Put no findings, background or
reasoning in the question -- that belongs in your normal output. If the
ask would need a paragraph to explain, it is a report, not a question.
"""

CLARIFYING_QUESTIONS_SCHEMA_PROPERTY = {
    "type": "array",
    "maxItems": MAX_CLARIFYING_QUESTIONS,
    "items": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "options": {"type": "array", "items": {"type": "string"}},
            "recommended_option": {"type": "string"},
        },
        "required": ["question", "options", "recommended_option"],
    },
}


def with_clarifying_questions_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Merges clarifying_questions into an agent's own OUTPUT_SCHEMA --
    every human-gated agent gets this (see CLARIFYING_QUESTIONS_GUIDANCE_TEXT),
    so the orchestrator can read result["clarifying_questions"] uniformly
    regardless of which agent produced it."""
    merged = dict(schema)
    properties = dict(merged.get("properties", {}))
    properties["clarifying_questions"] = CLARIFYING_QUESTIONS_SCHEMA_PROPERTY
    merged["properties"] = properties
    merged["required"] = list(merged.get("required", [])) + ["clarifying_questions"]
    return merged


def _shorten(text: str, limit: int) -> str:
    """Keeps whole sentences up to `limit` chars; a single over-long
    sentence is hard-cut with an ellipsis."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    kept = ""
    for sentence in re.split(r"(?<=[.?!])\s+", text):
        candidate = f"{kept} {sentence}".strip()
        if len(candidate) > limit:
            break
        kept = candidate
    return kept or text[: limit - 1].rstrip() + "…"


def validate_clarifying_questions(result: Dict[str, Any], agent_name: str) -> None:
    """Shared shape check for the clarifying_questions field every
    human-gated agent now returns. Never raises: this is an optional,
    best-effort UI convenience, shown to a human ONLY in Human approval
    mode (see CLARIFYING_QUESTIONS_GUIDANCE_TEXT) -- the agent's actual,
    load-bearing output is never contingent on it, so a malformed entry
    is repaired or dropped here rather than failing an entire
    multi-minute pipeline run over what is, at worst, an LLM phrasing
    quirk in a secondary feature (e.g. paraphrasing its own
    recommendation instead of repeating an option's exact text -- a
    real, observed failure this now tolerates instead of crashing on).
    Mutates result["clarifying_questions"] in place."""
    questions = result.get("clarifying_questions")
    if not isinstance(questions, list):
        result["clarifying_questions"] = []
        return

    cleaned: List[Dict[str, Any]] = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        question_text = q.get("question")
        options = q.get("options")
        if not question_text or not isinstance(options, list) or len(options) < 2:
            continue

        recommended = q.get("recommended_option")
        if recommended not in options:
            recommended = None

        # Enforce the decision-shaped limit (see MAX_QUESTION_CHARS).
        # Shortening runs AFTER the recommended-option match above so
        # trimming an option can't break it; the recommendation is
        # re-pointed at its trimmed twin.
        shortened = [_shorten(str(o), MAX_OPTION_CHARS) for o in options]
        if recommended is not None:
            recommended = shortened[options.index(recommended)]

        cleaned.append(
            {
                "question": _shorten(str(question_text), MAX_QUESTION_CHARS),
                "options": shortened,
                "recommended_option": recommended,
            }
        )

    # Cap applied to the CLEANED list, not the raw one -- a malformed
    # entry occupying an early slot must never crowd out a later, valid
    # one just because of raw positional bad luck.
    result["clarifying_questions"] = cleaned[:MAX_CLARIFYING_QUESTIONS]
