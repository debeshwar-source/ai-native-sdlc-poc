"""
Quality / TDD Agent.

Agentic: explores the repo to learn its real testing conventions,
then writes test files implementing the requirement's acceptance
criteria BEFORE any implementation exists -- genuine TDD ordering.
Restricted to writing only test-shaped paths.
"""

import re
from pathlib import Path
from typing import Callable, Any, Dict, Optional

import context.indexer as indexer
import llm
from agents._common import (
    CLARIFYING_QUESTIONS_GUIDANCE_TEXT,
    format_dict,
    format_note,
    format_requirement,
    make_repo_scoped_permission,
    validate_clarifying_questions,
    with_clarifying_questions_schema,
    written_paths_from_tool_calls,
)

SYSTEM_PROMPT = """
You are the Quality / TDD Agent in a Claude-Native SDLC system.

Your job is to write a minimal, deterministic test suite for the
requirement BEFORE any implementation exists. The tests will run
against whatever the Coding Agent produces next.

Read existing tests first with your tools and match the repo's real
conventions: test framework, fixtures, imports, existing helpers. Do
not invent a different testing style than what the repository
already uses.

Test only what the requirement's acceptance criteria actually claim.
Do not add tests for error codes, auth, validation, or edge cases
the requirement never mentioned. If an existing test already fully
covers the requirement, it is correct to write nothing new -- do not
create a redundant duplicate test just to have written something.

You may write ONLY to test-shaped paths (under a tests/ directory,
or named test_*, *_test, *.test.*, *.spec.*). Any other path is
rejected.

If FAILURE ANALYSIS is supplied below, this is a correction pass --
fix the identified test defect, do not recreate it.

When your test suite is complete, provide your final summary.

This is a fixed pipeline: Coding runs next automatically once you
finish, to implement against the tests you just wrote -- you do not
choose or name it.
""" + CLARIFYING_QUESTIONS_GUIDANCE_TEXT

TEST_PATH_PATTERNS = [
    r"(^|/)tests?/",
    r"(^|/)__tests__/",
    r"(^|/)spec/",
    r"(^|/)test_[^/]+\.py$",
    r"[^/]+_test\.py$",
    r"\.test\.[jt]sx?$",
    r"\.spec\.[jt]sx?$",
]


def is_test_path(path: str) -> bool:
    return any(re.search(pattern, path) for pattern in TEST_PATH_PATTERNS)


OUTPUT_SCHEMA = with_clarifying_questions_schema({
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": ["summary"],
})


def run(
    root: Path,
    index: indexer.RepoIndex,
    requirement: Dict[str, Any],
    architecture: Dict[str, Any],
    failure_analysis: Optional[Dict[str, Any]] = None,
    human_note: Optional[str] = None,
    on_event: Optional[Callable[[], None]] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    failure_section = ""
    if failure_analysis:
        failure_section = f"""
============================================================
FAILURE ANALYSIS (correction pass)
============================================================

{format_dict(failure_analysis)}
"""

    user_prompt = f"""
============================================================
REQUIREMENT
============================================================

{format_requirement(requirement)}

============================================================
ARCHITECTURE PLAN
============================================================

{format_dict(architecture)}
{failure_section}
============================================================
REPOSITORY OVERVIEW
============================================================

{indexer.repo_overview_text(index)}
{format_note(human_note)}
Explore existing tests with your tools first to match conventions,
write your test file(s), then provide your final summary.
"""

    agent_result = llm.run_agent(
        agent_name="Quality Agent",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        output_schema=OUTPUT_SCHEMA,
        required_keys=["summary", "clarifying_questions"],
        cwd=root,
        tools=["Read", "Write", "Edit", "Grep", "Glob"],
        can_use_tool=make_repo_scoped_permission(root, is_write_allowed=is_test_path),
        on_event=on_event,
        model=model,
    )

    # Writing nothing is valid: it means the Quality Agent judged that
    # existing tests already cover the requirement (expected on the
    # no-change-required path). Downstream testing runs the repo's full
    # suite either way, not just files listed here.
    result = dict(agent_result.result)
    validate_clarifying_questions(result, "Quality Agent")
    test_files: Dict[str, str] = {}
    for rel_path in written_paths_from_tool_calls(agent_result.tool_calls, root):
        full_path = root / rel_path
        if full_path.is_file():
            test_files[rel_path] = full_path.read_text(encoding="utf-8", errors="replace")
    result["test_files"] = test_files
    result["_tool_calls"] = agent_result.tool_calls
    return result
