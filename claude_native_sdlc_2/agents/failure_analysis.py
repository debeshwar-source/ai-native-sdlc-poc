"""
Failure Analysis Agent.

Not one of the pipeline's core stages, but the plumbing that makes
the coding<->testing retry loop self-correcting: diagnoses WHICH
artifact (application vs test) is responsible for a failure, rather
than blindly sending every failure back to the Coding Agent.
"""

from pathlib import Path
from typing import Callable, Any, Dict, Optional

import context.indexer as indexer
import llm
from agents._common import (
    CLARIFYING_QUESTIONS_GUIDANCE_TEXT,
    FAILURE_ANALYSIS_NEXT_AGENT_NAMES,
    FAILURE_ANALYSIS_ROUTING_TEXT,
    format_dict,
    format_note,
    format_requirement,
    make_repo_scoped_permission,
    validate_clarifying_questions,
    with_clarifying_questions_schema,
    with_next_agent_schema,
)

SYSTEM_PROMPT = """
You are the Failure Analysis Agent in a Claude-Native SDLC system.

Diagnose why the current test run failed and which artifact is
responsible. Use your tools to actually inspect the relevant code
and test file(s) -- do not guess from the error text alone.

A failing test is evidence of a discrepancy, not proof the
application is wrong. Blame the application only when the
requirement/architecture plan establishes the expected behavior and
the application violates it. A test with invented behavior, wrong
API usage, or broken setup is a TEST_DEFECT.

When you are done, provide your final diagnosis.
""" + FAILURE_ANALYSIS_ROUTING_TEXT + """
Map failure_type to next_agent: APPLICATION_DEFECT or SYNTAX_ERROR ->
CODING. TEST_DEFECT -> QUALITY. CONTRACT_MISMATCH (the plan itself was
wrong, not just its implementation) -> ARCHITECTURE. ENVIRONMENT_ERROR
-> CODING if it's fixable in the repo (e.g. a missing import),
FINALIZE if it's an infrastructure problem no agent here can fix.
UNKNOWN with LOW confidence -> still pick your best guess; do not
choose FINALIZE just because you're unsure.

If failure_type is ENVIRONMENT_ERROR and the fix is adding or editing
a dependency declaration (e.g. a package missing from the
test-execution manifest, unrelated to this feature's own files), name
the EXACT existing dependency-manifest file(s) that need editing in
`dependency_fix_files` (e.g. "requirements-dev.txt") and choose CODING
-- Coding is authorized to write to exactly those files for this pass,
on top of its normal scope, ONLY to apply this fix. Never name a
source file here; that is what files_to_modify/files_to_create are
for. Leave `dependency_fix_files` empty for every other failure_type,
and for an ENVIRONMENT_ERROR that isn't a dependency-manifest fix.
""" + CLARIFYING_QUESTIONS_GUIDANCE_TEXT

OUTPUT_SCHEMA = with_clarifying_questions_schema(with_next_agent_schema({
    "type": "object",
    "properties": {
        "failure_type": {
            "type": "string",
            "enum": [
                "APPLICATION_DEFECT",
                "TEST_DEFECT",
                "CONTRACT_MISMATCH",
                "SYNTAX_ERROR",
                "ENVIRONMENT_ERROR",
                "UNKNOWN",
            ],
        },
        "root_cause": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
        "dependency_fix_files": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Existing dependency-manifest file(s) (e.g. "
                "requirements-dev.txt, package.json) that need editing to "
                "fix a diagnosed ENVIRONMENT_ERROR -- empty for every "
                "other case. Never a source file."
            ),
        },
    },
    "required": ["failure_type", "root_cause", "evidence", "dependency_fix_files"],
}, FAILURE_ANALYSIS_NEXT_AGENT_NAMES))


def run(
    root: Path,
    index: indexer.RepoIndex,
    requirement: Dict[str, Any],
    architecture: Dict[str, Any],
    testing: Dict[str, Any],
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

============================================================
ARCHITECTURE PLAN
============================================================

{format_dict(architecture)}

============================================================
TEST EXECUTION RESULT
============================================================

{format_dict(testing.get('test_result', testing))}
{format_note(human_note)}
Inspect the relevant application and test file(s) with your tools,
then provide your final diagnosis.
"""

    agent_result = llm.run_agent(
        agent_name="Failure Analysis Agent",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        output_schema=OUTPUT_SCHEMA,
        required_keys=[
            "failure_type", "root_cause", "evidence", "next_agent",
            "clarifying_questions", "dependency_fix_files",
        ],
        cwd=root,
        tools=["Read", "Grep", "Glob"],
        can_use_tool=make_repo_scoped_permission(root),
        on_event=on_event,
        model=model,
    )

    result = agent_result.result
    validate_clarifying_questions(result, "Failure Analysis Agent")
    result["_tool_calls"] = agent_result.tool_calls
    return result
