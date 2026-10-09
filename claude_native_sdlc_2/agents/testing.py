"""
Testing Agent.

Mostly deterministic (real test execution against the workspace
checkout) with a single structured LLM call to turn the raw result
into a quality report and Go/No-Go recommendation.
"""

from pathlib import Path
from typing import Callable, Any, Dict, Optional

import llm
import tools.test_runner as test_runner
from agents._common import (
    CLARIFYING_QUESTIONS_GUIDANCE_TEXT,
    format_dict,
    format_note,
    format_requirement,
    validate_clarifying_questions,
    with_clarifying_questions_schema,
)

SYSTEM_PROMPT = """
You are the Testing Agent in a Claude-Native SDLC system. You are given
the ACTUAL, already-executed test results -- you do not run tests
yourself. Summarize them honestly and recommend Go/No-Go for release.

NO_GO if the test run failed or was skipped (no test command could
be detected). GO only when the tests actually passed.

This is a fixed pipeline: what runs next (Review, Failure Analysis, or
Finalize) is decided mechanically from the actual test result, not by
you -- just report honestly.
""" + CLARIFYING_QUESTIONS_GUIDANCE_TEXT

OUTPUT_SCHEMA = with_clarifying_questions_schema({
    "type": "object",
    "properties": {
        "quality_report": {
            "type": "string",
            "description": "2-4 sentence honest summary of test outcome and coverage of the acceptance criteria",
        },
        "go_no_go": {"type": "string", "enum": ["GO", "NO_GO"]},
        "rationale": {"type": "string"},
    },
    "required": ["quality_report", "go_no_go", "rationale"],
})


def run(
    root: Path,
    requirement: Dict[str, Any],
    review: Dict[str, Any],
    change_required: Optional[bool] = None,
    human_note: Optional[str] = None,
    on_event: Optional[Callable[[], None]] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    test_result = test_runner.run_tests(root)

    prompt = f"""
============================================================
REQUIREMENT
============================================================

{format_requirement(requirement)}

============================================================
change_required (from Architecture): {change_required}
============================================================

============================================================
REVIEW OUTCOME
============================================================

{format_dict(review)}

============================================================
TEST EXECUTION RESULT (already run; do not re-run)
============================================================

{format_dict(test_result)}
{format_note(human_note)}
"""

    agent_result = llm.run_agent(
        agent_name="Testing Agent",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
        output_schema=OUTPUT_SCHEMA,
        required_keys=["quality_report", "go_no_go", "rationale", "clarifying_questions"],
        on_event=on_event,
        model=model,
    )
    scorecard = agent_result.result
    validate_clarifying_questions(scorecard, "Testing Agent")

    if test_result["status"] != "PASS" and scorecard.get("go_no_go") != "NO_GO":
        # Orchestrator-level invariant: a non-passing test run can never
        # be talked into a GO by the summarizing call.
        scorecard["go_no_go"] = "NO_GO"

    return {
        "test_result": test_result,
        "scorecard": scorecard,
        "clarifying_questions": scorecard.get("clarifying_questions") or [],
    }
