"""
Release & Operations Agent.

Produces the release plan and the Jira/Confluence-shaped summaries.
It does NOT push, commit, or open the PR itself -- those are real,
deterministic actions the orchestrator performs afterward via
workspace.py/github_api.py.
"""

from typing import Callable, Any, Dict, List, Optional

import llm
from agents._common import (
    CLARIFYING_QUESTIONS_GUIDANCE_TEXT,
    format_dict,
    format_note,
    format_requirement,
    validate_clarifying_questions,
    with_clarifying_questions_schema,
)

SYSTEM_PROMPT = """
You are the Release & Operations Agent in a Claude-Native SDLC system.

The change below already passed testing and review. Produce a
release plan and the summaries used to record this change in the
team's issue tracker and wiki.

Describe deployment/rollout/monitoring at the level of detail
actually implied by the requirement and architecture -- do not
assume a specific cloud provider or CI/CD system unless the
repository's own manifests indicate one.
""" + CLARIFYING_QUESTIONS_GUIDANCE_TEXT

OUTPUT_SCHEMA = with_clarifying_questions_schema({
    "type": "object",
    "properties": {
        "release_title": {"type": "string"},
        "release_summary": {"type": "string"},
        "deployment_steps": {"type": "array", "items": {"type": "string"}},
        "rollout_strategy": {"type": "string"},
        "monitoring_plan": {"type": "array", "items": {"type": "string"}},
        "rollback_plan": {"type": "string"},
        "feedback_to_requirements": {
            "type": "string",
            "description": "gaps/follow-ups for the next cycle, including any FOLLOW_UP review findings",
        },
        "issue_summary": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
                "status": {"type": "string"},
            },
            "required": ["title", "description", "acceptance_criteria", "status"],
        },
        "wiki_summary": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "heading": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["heading", "content"],
                    },
                },
            },
            "required": ["title", "sections"],
        },
    },
    "required": [
        "release_title",
        "release_summary",
        "deployment_steps",
        "rollout_strategy",
        "monitoring_plan",
        "rollback_plan",
        "feedback_to_requirements",
        "issue_summary",
        "wiki_summary",
    ],
})


def run(
    requirement: Dict[str, Any],
    architecture: Dict[str, Any],
    testing: Dict[str, Any],
    review: Dict[str, Any],
    changed_files: List[str],
    human_note: Optional[str] = None,
    on_event: Optional[Callable[[], None]] = None,
    model: Optional[str] = None,
    qa_review: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    qa_section = ""
    if qa_review:
        qa_section = f"""
============================================================
QA REVIEW OUTCOME (acceptance-criteria coverage)
============================================================

{format_dict({k: v for k, v in qa_review.items() if not k.startswith('_')})}
"""

    prompt = f"""
============================================================
REQUIREMENT
============================================================

{format_requirement(requirement)}

============================================================
ARCHITECTURE PLAN
============================================================

{format_dict(architecture)}

============================================================
TEST RESULT / SCORECARD
============================================================

{format_dict(testing)}

============================================================
REVIEW OUTCOME
============================================================

{format_dict(review)}
{qa_section}
============================================================
CHANGED FILES
============================================================

{chr(10).join(f"- {path}" for path in changed_files)}
{format_note(human_note)}
"""

    agent_result = llm.run_agent(
        agent_name="Release Agent",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
        output_schema=OUTPUT_SCHEMA,
        required_keys=list(OUTPUT_SCHEMA["required"]),
        on_event=on_event,
        model=model,
    )

    result = agent_result.result
    validate_clarifying_questions(result, "Release Agent")
    return result
