"""
Requirement Agent.

A single structured call, not agentic -- turning a spec into a
testable requirement doesn't need repo exploration, just the cheap
overview already produced by the context layer. Deliberately
conservative: it must not invent scope beyond the spec.
"""

from typing import Callable, Any, Dict, List, Optional

import llm
from agents._common import (
    CLARIFYING_QUESTIONS_GUIDANCE_TEXT,
    format_note,
    validate_clarifying_questions,
    with_clarifying_questions_schema,
)

SYSTEM_PROMPT = """
You are the Requirement Agent in a Claude-Native SDLC system.

Your ONLY job is to translate a raw engineering spec into a concise,
testable requirement. You are NOT allowed to expand, embellish, or
add unstated behavior (authentication, validation rules, pagination,
error codes, performance requirements, etc.) unless the spec states
them or they are directly and unambiguously implied.

Every acceptance criterion must be traceable to specific words in
the spec. If you can't point to the words that support a criterion,
do not include it.

This is the first stage of a fixed pipeline: Architecture runs next
automatically once you finish -- you do not choose or name it.
""" + CLARIFYING_QUESTIONS_GUIDANCE_TEXT

OUTPUT_SCHEMA = with_clarifying_questions_schema({
    "type": "object",
    "properties": {
        "feature": {
            "type": "string",
            "description": "short description using only the spec's own words",
        },
        "scope_summary": {
            "type": "string",
            "description": "1-2 sentence summary of what must change",
        },
        "acceptance_criteria": {
            "type": "array",
            "items": {"type": "string"},
            "description": "each criterion directly supported by the spec",
        },
        "out_of_scope": {
            "type": "array",
            "items": {"type": "string"},
            "description": "explicitly excluded, or reasonably adjacent behavior NOT requested",
        },
    },
    "required": ["feature", "scope_summary", "acceptance_criteria", "out_of_scope"],
})


def _format_history(entries: List[dict]) -> str:
    if not entries:
        return "(none)"
    lines = []
    for entry in entries[:3]:
        lines.append(
            f"- {entry.get('spec_text', '')[:200]} "
            f"(outcome: {entry.get('outcome', 'UNKNOWN')}, "
            f"files touched: {entry.get('files_touched', [])})"
        )
    return "\n".join(lines)


def run(
    spec_text: str,
    repo_overview: str,
    prior_runs: Optional[List[dict]] = None,
    human_note: Optional[str] = None,
    on_event: Optional[Callable[[], None]] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    if not spec_text or not spec_text.strip():
        raise ValueError("Spec text cannot be empty.")

    prompt = f"""
============================================================
REPOSITORY OVERVIEW (context only; the spec is authoritative)
============================================================

{repo_overview}

============================================================
PRIOR RELATED RUNS AGAINST THIS REPO (context only)
============================================================

{_format_history(prior_runs or [])}
{format_note(human_note)}
============================================================
SPEC
============================================================

{spec_text}
"""

    agent_result = llm.run_agent(
        agent_name="Requirement Agent",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
        output_schema=OUTPUT_SCHEMA,
        required_keys=[
            "feature", "scope_summary", "acceptance_criteria", "out_of_scope",
            "clarifying_questions",
        ],
        on_event=on_event,
        model=model,
    )
    result = agent_result.result

    if not isinstance(result.get("acceptance_criteria"), list) or not result["acceptance_criteria"]:
        raise ValueError("Requirement Agent returned no acceptance criteria.")

    validate_clarifying_questions(result, "Requirement Agent")
    return result
