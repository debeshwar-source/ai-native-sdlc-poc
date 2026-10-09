"""
QA Review Agent -- the second of two reviewers.

Code Review (agents/review.py) asks "is this code correct, safe and in
scope?". QA Review asks a different question: "does the EVIDENCE prove
the requirement is met?". It checks each acceptance criterion against
the tests that actually ran, and flags criteria with no test, tests too
weak to prove anything (asserting nothing, mocking away the behavior),
missing negative/edge cases, and an unmet definition of done.

It deliberately does not redo Code Review's job (style, security,
scope) -- it is shown Code Review's outcome only so it doesn't repeat it.

Deterministic gate, same orchestrator-invariant pattern as Code
Review's static-analysis gate: every acceptance criterion is given an
id (C1..Cn) and the model must assess each. A criterion marked
UNCOVERED, or never assessed at all, is appended as a BLOCKING finding
regardless of the model's own status; PARTIAL (and COVERED with no
cited evidence) becomes a FOLLOW_UP. Unverified means not done.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import context.indexer as indexer
import llm
import workspace
from agents._common import (
    CLARIFYING_QUESTIONS_GUIDANCE_TEXT,
    QA_REVIEW_ROUTING_TEXT,
    REVIEW_NEXT_AGENT_NAMES,
    format_dict,
    format_note,
    make_repo_scoped_permission,
    validate_clarifying_questions,
    with_clarifying_questions_schema,
    with_next_agent_schema,
)

SYSTEM_PROMPT = """
You are the QA Review Agent in a Claude-Native SDLC system. A separate
Code Review Agent has already judged the code itself. Your job is
different: decide whether the tests and results PROVE each acceptance
criterion is met.

For EVERY acceptance criterion (given with ids C1, C2, ...), find the
test(s) that exercise it -- use Read/Grep/Glob to open the actual test
files -- and classify it:
- COVERED: a test exercises this behavior and would fail if it were
  broken. Cite the test (file::name) and say what it asserts.
- PARTIAL: tested, but only the happy path, or the assertion is too
  weak to prove the criterion (e.g. checks status code but not body).
- UNCOVERED: no test exercises it, or the only test is vacuous
  (asserts nothing meaningful, or mocks away the very behavior).

Also check:
- the architecture's definition_of_done -- is each item demonstrably
  met by the evidence?
- missing negative/edge cases the criteria clearly imply (invalid
  input, empty state, error path).

Do NOT repeat Code Review's work: no style, security or scope findings.
A finding must complete the sentence "the evidence does NOT show
<specific required thing>". If a criterion is genuinely covered, that
is a COVERED entry in criteria_coverage, never a finding. The normal,
expected outcome for well-tested work is PASS with an empty findings
list -- do not manufacture findings to seem thorough.

Classify each real finding as BLOCKING (the requirement is not proven
met) or FOLLOW_UP (worth doing, does not block release).
""" + QA_REVIEW_ROUTING_TEXT + CLARIFYING_QUESTIONS_GUIDANCE_TEXT

OUTPUT_SCHEMA = with_clarifying_questions_schema(with_next_agent_schema({
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["PASS", "CHANGES_REQUESTED"]},
        "summary": {"type": "string"},
        "criteria_coverage": {
            "type": "array",
            "description": "One entry per acceptance criterion id, no omissions.",
            "items": {
                "type": "object",
                "properties": {
                    "criterion_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["COVERED", "PARTIAL", "UNCOVERED"]},
                    "evidence": {
                        "type": "string",
                        "description": "The test (file::name) and what it asserts; or why none proves it.",
                    },
                },
                "required": ["criterion_id", "status", "evidence"],
            },
        },
        "findings": {
            "type": "array",
            "description": "ONLY gaps in the evidence. Empty if the requirement is proven met.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "severity": {"type": "string", "enum": ["BLOCKING", "FOLLOW_UP"]},
                    "finding": {"type": "string"},
                    "evidence": {"type": "string"},
                    "suggested_fix": {"type": "string"},
                },
                "required": ["id", "severity", "finding", "evidence"],
            },
        },
    },
    "required": ["status", "summary", "criteria_coverage", "findings"],
}, REVIEW_NEXT_AGENT_NAMES))


def criterion_ids(requirement: Dict[str, Any]) -> Dict[str, str]:
    """{"C1": "<criterion text>", ...} in the requirement's own order."""
    return {f"C{i}": c for i, c in enumerate(requirement.get("acceptance_criteria", []), start=1)}


def run(
    root: Path,
    index: indexer.RepoIndex,
    ws: "workspace.Workspace",
    requirement: Dict[str, Any],
    architecture: Dict[str, Any],
    quality: Dict[str, Any],
    testing: Dict[str, Any],
    code_review: Dict[str, Any],
    human_note: Optional[str] = None,
    on_event: Optional[Callable[[], None]] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    criteria = criterion_ids(requirement)
    if not criteria:
        raise ValueError("QA Review Agent invoked with a requirement that has no acceptance criteria.")

    diff_text = workspace.diff_against_base(ws) or "(no differences from the base branch)"
    criteria_text = "\n".join(f"{cid}: {text}" for cid, text in criteria.items())
    code_review_context = {
        "status": code_review.get("status"),
        "summary": code_review.get("summary"),
        "follow_up_findings": [
            f.get("finding") for f in code_review.get("findings", []) if f.get("severity") == "FOLLOW_UP"
        ],
    }

    user_prompt = f"""
============================================================
ACCEPTANCE CRITERIA (assess every one -- no omissions)
============================================================

{criteria_text}

============================================================
DEFINITION OF DONE & BOUNDARIES (from the Architecture work order)
============================================================

Definition of done:
{chr(10).join(f"- {d}" for d in architecture.get("definition_of_done", [])) or "- (none stated)"}
Boundaries:
{chr(10).join(f"- {b}" for b in architecture.get("boundaries", [])) or "- (none stated)"}

============================================================
TEST FILES WRITTEN BY THE QUALITY STAGE
============================================================

{format_dict({"test_files": sorted((quality.get("test_files") or {}).keys())})}

============================================================
TEST RESULTS (already executed -- not your own run)
============================================================

{format_dict(testing)}

============================================================
CODE REVIEW OUTCOME (context only -- do not repeat it)
============================================================

{format_dict(code_review_context)}

============================================================
DIFF
============================================================

{diff_text[:20000]}
{format_note(human_note)}
Open the test files, assess every criterion, then provide your verdict.
"""

    agent_result = llm.run_agent(
        agent_name="QA Review Agent",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        output_schema=OUTPUT_SCHEMA,
        required_keys=["status", "summary", "criteria_coverage", "findings", "next_agent", "clarifying_questions"],
        cwd=root,
        tools=["Read", "Grep", "Glob"],
        can_use_tool=make_repo_scoped_permission(root),
        on_event=on_event,
        model=model,
    )

    result = agent_result.result
    _apply_coverage_gate(result, criteria)
    _validate(result)
    validate_clarifying_questions(result, "QA Review Agent")
    result["criteria"] = criteria
    result["_tool_calls"] = agent_result.tool_calls
    return result


def _apply_coverage_gate(result: Dict[str, Any], criteria: Dict[str, str]) -> None:
    """Normalizes criteria_coverage to exactly the real criterion ids
    and appends the deterministic findings (see module docstring)."""
    assessed = {}
    for entry in result.get("criteria_coverage") or []:
        cid = str(entry.get("criterion_id", "")).strip().upper()
        if cid in criteria and cid not in assessed:
            assessed[cid] = entry

    coverage: List[Dict[str, Any]] = []
    findings = result.setdefault("findings", [])
    existing_ids = {f.get("id") for f in findings}

    for cid, text in criteria.items():
        entry = assessed.get(cid)
        if entry is None:
            entry = {"criterion_id": cid, "status": "UNCOVERED", "evidence": "QA did not assess this criterion."}
        status = entry.get("status")
        evidence = str(entry.get("evidence") or "").strip()
        if status == "COVERED" and not evidence:
            status = "PARTIAL"
            evidence = "Marked covered but no test was cited."
        coverage.append({"criterion_id": cid, "criterion": text, "status": status, "evidence": evidence})

        if status in ("UNCOVERED", "PARTIAL"):
            finding_id = f"qa-coverage:{cid}"
            if finding_id in existing_ids:
                continue
            blocking = status == "UNCOVERED"
            findings.append({
                "id": finding_id,
                "severity": "BLOCKING" if blocking else "FOLLOW_UP",
                "finding": (
                    f"Acceptance criterion {cid} is not proven by any test: {text}"
                    if blocking
                    else f"Acceptance criterion {cid} is only partially proven: {text}"
                ),
                "evidence": evidence,
                "suggested_fix": "Add or strengthen a test that fails if this behavior breaks.",
            })
            existing_ids.add(finding_id)

    result["criteria_coverage"] = coverage


def _validate(result: Dict[str, Any]) -> None:
    if result.get("status") not in {"PASS", "CHANGES_REQUESTED"}:
        raise ValueError(f"QA Review Agent returned invalid status: {result.get('status')}")

    findings = result.get("findings", [])
    has_blocking = any(f.get("severity") == "BLOCKING" for f in findings)
    # Authoritative over the model's own status, in both directions --
    # same invariant as Code Review.
    result["status"] = "CHANGES_REQUESTED" if has_blocking else "PASS"

    if result["status"] == "CHANGES_REQUESTED" and result.get("next_agent") == "RELEASE_PLANNING":
        # The gate flipped a PASS after the model chose its next step.
        # A coverage gap is Quality's to fix; anything else is Coding's.
        coverage_gap = any(str(f.get("id", "")).startswith("qa-coverage:") and f.get("severity") == "BLOCKING"
                           for f in findings)
        result["next_agent"] = "QUALITY" if coverage_gap else "CODING"
        result["next_agent_reason"] = (
            "Overridden: blocking QA findings exist, so release is not available."
        )
