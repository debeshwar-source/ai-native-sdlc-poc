"""
Code Review Agent.

Agentic, read-only: reviews the actual diff (precomputed and handed
to it directly, rather than as an interactive tool call -- simpler
and just as accurate, since the diff is the same regardless of when
during the turn it's fetched) and surrounding code before raising
findings. Findings are split into BLOCKING (must be fixed before
release) and FOLLOW_UP (legitimate but non-blocking).

Also folds in tools/static_analysis.py's deterministic Semgrep/
pip-audit report: general security findings are given to the model as
context (informational -- that ruleset has a real false-positive
rate), while hardcoded secrets and known-vulnerable dependencies are
appended as BLOCKING findings unconditionally, the same
orchestrator-level-invariant pattern agents/testing.py uses for a
failed test run.

Also folds in tools/manifest_check.py's deterministic comparison of
the actual diff against Architecture's declared files_to_modify/
files_to_create/files_to_delete: any changed file Architecture never
declared is appended as a BLOCKING finding unconditionally, the same
way a static-analysis blocking finding is.
"""

from pathlib import Path
from typing import Callable, Any, Dict, Optional

import context.indexer as indexer
import llm
import workspace
from agents._common import (
    CLARIFYING_QUESTIONS_GUIDANCE_TEXT,
    REVIEW_NEXT_AGENT_NAMES,
    REVIEW_ROUTING_TEXT,
    format_dict,
    format_note,
    format_requirement,
    make_repo_scoped_permission,
    validate_clarifying_questions,
    with_clarifying_questions_schema,
    with_next_agent_schema,
)

SYSTEM_PROMPT = """
You are the Code Review Agent in a Claude-Native SDLC system.

Review the diff below against the requirement, architecture plan,
and test results. Use Read/Grep/Glob to inspect surrounding code when
the diff alone isn't enough context.

MOST IMPORTANT RULE: a finding describes something WRONG. It is not
a checklist entry, a confirmation, or a restatement of what the diff
or test results already show. If an acceptance criterion is
correctly satisfied, that is NOT a finding -- say nothing about it.
Never write a finding whose content is "X was done correctly" or "X
passed as expected" -- that is evidence for your summary, not a
defect. Only write a finding when you can complete the sentence "the
code does NOT do <specific required thing>" or "the code has this
specific problem: ...".

Classify every real defect as exactly one of:
- BLOCKING: a real correctness, security, or requirement-compliance
  problem that must be fixed before this can be released.
- FOLLOW_UP: a legitimate improvement (style, maintainability,
  future-proofing) that does NOT need to block this release. Record
  it, don't gate on it.

Do not invent requirements. Do not demand infrastructure or
refactors the requirement never asked for. The normal, expected
outcome when the implementation is sound is PASS with an EMPTY
findings list -- do not manufacture findings, blocking or otherwise,
just to seem thorough or to document that things went well.

When you are done, provide your final findings.
""" + REVIEW_ROUTING_TEXT + """
If status is PASS, choose RELEASE_PLANNING. If CHANGES_REQUESTED,
choose whichever agent actually owns the fix: CODING for an
implementation defect, QUALITY if the test itself is wrong or
missing coverage, or ARCHITECTURE if the underlying plan was wrong,
not just its implementation.
""" + CLARIFYING_QUESTIONS_GUIDANCE_TEXT

OUTPUT_SCHEMA = with_clarifying_questions_schema(with_next_agent_schema({
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["PASS", "CHANGES_REQUESTED"]},
        "summary": {"type": "string"},
        "findings": {
            "type": "array",
            "description": (
                "ONLY real defects. Leave this empty if the implementation "
                "is correct -- do not add entries that merely confirm "
                "something was done right."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "severity": {"type": "string", "enum": ["BLOCKING", "FOLLOW_UP"]},
                    "finding": {
                        "type": "string",
                        "description": (
                            "What is WRONG, stated as a problem (e.g. "
                            "'the /notes endpoint still returns 200 for "
                            "an empty body'). Never a confirmation that "
                            "something works."
                        ),
                    },
                    "evidence": {"type": "string"},
                    "suggested_fix": {"type": "string"},
                },
                "required": ["id", "severity", "finding", "evidence"],
            },
        },
    },
    "required": ["status", "summary", "findings"],
}, REVIEW_NEXT_AGENT_NAMES))


def run(
    root: Path,
    index: indexer.RepoIndex,
    ws: "workspace.Workspace",
    requirement: Dict[str, Any],
    architecture: Dict[str, Any],
    testing: Dict[str, Any],
    static_analysis: Optional[Dict[str, Any]] = None,
    manifest_check: Optional[Dict[str, Any]] = None,
    human_note: Optional[str] = None,
    on_event: Optional[Callable[[], None]] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    diff_text = workspace.diff_against_base(ws)
    if not diff_text.strip():
        diff_text = "(no differences from the base branch)"

    manifest_section = ""
    if manifest_check and manifest_check.get("has_undeclared_files"):
        manifest_section = f"""
============================================================
DECLARED-FILE-MANIFEST CHECK (deterministic, already run -- not your
own judgment call)
============================================================

The diff touches files Architecture never declared in
files_to_modify/files_to_create/files_to_delete:

{format_dict(manifest_check)}

These will be treated as BLOCKING scope creep regardless of your own
assessment -- you do not need to re-derive them, just be aware your
final findings list will include them either way.
"""

    static_analysis_section = ""
    if static_analysis and (static_analysis.get("security_findings") or static_analysis.get("blocking_findings")):
        static_analysis_section = f"""
============================================================
STATIC/SECURITY ANALYSIS (deterministic, already run -- Semgrep +
pip-audit, not your own judgment call)
============================================================

security_findings below are Semgrep's general security-audit/OWASP
rulesets -- these DO have a real false-positive rate; use the
surrounding code to judge whether each is a real defect before citing
it as a finding.

blocking_findings below (hardcoded secrets, known-vulnerable
dependencies) will be treated as BLOCKING regardless of your own
assessment -- you do not need to re-derive them, just be aware your
final findings list will include them either way.

{format_dict(static_analysis)}
"""

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
TEST RESULTS
============================================================

{format_dict(testing)}

============================================================
DIFF (all current changes against the base branch)
============================================================

{diff_text[:20000]}
{static_analysis_section}
{manifest_section}
{format_note(human_note)}
Read surrounding code as needed to evaluate the diff, then provide
your final findings.
"""

    agent_result = llm.run_agent(
        agent_name="Review Agent",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        output_schema=OUTPUT_SCHEMA,
        required_keys=["status", "summary", "findings", "next_agent", "clarifying_questions"],
        cwd=root,
        tools=["Read", "Grep", "Glob"],
        can_use_tool=make_repo_scoped_permission(root),
        on_event=on_event,
        model=model,
    )

    result = agent_result.result
    _apply_static_analysis_gate(result, static_analysis)
    _apply_manifest_gate(result, manifest_check)
    _validate(result)
    validate_clarifying_questions(result, "Review Agent")
    result["_tool_calls"] = agent_result.tool_calls
    result["static_analysis"] = static_analysis
    result["manifest_check"] = manifest_check
    return result


def _apply_static_analysis_gate(
    result: Dict[str, Any], static_analysis: Optional[Dict[str, Any]]
) -> None:
    """
    Appends a BLOCKING finding for every static-analysis finding in
    the near-zero-false-positive category (hardcoded secrets, known
    CVEs at the exact installed version) that the model didn't already
    surface itself -- this is what actually gates release_apply,
    via _validate below deriving `status` from the findings list, the
    same way agents/testing.py's own orchestrator invariant overrides
    the model's summary for a failed test run.
    """
    if not static_analysis or not static_analysis.get("blocking_findings"):
        return

    findings = result.setdefault("findings", [])
    existing_ids = {f.get("id") for f in findings}

    for item in static_analysis["blocking_findings"]:
        if "vulnerability_id" in item:
            finding_id = f"pip-audit:{item.get('vulnerability_id')}:{item.get('package')}"
            finding = {
                "id": finding_id,
                "severity": "BLOCKING",
                "finding": (
                    f"Known vulnerability {item.get('vulnerability_id')} in "
                    f"dependency {item.get('package')} {item.get('installed_version')}"
                ),
                "evidence": item.get("description", ""),
                "suggested_fix": (
                    f"Upgrade to {', '.join(item.get('fix_versions') or []) or 'a patched version'}."
                ),
            }
        else:
            finding_id = f"semgrep-secrets:{item.get('rule_id')}:{item.get('path')}:{item.get('line')}"
            finding = {
                "id": finding_id,
                "severity": "BLOCKING",
                "finding": f"Semgrep secrets rule matched: {item.get('rule_id')} in {item.get('path')}:{item.get('line')}",
                "evidence": item.get("message", ""),
                "suggested_fix": "Remove the hardcoded secret and load it from environment/config instead.",
            }

        if finding_id in existing_ids:
            continue
        findings.append(finding)
        existing_ids.add(finding_id)


def _apply_manifest_gate(
    result: Dict[str, Any], manifest_check: Optional[Dict[str, Any]]
) -> None:
    """
    Appends a BLOCKING finding for every file the diff touched that
    Architecture never declared in files_to_modify/files_to_create/
    files_to_delete -- a plain set comparison, so there's no
    false-positive rate to weigh, unlike Semgrep's general rulesets.
    This is what actually gates release_apply on undeclared scope
    creep, via _validate below deriving `status` from the findings
    list, the same way _apply_static_analysis_gate already gates on
    secrets/CVEs.
    """
    if not manifest_check or not manifest_check.get("undeclared_files"):
        return

    findings = result.setdefault("findings", [])
    existing_ids = {f.get("id") for f in findings}

    for path in manifest_check["undeclared_files"]:
        finding_id = f"manifest:{path}"
        if finding_id in existing_ids:
            continue
        findings.append(
            {
                "id": finding_id,
                "severity": "BLOCKING",
                "finding": (
                    f"'{path}' was changed but Architecture never declared "
                    f"it in files_to_modify/files_to_create/files_to_delete."
                ),
                "evidence": (
                    "Declared: "
                    f"{manifest_check.get('declared_files')}. Actually "
                    f"changed: {manifest_check.get('changed_files')}."
                ),
                "suggested_fix": (
                    "Revert this file, or have Architecture explicitly "
                    "declare it if the change is genuinely required."
                ),
            }
        )
        existing_ids.add(finding_id)


def _validate(result: Dict[str, Any]) -> None:
    if result.get("status") not in {"PASS", "CHANGES_REQUESTED"}:
        raise ValueError(
            f"Review Agent returned invalid status: {result.get('status')}"
        )

    findings = result.get("findings", [])
    has_blocking = any(f.get("severity") == "BLOCKING" for f in findings)

    # The orchestrator's invariant is authoritative over the LLM's own
    # summary, in both directions: a BLOCKING finding always means
    # changes are required, and the absence of one always means it
    # isn't, regardless of what the model's own `status` field said.
    result["status"] = "CHANGES_REQUESTED" if has_blocking else "PASS"

    if result["status"] == "CHANGES_REQUESTED" and result.get("next_agent") == "RELEASE_PLANNING":
        # _apply_static_analysis_gate can flip PASS to
        # CHANGES_REQUESTED after the model already chose next_agent
        # -- if it picked RELEASE_PLANNING believing it had passed,
        # that choice is now stale. CODING is the safe default target
        # for the BLOCKING findings the gate just appended.
        result["next_agent"] = "CODING"
        result["next_agent_reason"] = (
            "Overridden: a deterministic static-analysis finding forced "
            "CHANGES_REQUESTED after this agent's own next_agent choice "
            "was made."
        )
