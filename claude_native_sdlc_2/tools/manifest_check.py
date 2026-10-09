"""
Deterministic check that the actual git diff only touches files the
Architecture Agent explicitly declared (files_to_modify/files_to_create/
files_to_delete).

No LLM call. Gives agents/review.py a signal to gate on the same way
tools/static_analysis.py already does for Semgrep/pip-audit -- a plain
set comparison has no false-positive rate to weigh, so an undeclared
file is treated as BLOCKING scope creep unconditionally, not handed to
the model as something to judge.

Based on the *idea* of bashebr/ai-native-sdlc's check_plan_sync.py
(plan.md vs diff comparison), not its implementation.
"""

from typing import Any, Dict, List, Optional


def run(
    architecture: Dict[str, Any],
    changed_files: List[str],
    extra_declared: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """extra_declared covers files authorized outside Architecture's own
    plan -- currently just Failure Analysis's dependency_fix_files (see
    agents/coding.py): a legitimate, narrowly-scoped fix the scope-creep
    check below must not treat as undeclared drift."""
    declared = set(architecture.get("files_to_modify") or [])
    declared |= set(architecture.get("files_to_create") or [])
    declared |= set(architecture.get("files_to_delete") or [])
    declared |= set(extra_declared or [])

    undeclared = sorted(set(changed_files) - declared)

    return {
        "declared_files": sorted(declared),
        "changed_files": sorted(changed_files),
        "undeclared_files": undeclared,
        "has_undeclared_files": bool(undeclared),
    }
