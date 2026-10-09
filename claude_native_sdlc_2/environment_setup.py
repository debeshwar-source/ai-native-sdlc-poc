"""
Ties tools/test_runner.py's deterministic per-language install/test
detection to agents/environment.py's LLM fallback for anything it
doesn't recognize -- this is what lets the system work with any repo,
not just Python/Node/Go/Rust, the same way Claude Code itself has no
language whitelist and just figures out what a given repo needs.

tools/test_runner.py stays exactly what its own docstring says it is
("deterministic, no LLM calls") -- it never imports agents/
environment.py itself. This module is the one place that decides
WHEN to escalate to the agent (only once, only when nothing hardcoded
matched) and hands test_runner.py the result through its existing
public cache/execution primitives, never bypassing them.
"""

from pathlib import Path
from typing import Callable, Optional

import agents.environment as environment_agent
import context.indexer as indexer
import tools.test_runner as test_runner


def ensure_environment(
    root: Path,
    index: "indexer.RepoIndex",
    on_event: Optional[Callable[[], None]] = None,
    model: Optional[str] = None,
) -> dict:
    """Drop-in replacement for tools.test_runner.ensure_environment
    that also tries the Environment Agent when none of the hardcoded
    ecosystems (Python/Node/Go/Rust) matched at all. `on_event`, if
    given, pings on each step of the agent's exploration -- see
    llm.py's run_agent(on_event=...). `model`, if given, overrides
    config.MODEL_ID for the Environment Agent's own call only (see
    config.AGENT_NAMES)."""
    results = test_runner.ensure_environment(root)

    if results or test_runner.detect_test_command(root) is not None:
        # A hardcoded ecosystem installed dependencies, or a test
        # command is already detectable without any agent -- e.g. a
        # toy repo with test_*.py files but no real dependencies.
        # Nothing unrecognized here; leave the agent out of it.
        return results

    decision = environment_agent.run(root, index, on_event=on_event, model=model)
    ecosystem = decision.get("ecosystem", "unknown")

    if decision.get("test_command"):
        test_runner.set_fallback_test_command(root, decision["test_command"])

    if not decision.get("install_command"):
        results["environment_agent"] = {
            "status": "SKIPPED",
            "ecosystem": ecosystem,
            "test_command": decision.get("test_command"),
            "reason": decision.get("notes", ""),
        }
        return results

    install_result = test_runner.run_command(root, decision["install_command"])
    results["environment_agent"] = {
        "status": install_result["status"],
        "ecosystem": ecosystem,
        "install_command": " ".join(decision["install_command"]),
        "test_command": " ".join(decision.get("test_command", [])) or None,
        "error": install_result.get("error", ""),
        "reason": decision.get("notes", ""),
    }
    return results
