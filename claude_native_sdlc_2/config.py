"""Runtime configuration for the Claude-Native SDLC engine."""

import os
from pathlib import Path

# The Claude Agent SDK talks to the bundled Claude Code CLI, which
# authenticates the same way an interactive Claude Code session does
# (a Claude subscription login via `claude login`, or ANTHROPIC_API_KEY /
# ANTHROPIC_AUTH_TOKEN if set) -- no separate API key is required. See
# README.md for the auth prerequisite.
MODEL_ID = os.environ.get("CLAUDE_SDLC_MODEL_ID", "claude-sonnet-5")

# Models selectable per agent (see AGENT_NAMES below) in the Streamlit
# sidebar's "Model per agent" expander and the CLI's --agent-model
# flag -- the current Claude model roster available via a Claude Code
# subscription login, same as MODEL_ID above.
AVAILABLE_MODELS = [
    "claude-sonnet-5",
    "claude-opus-5-5",
    "claude-haiku-4-5-20251001",
    "claude-fable-5-1",
]

# The 10 agents whose model is independently choosable, keyed by the
# same lowercase name each agents/<name>.py module is named after
# (NOT orchestrator/graph.py's node names -- "release" here is
# agents/release.py, invoked from the "release_planning" node).
AGENT_NAMES = [
    "requirement", "architecture", "quality", "coding",
    "testing", "failure_analysis", "review", "qa_review", "release", "environment",
]

# Agentic turns (architecture/quality/coding/review/failure_analysis) are
# capped to bound cost and prevent a runaway agent from looping forever.
MAX_AGENT_TURNS = 20

# Pipeline-level retry/stagnation bounds (coding <-> testing loop).
MAX_ITERATIONS = 6
MAX_STAGNANT_ITERATIONS = 2

# Where repos get cloned and where run artifacts/feedback are kept.
WORKSPACE_ROOT = Path(
    os.environ.get(
        "CLAUDE_SDLC_WORKSPACE_ROOT",
        str(Path(__file__).resolve().parent / "workspaces"),
    )
)
# One persistent local clone per repo, reused (via `git clone --reference`)
# across runs so a repo already seen is fetched incrementally instead of
# re-downloaded from GitHub every time.
REPO_CACHE_ROOT = Path(
    os.environ.get(
        "CLAUDE_SDLC_REPO_CACHE_ROOT",
        str(Path(__file__).resolve().parent / ".repo_cache"),
    )
)
RUNS_ROOT = Path(
    os.environ.get(
        "CLAUDE_SDLC_RUNS_ROOT",
        str(Path(__file__).resolve().parent / "runs"),
    )
)
FEEDBACK_DB_PATH = Path(
    os.environ.get(
        "CLAUDE_SDLC_FEEDBACK_DB",
        str(Path(__file__).resolve().parent / "feedback" / "history.jsonl"),
    )
)

# One continuously-growing hash chain of every HITL gate decision
# across every run (see approval_ledger.py) -- kept separate from
# per-run history.json since its tamper-evidence comes from being one
# long chain, not fragmented per run.
APPROVAL_LEDGER_PATH = Path(
    os.environ.get(
        "CLAUDE_SDLC_APPROVAL_LEDGER",
        str(Path(__file__).resolve().parent / "runs" / "approval_ledger.jsonl"),
    )
)

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
GITHUB_API_URL = "https://api.github.com"
