"""Shared run-log dumping, used by both cli.py and streamlit_app.py."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import config
import feedback.store as feedback_store
import workspace as workspace_module

# A plain "still working" pulse, in the spirit of Claude Code's own
# spinner status words -- not a transcript of what's actually
# happening, just a sign of life that changes every so often. Shared
# by cli.py's own ticker and streamlit_app.py's _status_ticker so
# both entry points show the same whimsical vocabulary.
STATUS_WORDS = [
    "Combobulating", "Accomplishing", "Percolating", "Noodling",
    "Conjuring", "Pondering", "Wrangling", "Synthesizing",
    "Marinating", "Spelunking", "Cogitating", "Tinkering",
    "Summoning", "Deliberating", "Herding", "Frolicking",
    "Ruminating", "Puzzling", "Musing", "Reticulating",
]


def dump_run_state(
    run_id: str,
    final_state: dict,
    ws,
    error: Optional[str] = None,
) -> Path:
    run_dir = config.RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_id,
        "repo": ws.repo_url,
        "base_branch": ws.base_branch,
        "working_branch": ws.working_branch,
        "run_outcome": final_state.get("run_outcome"),
        "error": error,
        "history": final_state.get("history", []),
    }
    (run_dir / "history.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    return run_dir


# ============================================================
# ROUTING VISUALIZATION
#
# With a fixed graph, the sequence of stages was always derivable
# from reading orchestrator/graph.py itself. Now that agents choose
# their own next_agent (see orchestrator/graph.py's resolve_next_node
# module docstring), the only ground truth for what a run ACTUALLY
# did is the order stages appear in `history` -- consecutive entries'
# "stage" values are exactly the edges resolve_next_node resolved to,
# including any case where it overrode an agent's own raw choice.
# These helpers turn that into something legible without reading
# history.json by hand.
# ============================================================

def stage_sequence_from_history(history: List[dict]) -> List[str]:
    return [entry["stage"] for entry in history if entry.get("stage")]


def stage_sequence_from_events(events: List[Tuple[str, dict]]) -> List[str]:
    """Same idea, for streamlit_app.py's st.session_state.completed_events
    (a list of (node_name, node_result) tuples) -- used to show the
    diagram live, before a run has finished and been dumped to
    history.json at all."""
    return [node_name for node_name, _ in events]


def looped_stages(sequence: List[str]) -> List[str]:
    """Stage names visited more than once, in first-repeat order --
    i.e. the stages the run actually looped back through."""
    seen = set()
    looped: List[str] = []
    for stage in sequence:
        if stage in seen and stage not in looped:
            looped.append(stage)
        seen.add(stage)
    return looped


def format_routing_text(sequence: List[str]) -> str:
    """Plain-text rendering for the CLI: an arrow chain with looped
    stages starred, plus a one-line hop/loop summary."""
    if not sequence:
        return "(no stages recorded)"

    looped = set(looped_stages(sequence))
    arrow = " -> ".join(f"{stage}*" if stage in looped else stage for stage in sequence)
    hops = len(sequence) - 1

    if looped:
        summary = f"{hops} hop(s); looped back through: {', '.join(sorted(looped))} (starred above)"
    else:
        summary = f"{hops} hop(s), no loops"

    return f"{arrow}\n  {summary}"


def record_human_rejection(
    run_id: str,
    ws,
    spec_text: str,
    final_state: dict,
    stage_name: str,
    reason: str = "",
) -> None:
    """
    A human-in-the-loop rejection stops the graph before it ever
    reaches finalize_node, so no feedback entry would otherwise be
    written for this run. Record one here so a rejected run is still
    visible to future runs against the same repo, same as any other
    outcome.
    """
    entry = {
        "run_id": run_id,
        "repo_key": feedback_store.repo_key(ws.owner, ws.name),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "spec_text": spec_text,
        "outcome": "REJECTED_BY_HUMAN",
        "iteration_count": final_state.get("iteration", 0),
        "files_touched": workspace_module.changed_files(ws),
        "review_findings_count": len(final_state.get("review", {}).get("findings", [])),
        "qa_findings_count": len((final_state.get("qa_review") or {}).get("findings", [])),
        "release": None,
        "rejected_at_stage": stage_name,
        "rejection_reason": reason,
    }
    feedback_store.append_run_summary(entry)
