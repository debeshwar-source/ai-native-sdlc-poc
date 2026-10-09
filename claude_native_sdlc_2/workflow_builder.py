"""Drag-and-drop workflow builder for the Streamlit UI.

A tiny custom component (components/workflow_builder/index.html, plain
HTML/JS, no build step). The catalog starts empty: the user creates
agents in its "+ Add agent" tab (name, job description, skills) and
drags them into the workspace, where they are ordered and given orders.
The component returns {"catalog": [...], "stages": [{"id", "orders"}]}.
The catalog is persisted to agent_catalog.json so agents survive
restarts.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit.components.v1 as components

import config
import run_log

_FRONTEND = Path(__file__).parent / "components" / "workflow_builder"

# The browser caches a component's HTML by URL, so after an edit it kept
# serving the old page until a hard refresh. Putting a hash of the file in
# the component's name changes the URL whenever the file changes.
_component = components.declare_component(
    "workflow_builder_" + hashlib.md5((_FRONTEND / "index.html").read_bytes()).hexdigest()[:8],
    path=str(_FRONTEND),
)

CATALOG_PATH = Path(__file__).parent / "agent_catalog.json"


def load_catalog() -> List[Dict[str, Any]]:
    try:
        data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        return [a for a in data if isinstance(a, dict) and a.get("id") and a.get("name")]
    except (OSError, ValueError):
        return []


def save_catalog(catalog: List[Dict[str, Any]]) -> None:
    CATALOG_PATH.write_text(json.dumps(catalog, indent=2), encoding="utf-8")


def workflow_builder(
    catalog: List[Dict[str, Any]],
    stages: List[Dict[str, Any]],
    edges: List[Dict[str, str]],
    *,
    key: str,
    disabled: bool = False,
    reset_token: int = 0,
    outputs: Optional[Dict[str, Any]] = None,
    memory: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, str]], Optional[Dict[str, Any]]]:
    """Render the builder; returns (catalog, stages, edges, command).

    The workspace is a canvas: `stages` are the agent nodes on it (id, the
    draft text of its composer, size, position) and `edges` the wires between
    them, {"from": agent id, "to": agent id}: the output of `from` is passed
    into the context of `to`. All three seed the component and are returned
    unchanged until the user edits something; bump `reset_token` to force it to
    re-seed from them. `outputs` is executor.snapshot(): each agent's
    conversation, rendered as a chat inside its node. `memory` is
    executor.memory_snapshot(): what the Orchestrator knows, shown when the
    user clicks the small Orchestrator button above the workspace. `command`
    is the user's last action, {"type": "send" | "stop" | "new" | "answer" |
    "clear_memory", "id", "text", "nonce", and for "answer" also "qid" and
    "answers"}, or None -- compare the nonce with the last one handled."""
    result = _component(
        catalog=catalog, stages=stages, edges=edges, models=config.AVAILABLE_MODELS,
        default_model=config.MODEL_ID, disabled=disabled, reset_token=reset_token,
        outputs=outputs or {}, memory=memory or {"knowledge": [], "history": []},
        status_words=run_log.STATUS_WORDS, key=key,
        default={"catalog": catalog, "stages": stages, "edges": edges},
    ) or {}
    return (result.get("catalog", catalog), result.get("stages", stages),
            result.get("edges", edges), result.get("command"))


def validate(stages: List[Dict[str, str]], catalog: List[Dict[str, Any]]) -> List[str]:
    ids = {a["id"] for a in catalog}
    if not stages:
        return ["Add at least one agent to the workspace."]
    return [f"'{s['id']}' is no longer in the catalog." for s in stages if s["id"] not in ids]


def to_run_config(stages: List[Dict[str, str]], catalog: List[Dict[str, Any]]):
    """(user_agents, custom_pipeline, stage_orders) for a pipeline state."""
    by_id = {a["id"]: a for a in catalog}
    user_agents = {
        s["id"]: {
            "name": by_id[s["id"]]["name"],
            "job_description": by_id[s["id"]].get("job_description", ""),
            "skills": by_id[s["id"]].get("skills", ""),
            "model": by_id[s["id"]].get("model") or "",
            "can_edit": bool(by_id[s["id"]].get("can_edit")),
        }
        for s in stages
    }
    orders = {s["id"]: s["orders"].strip() for s in stages if s.get("orders", "").strip()}
    return user_agents, [s["id"] for s in stages], orders

