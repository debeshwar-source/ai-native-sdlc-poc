"""
Design layer: turns the Architecture Agent's component sketch into two
comprehensible views of the repository -- the architecture as it is
today, and the architecture after the proposed change.

The model proposes the components (architecture-level modules/services/
layers, not files) and the relationships between them. Everything that
can be checked in code, is:

  - an existing component must cite files that really exist in the repo,
    or it is dropped (no invented components);
  - each component's change status (UNCHANGED / MODIFIED / NEW / REMOVED)
    is COMPUTED from Architecture's own files_to_modify / files_to_create
    / files_to_delete, never taken from the model -- so the "proposed"
    picture can't disagree with the plan Coding will actually execute;
  - any planned file the model failed to place in a component is added
    under a catch-all component, so every planned change is visible;
  - relationships to unknown components are dropped.

If the model gave no usable components, a deterministic fallback groups
the repo by top-level directory, so there is always a picture.

Pure functions, no I/O and no LLM calls: unit-testable, and normalize()
never raises (a design is a comprehension aid; it must not fail a run).
"""

import re
from typing import Any, Dict, Iterable, List, Optional

UNCHANGED, MODIFIED, NEW, REMOVED = "UNCHANGED", "MODIFIED", "NEW", "REMOVED"
MAX_COMPONENTS = 14
MAX_LABEL_CHARS = 40

# Fill / border per status. Dark text on light fills so the diagram
# stays readable on Streamlit's dark theme too.
_STYLE = {
    UNCHANGED: ('"#EEF1FB"', '"#4F46E5"', "filled", 1),
    MODIFIED: ('"#FDE9D9"', '"#D97706"', "filled", 2),
    NEW: ('"#DCFCE7"', '"#16A34A"', '"filled,dashed"', 2),
    REMOVED: ('"#FEE2E2"', '"#DC2626"', '"filled,dashed"', 2),
}

LEGEND = (
    "Blue = unchanged · Orange = modified · Green (dashed) = new · "
    "Red (dashed) = removed · Red arrow = new relationship"
)

DESIGN_SCHEMA = {
    "type": "object",
    "description": (
        "A comprehensible architecture-level picture of the repo: 4-12 "
        "components (modules/services/layers, NOT individual files) and "
        "how they depend on each other, covering both what exists today "
        "and what this change adds."
    ),
    "properties": {
        "existing_overview": {
            "type": "string",
            "description": "2-3 plain-language sentences: how the system is organised today.",
        },
        "proposed_overview": {
            "type": "string",
            "description": "2-3 plain-language sentences: what this change adds or alters, and why.",
        },
        "components": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "short slug, unique"},
                    "name": {"type": "string"},
                    "description": {"type": "string", "description": "one line: what it is / does"},
                    "files": {"type": "array", "items": {"type": "string"},
                              "description": "real repo paths belonging to it"},
                    "is_new": {"type": "boolean", "description": "true only if this change introduces it"},
                },
                "required": ["id", "name", "description", "files", "is_new"],
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "string", "description": "component id (the caller/dependent)"},
                    "to": {"type": "string", "description": "component id (the callee/dependency)"},
                    "label": {"type": "string", "description": "a few words: what flows or is called"},
                    "is_new": {"type": "boolean", "description": "true only if this change introduces it"},
                },
                "required": ["from", "to", "label", "is_new"],
            },
        },
    },
    "required": ["existing_overview", "proposed_overview", "components", "relationships"],
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-") or "component"


def _strs(value: Any) -> List[str]:
    return [str(v) for v in value if isinstance(v, str) and v] if isinstance(value, list) else []


def _short(text: Any, limit: int = MAX_LABEL_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _fallback_components(file_list: List[str], planned: set) -> List[Dict[str, Any]]:
    """One component per top-level directory (root-level files grouped
    as '(root)'). Used only when the model gave nothing usable."""
    groups: Dict[str, List[str]] = {}
    for path in file_list:
        top = path.split("/", 1)[0] if "/" in path else "(root)"
        groups.setdefault(top, []).append(path)
    ranked = sorted(groups.items(), key=lambda kv: (not (set(kv[1]) & planned), -len(kv[1])))
    return [
        {"id": _slug(name), "name": name, "description": f"{len(files)} file(s)", "files": files, "is_new": False}
        for name, files in ranked[:MAX_COMPONENTS]
    ]


def normalize(
    raw: Any, architecture: Dict[str, Any], file_list: Iterable[str]
) -> Optional[Dict[str, Any]]:
    """Cleans the model's design into the structure the renderers use.
    Returns None only if there is nothing at all to show."""
    try:
        return _normalize(raw, architecture or {}, list(file_list or []))
    except Exception:  # noqa: BLE001 -- a design is advisory, never fatal
        return None


def _normalize(raw: Any, architecture: Dict[str, Any], file_list: List[str]) -> Optional[Dict[str, Any]]:
    repo_files = set(file_list)
    modify = set(_strs(architecture.get("files_to_modify")))
    create = set(_strs(architecture.get("files_to_create")))
    delete = set(_strs(architecture.get("files_to_delete")))
    planned = modify | create | delete
    raw = raw if isinstance(raw, dict) else {}

    notes: List[str] = []
    components: List[Dict[str, Any]] = []
    seen_ids = set()
    dropped = 0

    for item in raw.get("components") or []:
        if not isinstance(item, dict) or not str(item.get("name") or "").strip():
            continue
        is_new = bool(item.get("is_new"))
        files = _strs(item.get("files"))
        if not is_new:
            verified = [f for f in files if f in repo_files]
            if not verified:
                dropped += 1  # an existing component with no real file behind it
                continue
            files = verified
        cid = _slug(item.get("id") or item.get("name"))
        while cid in seen_ids:
            cid += "-2"
        seen_ids.add(cid)
        components.append({
            "id": cid,
            "name": _short(item["name"], 60),
            "description": _short(item.get("description"), 160),
            "files": files,
            "is_new": is_new,
        })
    if dropped:
        notes.append(f"{dropped} component(s) cited no real file and were dropped.")

    source = "model"
    if not components:
        source = "derived"
        components = _fallback_components(file_list, planned)
        notes.append("No usable components from the model -- grouped by top-level directory instead.")
        if not components and not planned:
            return None

    # Statuses are computed from the plan, never trusted from the model.
    for comp in components:
        files = set(comp["files"])
        if comp["is_new"]:
            comp["status"] = NEW
        elif files and files <= delete:
            comp["status"] = REMOVED
        elif files & (modify | delete):
            comp["status"] = MODIFIED
        else:
            comp["status"] = UNCHANGED
        comp["changed_files"] = sorted(files & planned)

    # Every planned change must appear somewhere.
    covered = {f for c in components for f in c["files"]}
    leftover_changed = sorted((modify | delete) - covered)
    leftover_new = sorted(create - covered)
    # A single new component that named no files of its own is the
    # obvious home for the files this change creates.
    empty_new = [c for c in components if c["status"] == NEW and not c["files"]]
    if leftover_new and len(empty_new) == 1:
        empty_new[0]["files"] = leftover_new
        empty_new[0]["changed_files"] = leftover_new
        leftover_new = []
    if leftover_changed:
        components.append({"id": "other-changes", "name": "Other changed files", "description":
                           "Planned changes not grouped into a component above.",
                           "files": leftover_changed, "is_new": False, "status": MODIFIED,
                           "changed_files": leftover_changed})
    if leftover_new:
        components.append({"id": "new-files", "name": "New files", "description":
                           "Files this change creates.", "files": leftover_new, "is_new": True,
                           "status": NEW, "changed_files": leftover_new})

    if len(components) > MAX_COMPONENTS:
        order = {NEW: 0, MODIFIED: 1, REMOVED: 2, UNCHANGED: 3}
        components.sort(key=lambda c: (order[c["status"]], -len(c["files"])))
        components = components[:MAX_COMPONENTS]
        notes.append(f"Showing the {MAX_COMPONENTS} most relevant components.")

    ids = {c["id"] for c in components}
    new_ids = {c["id"] for c in components if c["status"] == NEW}
    relationships, seen_edges = [], set()
    for rel in raw.get("relationships") or []:
        if not isinstance(rel, dict):
            continue
        src, dst = _slug(rel.get("from")), _slug(rel.get("to"))
        if src not in ids or dst not in ids or src == dst or (src, dst) in seen_edges:
            continue
        seen_edges.add((src, dst))
        relationships.append({
            "from": src, "to": dst, "label": _short(rel.get("label")),
            # An edge touching a brand-new component can only be new.
            "is_new": bool(rel.get("is_new")) or src in new_ids or dst in new_ids,
        })

    return {
        "source": source,
        "existing_overview": _short(raw.get("existing_overview"), 600),
        "proposed_overview": _short(raw.get("proposed_overview"), 600),
        "components": components,
        "relationships": relationships,
        "notes": notes,
        "has_change": any(c["status"] != UNCHANGED for c in components),
    }


# ============================================================
# DOT RENDERING (Graphviz; Streamlit's st.graphviz_chart renders it
# client-side, no system `dot` needed)
# ============================================================

def _esc(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"')


def _node(comp: Dict[str, Any], status: str) -> str:
    fill, border, style, pen = _STYLE[status]
    prefix = "✕ " if status == REMOVED else ""
    return (
        f'  "{_esc(comp["id"])}" [label="{prefix}{_esc(comp["name"])}", fillcolor={fill}, '
        f'color={border}, style={style}, penwidth={pen}, tooltip="{_esc(comp["description"])}"];'
    )


def _header(name: str) -> List[str]:
    return [
        f"digraph {name} {{",
        '  rankdir="LR";',
        '  bgcolor="transparent";',
        '  nodesep=0.45; ranksep=0.7;',
        '  node [shape=box, style=filled, fontname="Helvetica", fontcolor="#111827", margin="0.18,0.10"];',
        '  edge [fontname="Helvetica", fontsize=10];',
    ]


def existing_dot(design: Dict[str, Any]) -> Optional[str]:
    """The architecture as it is TODAY: everything this change doesn't
    introduce, with the relationships that already exist."""
    comps = [c for c in design["components"] if c["status"] != NEW]
    if not comps:
        return None
    ids = {c["id"] for c in comps}
    lines = _header("existing")
    lines += [_node(c, UNCHANGED) for c in comps]
    for r in design["relationships"]:
        if not r["is_new"] and r["from"] in ids and r["to"] in ids:
            lines.append(
                f'  "{_esc(r["from"])}" -> "{_esc(r["to"])}" '
                f'[color="#9CA3AF", label="{_esc(r["label"])}", fontcolor="#9CA3AF"];'
            )
    lines.append("}")
    return "\n".join(lines)


def proposed_dot(design: Dict[str, Any]) -> Optional[str]:
    """The architecture AFTER the proposed change: the same picture,
    with modified/new/removed components and new relationships
    highlighted against the unchanged baseline, so the difference is
    visible in one glance."""
    if not design["components"]:
        return None
    lines = _header("proposed")
    lines += [_node(c, c["status"]) for c in design["components"]]
    status_of = {c["id"]: c["status"] for c in design["components"]}
    for r in design["relationships"]:
        removed = REMOVED in (status_of[r["from"]], status_of[r["to"]])
        if r["is_new"]:
            attrs = '[color="#DC2626", penwidth=2, fontcolor="#DC2626"'
        elif removed:
            attrs = '[color="#9CA3AF", style=dashed, fontcolor="#9CA3AF"'
        else:
            attrs = '[color="#9CA3AF", fontcolor="#9CA3AF"'
        lines.append(
            f'  "{_esc(r["from"])}" -> "{_esc(r["to"])}" {attrs}, label="{_esc(r["label"])}"];'
        )
    lines.append("}")
    return "\n".join(lines)
