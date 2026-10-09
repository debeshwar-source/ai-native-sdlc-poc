import html
import random
import re
import threading

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

import approval_ledger
import config
import design as design_module
import context.indexer as indexer
import atlassian
import environment_setup
import executor
import knowledge
import feedback.store as feedback_store
import orchestrator.graph as graph
import run_log
import workspace as workspace_module
import workflow_builder

st.set_page_config(page_title="Claude-Native SDLC", page_icon="⚙️", layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="stCodeBlock"] pre {
        font-size: 0.82rem;
    }

    /* Gradient title -- a small, tasteful signature rather than a
    flat black heading, matching the teal accent in .streamlit/config.toml. */
    h1 {
        background: linear-gradient(90deg, #0F766E 0%, #14B8A6 60%, #2DD4BF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700 !important;
    }

    /* Use the full screen width: ~2.5% gutters either side. */
    div[data-testid="stMainBlockContainer"], .block-container {
        padding-left: 2.5%;
        padding-right: 2.5%;
        padding-top: 1.5rem;
        max-width: 100%;
    }

    /* Chat bubbles: soft card look instead of Streamlit's flat default. */
    div[data-testid="stChatMessage"] {
        border-radius: 14px;
        border: 1px solid #E1EDEB;
        box-shadow: 0 1px 3px rgba(15, 118, 110, 0.06);
        padding: 4px 6px;
        margin-bottom: 6px;
    }

    /* Buttons: rounded with a subtle lift on hover. */
    div[data-testid="stButton"] button, div.stButton button {
        border-radius: 0.6rem;
        transition: transform 0.08s ease, box-shadow 0.08s ease;
    }
    div[data-testid="stButton"] button:hover, div.stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 8px rgba(15, 118, 110, 0.15);
    }

    /* Badges: a touch more presence than Streamlit's default. */
    span[data-testid="stBadge"] {
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    /* Sidebar expanders: slightly rounder, calmer border. */
    div[data-testid="stExpander"] {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# Display label per agent key (config.AGENT_NAMES) -- used by the
# sidebar's per-agent model pickers below. Distinct from NODE_DISPLAY
# further down: agent keys match agents/<name>.py module names, not
# orchestrator/graph.py's node names ("release" here is agents/
# release.py, invoked from the "release_planning" node; "environment"
# isn't a graph node at all -- it's the any-repo fallback agent
# environment_setup.py calls before the pipeline even starts).
AGENT_DISPLAY_NAMES = {
    "requirement": "Requirement",
    "architecture": "Architecture",
    "quality": "Quality",
    "coding": "Coding",
    "testing": "Testing",
    "failure_analysis": "Failure Analysis",
    "review": "Code Review",
    "qa_review": "QA Review",
    "release": "Release Planning",
    "environment": "Environment (any-repo fallback)",
}

# ============================================================
# RUN STATE -- whether a run is currently in progress.
# ============================================================

_run_active = st.session_state.get("run_active", False)

# ============================================================
# STAGE METADATA
# ============================================================

STAGE_META = [
    ("requirement", "Requirement", "📝"),
    ("architecture", "Architecture", "🏛️"),
    ("quality", "Quality", "✅"),
    ("coding", "Coding", "💻"),
    ("testing", "Testing", "🧪"),
    ("review", "Code Review", "🛡️"),
    ("qa_review", "QA Review", "🔬"),
    ("release_apply", "Release", "🚀"),
]

# Same accent per stage everywhere, so every rendering (progress
# strip, routing diagram, chat avatars) reads as the same system.
STAGE_ACCENT = {
    "requirement": "#1565C0",
    "architecture": "#AD1457",
    "quality": "#2E7D32",
    "coding": "#E65100",
    "testing": "#F9A825",
    "failure_analysis": "#455A64",
    "review": "#6A1B9A",
    "qa_review": "#00838F",
    "release_planning": "#283593",
    "release_apply": "#283593",
    "finalize": "#616161",
}

STAGE_ICON = {
    "requirement": "📝",
    "architecture": "🏛️",
    "quality": "✅",
    "coding": "💻",
    "testing": "🧪",
    "failure_analysis": "🔍",
    "review": "🛡️",
    "qa_review": "🔬",
    "release_planning": "🚀",
    "release_apply": "🚀",
    "finalize": "🏁",
}

STAGE_ALIASES = {
    "failure_analysis": "testing",
    "release_planning": "release_apply",
    "finalize": "release_apply",
}

NODE_DISPLAY = {
    "requirement": "Requirement",
    "architecture": "Architecture",
    "quality": "Quality",
    "coding": "Coding",
    "testing": "Testing",
    "failure_analysis": "Failure Analysis",
    "review": "Code Review",
    "qa_review": "QA Review",
    "release_planning": "Release Planning",
    "release_apply": "Release Apply",
    "finalize": "Finalize",
}

# What each node hands off to next, so the live status widget can name
# the in-flight step while it's still running (an agentic tool-use
# call can take a long time, and a stage only becomes renderable once
# it *finishes* -- without this, the UI has nothing to show while the
# next step is in progress). Mirrors orchestrator/graph.py's own
# build_graph() edges exactly, so this driver (which runs nodes one at
# a time instead of via the compiled graph's own .stream(), to allow a
# human-in-the-loop gate to pause between them) can never drift out of
# sync with the real routing logic: only failure_analysis and review
# have any routing autonomy at all (graph.resolve_next_node); every
# other stage is a fixed next step or a deterministic, data-driven
# router (graph.architecture_router / graph.testing_router).
NEXT_NODE_FN = {
    "requirement": lambda s: "architecture",
    "architecture": graph.architecture_router,
    "quality": lambda s: "coding",
    "coding": lambda s: "testing",
    "testing": graph.testing_router,
    "failure_analysis": graph.resolve_next_node,
    "review": graph.resolve_next_node,
    "qa_review": graph.resolve_next_node,
    "release_planning": lambda s: "release_apply",
    "release_apply": lambda s: "finalize",
    "finalize": lambda s: None,
}

def next_node(current: str, state: dict):
    """What runs after `current`: the user's own workflow order when one
    was built (graph.custom_next_node), otherwise the standard routing
    in NEXT_NODE_FN above."""
    if state.get("custom_pipeline"):
        return graph.custom_next_node(current, state)
    fn = NEXT_NODE_FN.get(current)
    return fn(state) if fn else None


# Wrapped with the same with_step_counter() build_graph() uses for
# these, so MAX_TOTAL_STEPS still applies when driven node-by-node
# here instead of via the compiled graph's own .stream(). Not applied
# to the fixed release_planning/release_apply/finalize tail, same as
# in build_graph() -- it isn't agent-routed and runs at most once.
NODE_FN = {
    "requirement": graph.with_step_counter(graph.requirement_node),
    "architecture": graph.with_step_counter(graph.architecture_node),
    "quality": graph.with_step_counter(graph.quality_node),
    "coding": graph.with_step_counter(graph.coding_node),
    "testing": graph.with_step_counter(graph.testing_node),
    "failure_analysis": graph.with_step_counter(graph.failure_analysis_node),
    "review": graph.with_step_counter(graph.review_node),
    "qa_review": graph.with_step_counter(graph.qa_review_node),
    "release_planning": graph.release_planning_node,
    "release_apply": graph.release_apply_node,
    "finalize": graph.finalize_node,
}


# ============================================================
# RENDER HELPERS
# ============================================================

# A plain "still working" pulse, in the spirit of Claude Code's own
# spinner status words -- not a transcript of what's actually
# happening, just a sign of life that changes every so often. Shared
# with cli.py via run_log.STATUS_WORDS so both entry points use the
# same vocabulary.
STATUS_WORDS = run_log.STATUS_WORDS


def _status_ticker(label_slot, prefix: str):
    """Returns a callback -- llm.py's run_agent calls it with no
    arguments, orchestrator/graph.py's per-stage wrapper calls it with
    a stage name, so it accepts (and ignores) anything -- that just
    cycles label_slot to a new random status word each time it's
    called, e.g. "**Architecture** — ✻ Spelunking…"."""

    def _tick(*_args) -> None:
        label_slot.markdown(f"**{prefix}** — ✻ {random.choice(STATUS_WORDS)}…")

    return _tick


class _WallClockTicker:
    """A second, independent source of ticks for the same label, on a
    plain wall-clock interval rather than waiting for the model to
    produce a tool call or narration. Some agents (Requirement,
    Testing's scorecard call, Release) never use tools and often
    finish in one quiet turn with no visible text block at all -- for
    those, llm.py's on_event might not fire even once before the call
    completes, so the whimsical word would never actually appear.
    Requirement runs first in every single run, so this was the very
    first thing making the feature look broken. A background thread
    (with Streamlit's ScriptRunContext explicitly attached, the
    documented way to update a placeholder from off the main thread)
    ticks the same label on its own cadence for as long as the node
    call is in flight, independent of whatever the model does."""

    def __init__(self, label_slot, prefix: str, interval: float = 0.6):
        self._tick = _status_ticker(label_slot, prefix)
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        ctx = get_script_run_ctx()
        if ctx is not None:
            add_script_run_ctx(self._thread, ctx)

    def _loop(self) -> None:
        while not self._stop_event.wait(self._interval):
            self._tick()

    def start(self) -> "_WallClockTicker":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()


def render_design(design) -> None:
    """The design layer: the architecture as it is today, and as
    proposed -- same components, with what changes highlighted."""
    if not design:
        st.caption("No architecture diagram available for this plan.")
        return

    st.markdown("**Design**")
    existing_tab, proposed_tab = st.tabs(["Existing architecture", "Proposed architecture"])
    status_label = {"UNCHANGED": "unchanged", "MODIFIED": "modified", "NEW": "new", "REMOVED": "removed"}

    def _components_table(comps) -> None:
        for c in comps:
            files = ", ".join(f"`{f}`" for f in (c["changed_files"] or c["files"])[:4])
            more = max(0, len(c["changed_files"] or c["files"]) - 4)
            st.markdown(
                f"- **{c['name']}** ({status_label[c['status']]}) — {c['description']}"
                + (f"  \n  {files}" + (f" +{more} more" if more else "") if files else "")
            )

    with existing_tab:
        if design.get("existing_overview"):
            st.write(design["existing_overview"])
        dot = design_module.existing_dot(design)
        if dot:
            st.graphviz_chart(dot, width="stretch")
        with st.expander("Components", expanded=False):
            _components_table([c for c in design["components"] if c["status"] != "NEW"])

    with proposed_tab:
        if design.get("proposed_overview"):
            st.write(design["proposed_overview"])
        if not design.get("has_change"):
            st.caption("No change proposed -- the architecture stays as it is.")
        dot = design_module.proposed_dot(design)
        if dot:
            st.graphviz_chart(dot, width="stretch")
            st.caption(design_module.LEGEND)
        with st.expander("Components", expanded=False):
            _components_table(design["components"])

    for note in design.get("notes", []):
        st.caption(f"ℹ️ {note}")


def render_pipeline(completed_node: str = None) -> None:
    """Live run-progress strip -- same three states as before
    (already-done / just-completed / still-pending), restyled as
    colored pills. Sits above the chat log as an at-a-glance header,
    same role a chat client's "typing.../online" status line plays."""
    completed_key = STAGE_ALIASES.get(completed_node, completed_node)
    seen = False
    pills = []
    stage_meta = (
        [(k, a["name"], "🧑‍💼") for k, a in _run_agents.items()] if _run_agents else STAGE_META
    )
    for key, label, icon in stage_meta:
        accent = STAGE_ACCENT.get(key, "#546E7A")
        if key == completed_key:
            style = f"background:{accent};color:#fff;font-weight:700;"
            pills.append(f'<span style="{style}border-radius:14px;padding:4px 12px;font-size:0.8rem;">{icon} {label}</span>')
            seen = True
        elif not seen:
            style = f"background:{accent}22;color:{accent};font-weight:600;"
            pills.append(f'<span style="{style}border-radius:14px;padding:4px 12px;font-size:0.8rem;">✓ {label}</span>')
        else:
            pills.append(f'<span style="background:#eceff1;color:#90a4ae;border-radius:14px;padding:4px 12px;font-size:0.8rem;">{label}</span>')

    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin:4px 0 10px;">{"".join(pills)}</div>',
        unsafe_allow_html=True,
    )


def render_routing_diagram(events) -> None:
    """The actual path this run took, stage by stage -- ground truth
    for what orchestrator/graph.py's resolve_next_node decided at
    each hop, not just what an agent itself asked for (see
    run_log.py's ROUTING VISUALIZATION section). A repeated stage
    means the run looped back through it; those get a small "×N"
    badge and a highlighted border so a loop is visible at a glance
    instead of having to read history.json by hand."""
    sequence = run_log.stage_sequence_from_events(events)
    if not sequence:
        st.caption("No stages recorded yet.")
        return

    looped = set(run_log.looped_stages(sequence))
    visit_number = {}
    chips = []
    for stage in sequence:
        visit_number[stage] = visit_number.get(stage, 0) + 1
        accent = STAGE_ACCENT.get(stage, "#546E7A")
        label = NODE_DISPLAY.get(stage, stage)
        is_loop = stage in looped
        border = f"2px solid {accent}" if is_loop else "1px solid transparent"
        badge = (
            f'<sup style="color:{accent};font-weight:700;">×{visit_number[stage]}</sup>'
            if is_loop else ""
        )
        chips.append(
            f'<span style="background:{accent}1a;color:{accent};border:{border};'
            f'border-radius:12px;padding:3px 10px;font-size:0.78rem;font-weight:600;'
            f'white-space:nowrap;">{label}{badge}</span>'
        )

    arrow = '<span style="color:#90a4ae;padding:0 2px;">→</span>'
    hops = len(sequence) - 1
    loop_note = (
        f" — looped back through: {', '.join(NODE_DISPLAY.get(s, s) for s in sorted(looped))}"
        if looped else ""
    )

    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:2px;margin:6px 0;">'
        f'{arrow.join(chips)}</div>'
        f'<div style="color:#90a4ae;font-size:0.75rem;">{hops} hop(s){loop_note}</div>',
        unsafe_allow_html=True,
    )


def render_tool_calls(tool_calls) -> None:
    if not tool_calls:
        st.caption("No tool calls recorded.")
        return
    lines = []
    for call in tool_calls:
        tool = call.get("tool", "?")
        tool_input = call.get("input", {})
        input_preview = ", ".join(f"{k}={v!r}" for k, v in tool_input.items())
        if len(input_preview) > 140:
            input_preview = input_preview[:140] + "…"
        lines.append(f"{tool}({input_preview})")
    st.code("\n".join(lines), language="text")


_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
MAX_DIFF_LINES = 600


def render_diff(diff_text: str) -> None:
    """A GitHub-PR-style colored diff -- green/added and red/removed
    line backgrounds with old/new line-number gutters -- instead of a
    plain st.code(diff_text, language="diff") block with bare +/-
    prefixes and no color. Handles a diff spanning multiple files
    (each "diff --git a/X b/Y" starts a new file section) the same as
    a single-file one, since that's the real shape git diff produces
    either way."""
    if not diff_text or not diff_text.strip():
        st.caption("(no differences)")
        return

    def _row(bg: str, old_no, new_no, text_color: str, content: str) -> str:
        old_cell = str(old_no) if old_no else ""
        new_cell = str(new_no) if new_no else ""
        return (
            f'<div style="display:flex;background:{bg};font-family:ui-monospace,'
            f'SFMono-Regular,Consolas,monospace;font-size:0.78rem;white-space:pre;'
            f'line-height:1.5;">'
            f'<div style="width:34px;flex-shrink:0;text-align:right;padding-right:6px;'
            f'color:#8b949e;user-select:none;">{old_cell}</div>'
            f'<div style="width:34px;flex-shrink:0;text-align:right;padding-right:6px;'
            f'color:#8b949e;user-select:none;border-right:1px solid rgba(140,140,140,.25);">'
            f'{new_cell}</div>'
            f'<div style="flex:1;padding-left:8px;color:{text_color};">{content}</div>'
            f"</div>"
        )

    lines = diff_text.splitlines()
    truncated = len(lines) > MAX_DIFF_LINES
    lines = lines[:MAX_DIFF_LINES]

    rows = []
    old_line = new_line = 0

    for raw_line in lines:
        if raw_line.startswith("diff --git"):
            match = re.match(r"diff --git a/(.+) b/(.+)$", raw_line)
            file_label = match.group(2) if match else raw_line
            border = "border-top:1px solid rgba(140,140,140,.3);" if rows else ""
            rows.append(
                f'<div style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace;'
                f'font-size:0.8rem;font-weight:700;padding:6px 8px;{border}">'
                f"{html.escape(file_label)}</div>"
            )
            continue
        if raw_line.startswith(("index ", "--- ", "+++ ")):
            continue  # redundant with the file-label header line above
        if raw_line.startswith("\\"):  # "\ No newline at end of file"
            rows.append(f'<div style="color:#8b949e;font-size:0.75rem;padding-left:44px;">{html.escape(raw_line)}</div>')
            continue

        hunk_match = _HUNK_HEADER_RE.match(raw_line)
        if hunk_match:
            old_line, new_line = int(hunk_match.group(1)), int(hunk_match.group(2))
            rows.append(
                f'<div style="background:rgba(56,139,253,.1);color:#58a6ff;'
                f'font-family:ui-monospace,SFMono-Regular,Consolas,monospace;'
                f'font-size:0.78rem;padding:2px 8px;">{html.escape(raw_line)}</div>'
            )
            continue

        if raw_line.startswith("+"):
            rows.append(_row("rgba(46,160,67,.15)", "", new_line, "#1a7f37", html.escape(raw_line)))
            new_line += 1
        elif raw_line.startswith("-"):
            rows.append(_row("rgba(248,81,73,.15)", old_line, "", "#cf222e", html.escape(raw_line)))
            old_line += 1
        else:
            rows.append(_row("transparent", old_line, new_line, "inherit", html.escape(raw_line) or "&nbsp;"))
            old_line += 1
            new_line += 1

    if truncated:
        rows.append(
            '<div style="padding:6px 8px;color:#8b949e;font-size:0.78rem;">'
            "… diff truncated, see the full patch file for the rest …</div>"
        )

    st.markdown(
        f'<div style="border:1px solid rgba(140,140,140,.3);border-radius:6px;'
        f'overflow-x:auto;margin:4px 0;">{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )


def render_findings(findings) -> None:
    for finding in findings:
        severity = finding.get("severity", "?")
        color = "red" if severity == "BLOCKING" else "orange"
        st.badge(f"{severity} · {finding.get('id', '')}", color=color)
        st.write(finding.get("finding", ""))
        evidence = finding.get("evidence", "")
        if evidence:
            st.caption(evidence)


def stage_label(node_name: str, node_result: dict, pass_number: int = 1) -> str:
    """The informative, final label for a completed stage -- the bold
    first line of its chat bubble. pass_number counts how many times
    this stage has run in this session (both the automated retry
    loops and human-requested revisions increment it); it's folded
    into the label as " — pass N" when > 1, except for coding, whose
    own iteration count already conveys this."""
    suffix = f" — pass {pass_number}" if pass_number > 1 and node_name != "coding" else ""

    if node_name in _run_agents:
        return f"{_run_agents[node_name]['name']}{suffix}"

    if node_name == "requirement":
        feature = node_result.get("requirement", {}).get("feature", "")
        return f"Requirement{suffix} — {feature}" if feature else f"Requirement{suffix}"

    if node_name == "architecture":
        arch = node_result.get("architecture", {})
        tag = "change required" if arch.get("change_required") else "no change required"
        return f"Architecture{suffix} — {tag}"

    if node_name == "quality":
        n = len(node_result.get("quality", {}).get("test_files", {}))
        return f"Quality{suffix} — {n} test file(s) touched"

    if node_name == "coding":
        iteration = node_result.get("iteration", 1)
        revised = f" (revised, pass {pass_number})" if pass_number > 1 else ""
        return f"Coding — iteration {iteration}{revised}"

    if node_name == "testing":
        status = node_result.get("testing", {}).get("test_result", {}).get("status", "UNKNOWN")
        return f"Testing{suffix} — {status}"

    if node_name == "failure_analysis":
        analysis = node_result.get("failure_analysis", {})
        return f"Failure Analysis{suffix} — {analysis.get('failure_type', 'UNKNOWN')}"

    if node_name == "review":
        review = node_result.get("review", {})
        findings = review.get("findings", [])
        blocking = sum(1 for f in findings if f.get("severity") == "BLOCKING")
        return f"Code Review{suffix} — {review.get('status', '?')} ({blocking} blocking)"

    if node_name == "qa_review":
        qa = node_result.get("qa_review", {})
        findings = qa.get("findings", [])
        blocking = sum(1 for f in findings if f.get("severity") == "BLOCKING")
        covered = sum(1 for c in qa.get("criteria_coverage", []) if c.get("status") == "COVERED")
        total = len(qa.get("criteria_coverage", []))
        return f"QA Review{suffix} — {qa.get('status', '?')} ({covered}/{total} criteria covered, {blocking} blocking)"

    if node_name == "release_planning":
        title = node_result.get("release_plan", {}).get("release_title", "")
        return f"Release Planning{suffix} — {title}" if title else f"Release Planning{suffix}"

    if node_name == "release_apply":
        return "Release Apply"

    if node_name == "finalize":
        return f"Finalize — {node_result.get('run_outcome', 'UNKNOWN')}"

    return NODE_DISPLAY.get(node_name, node_name)


def _pass_number(node_name: str, events) -> int:
    return sum(1 for n, _ in events if n == node_name)


def render_stage_body(node_name: str, node_result: dict) -> None:
    """Write one stage's detailed content. Must be called from inside
    an open st.chat_message(...) block, which supplies the avatar."""

    # ------------------------------------------------
    if node_name in _run_agents:
        result = node_result["history"][-1]["result"]
        st.write(result.get("summary", ""))
        if result.get("files_changed"):
            st.caption("Files changed: " + ", ".join(f"`{f}`" for f in result["files_changed"]))
        with st.expander("Full output", expanded=True):
            st.markdown(result.get("output", ""))
        with st.expander("Exploration", expanded=False):
            render_tool_calls(result.get("_tool_calls", []))

    # ------------------------------------------------
    elif node_name == "requirement":
        req = node_result.get("requirement", {})
        with st.expander("Details", expanded=False):
            st.json({k: v for k, v in req.items() if k != "_tool_calls"})

    # ------------------------------------------------
    elif node_name == "architecture":
        arch = node_result.get("architecture", {})
        if arch.get("change_required"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption("Modify")
                for f in arch.get("files_to_modify", []) or ["—"]:
                    st.write(f"`{f}`" if f != "—" else f)
            with col2:
                st.caption("Create")
                for f in arch.get("files_to_create", []) or ["—"]:
                    st.write(f"`{f}`" if f != "—" else f)
            with col3:
                st.caption("Delete")
                for f in arch.get("files_to_delete", []) or ["—"]:
                    st.write(f"`{f}`" if f != "—" else f)
        render_design(node_result.get("design"))
        with st.expander("Exploration & full plan", expanded=False):
            render_tool_calls(arch.get("_tool_calls", []))
            st.json({k: v for k, v in arch.items() if k != "_tool_calls"})

    # ------------------------------------------------
    elif node_name == "quality":
        quality = node_result.get("quality", {})
        test_files = quality.get("test_files", {})
        with st.expander("Test files & exploration", expanded=False):
            render_tool_calls(quality.get("_tool_calls", []))
            for path, content in test_files.items():
                st.caption(path)
                st.code(content, language="python")

    # ------------------------------------------------
    elif node_name == "coding":
        history = node_result.get("history", [])
        result = history[-1].get("result", {}) if history else {}
        files_written = result.get("files_written", {})
        files_deleted = result.get("files_deleted", [])
        diffs = result.get("diffs", {})
        if files_written:
            st.write(
                f"**{len(files_written)} file(s) written:** "
                + ", ".join(f"`{p}`" for p in files_written)
            )
        if files_deleted:
            st.write(
                f"**{len(files_deleted)} file(s) deleted:** "
                + ", ".join(f"`{p}`" for p in files_deleted)
            )

        # A per-file diff against the base branch, captured at the
        # moment this pass ran -- shows exactly which lines were
        # added/changed/removed, not just the resulting full file.
        for path in files_written:
            st.caption(path)
            diff_text = diffs.get(path, "")
            if diff_text.strip():
                render_diff(diff_text)
            else:
                st.code(files_written[path], language="python")
        for path in files_deleted:
            st.caption(f"{path} (deleted)")
            diff_text = diffs.get(path, "")
            if diff_text.strip():
                render_diff(diff_text)

        with st.expander("Exploration (tool calls)", expanded=False):
            render_tool_calls(result.get("_tool_calls", []))

    # ------------------------------------------------
    elif node_name == "testing":
        testing = node_result.get("testing", {})
        test_result = testing.get("test_result", {})
        st.caption(f"`{test_result.get('command')}`")
        if test_result.get("status") != "PASS":
            with st.expander("Test output", expanded=True):
                st.code(test_result.get("output", ""))
                st.code(test_result.get("error", ""))
        scorecard = testing.get("scorecard", {})
        if scorecard:
            st.write(f"**Go/No-Go:** {scorecard.get('go_no_go', '')}")
            st.caption(scorecard.get("quality_report", ""))

    # ------------------------------------------------
    elif node_name == "failure_analysis":
        analysis = node_result.get("failure_analysis", {})
        with st.expander("Diagnosis", expanded=False):
            render_tool_calls(analysis.get("_tool_calls", []))
            st.write(analysis.get("root_cause", ""))

    # ------------------------------------------------
    elif node_name == "review":
        review = node_result.get("review", {})
        findings = review.get("findings", [])

        # Declared-file-manifest check: shown on every run, not just
        # when it fails -- a clean run should visibly prove the
        # governance check ran, not just stay silent about it.
        manifest = review.get("manifest_check") or {}
        changed_count = len(manifest.get("changed_files", []))
        undeclared = manifest.get("undeclared_files") or []
        if changed_count == 0:
            st.badge("Scope check: no files changed", color="gray")
        elif undeclared:
            st.badge(f"Scope check: {len(undeclared)} undeclared file(s)", color="red")
        else:
            st.badge(
                f"Scope check: {changed_count}/{changed_count} changed files declared",
                color="green",
            )

        with st.expander("Findings & diff exploration", expanded=bool(findings)):
            render_tool_calls(review.get("_tool_calls", []))
            render_findings(findings)

    # ------------------------------------------------
    elif node_name == "qa_review":
        qa = node_result.get("qa_review", {})
        coverage = qa.get("criteria_coverage", [])
        findings = qa.get("findings", [])

        # Shown on every run, not just a failing one, so a clean pass
        # visibly proves each criterion was checked against a test.
        status_color = {"COVERED": "green", "PARTIAL": "orange", "UNCOVERED": "red"}
        for entry in coverage:
            st.badge(f"{entry.get('criterion_id')}: {entry.get('status')}",
                     color=status_color.get(entry.get("status"), "gray"))
            st.caption(f"{entry.get('criterion', '')} — {entry.get('evidence', '')}")

        with st.expander("Findings & test exploration", expanded=bool(findings)):
            render_tool_calls(qa.get("_tool_calls", []))
            render_findings(findings)

    # ------------------------------------------------
    elif node_name == "release_planning":
        plan = node_result.get("release_plan", {})
        with st.expander("Plan", expanded=False):
            st.json({k: v for k, v in plan.items() if k != "_tool_calls"})

    # ------------------------------------------------
    elif node_name == "release_apply":
        result = node_result.get("release_result", {})
        push = result.get("push", {})
        pr = result.get("pull_request", {})

        if result.get("commit_sha"):
            st.write(f"Commit `{result['commit_sha']}`")
        else:
            st.write("Nothing to commit.")

        if push.get("via_fork"):
            st.info(
                f"Direct push wasn't possible (read-only access) — "
                f"pushed to fork `{push['via_fork']}` instead."
            )

        push_status = push.get("status")
        push_color = {"PASS": "green", "SKIPPED": "gray"}.get(push_status, "red")
        st.badge(f"Push: {push_status}", color=push_color)
        if push_status not in ("PASS", "SKIPPED"):
            st.caption(push.get("error", push.get("reason", "")))

        if pr.get("status") == "PASS":
            st.success(f"Pull request opened: {pr.get('url')}")
        elif pr.get("status") != "SKIPPED":
            st.write(f"PR: {pr.get('status')} — {pr.get('reason', pr.get('error', ''))}")

        diff_text = result.get("diff")
        if diff_text:
            st.markdown("**Files changed**")
            st.caption(result.get("diff_stat", "").strip())
            render_diff(diff_text)

        with st.expander("Issue tracker / wiki artifacts", expanded=False):
            st.json({"issue": result.get("issue"), "wiki": result.get("wiki")})

    # ------------------------------------------------
    elif node_name == "finalize":
        st.write(f"Outcome: **{node_result.get('run_outcome', 'UNKNOWN')}**")
        st.caption("Routing path")
        render_routing_diagram(st.session_state.completed_events)

    # Only failure_analysis, review and qa_review ever set next_agent (the only 3
    # stages with any routing autonomy -- see orchestrator/graph.py's
    # module docstring) -- shown here rather than duplicated in just
    # those two branches above. Every other stage's own result has no
    # such field, so this is simply absent/falsy for them.
    next_agent = node_result.get("next_agent")
    if next_agent:
        st.caption(f"→ Chose **{next_agent}** next — {node_result.get('next_agent_reason', '')}")


def render_stage_message(node_name: str, node_result: dict, pass_number: int = 1) -> None:
    """One stage, as one chat bubble -- the chat-log equivalent of the
    old st.status(...) block."""
    with st.chat_message("assistant", avatar=STAGE_ICON.get(node_name, "🤖")):
        st.markdown(f"**{stage_label(node_name, node_result, pass_number)}**")
        render_stage_body(node_name, node_result)


# ============================================================
# RUN STATE MACHINE
#
# Streamlit reruns this whole script on every interaction (including
# an Approve/Suggest-changes/Override click, or a new chat message), so
# the pipeline is driven node by node directly (not via
# graph.build_graph().stream(), which can only advance forward) --
# this is what lets a human-in-the-loop gate send a stage back to
# revise its own last output. The current node name and accumulated
# stage history are kept in st.session_state across reruns: each
# script execution advances as far as it can go without approval,
# then re-renders the full chat log so far, so nothing already shown
# ever disappears.
# ============================================================

def _render_setup_summary(snapshot: dict) -> None:
    """The clone/setup message's content, MINUS the live dependency-
    install spinner (which only makes sense while that install is
    actually in progress) -- everything else is plain, replayable
    rendering over already-known data. Called once live, right after
    _start_new_run finishes installing dependencies, and then again on
    every later script execution (see the top-level replay below) so
    it persists in the conversation like everything else, instead of
    vanishing the instant the immediate post-setup st.rerun() fires."""
    ws = snapshot["ws"]
    index = snapshot["index"]
    agent_models = snapshot["agent_models"]
    env_result = snapshot["env_result"]
    install_deps = snapshot["install_deps"]

    st.markdown(f"**Setting up** — cloned `{ws.repo_url}` @ `{ws.base_branch}`")
    st.success(
        f"Cloned into `{ws.root}` on branch `{ws.working_branch}` — "
        f"{index.primary_language or 'unknown language'}, "
        f"{len(index.file_list)} files indexed."
    )

    with st.expander(
        "Indexed files (plain text)",
        expanded=False,
        key="setup_repo_structure_expander",
    ):
        st.text(index.tree_text)

    if agent_models:
        overrides = ", ".join(
            f"{AGENT_DISPLAY_NAMES.get(k, k)} → `{v}`" for k, v in agent_models.items()
        )
        st.caption(f"Model overrides for this run: {overrides}. Everything else: `{config.MODEL_ID}`.")

    if not install_deps:
        st.info(
            "Skipped dependency install (unchecked) — tests may fail on "
            "ModuleNotFoundError for reasons unrelated to the actual change."
        )

    any_dependency_failures = any(
        step["status"] == "FAIL"
        for result in env_result.values()
        for step in result.get("install_steps", [])
    ) or any(result.get("status") == "FAIL" for result in env_result.values())

    if any_dependency_failures:
        st.warning(
            "One or more dependencies failed to install. The pipeline "
            "will still run, but tests may fail on ModuleNotFoundError "
            "for reasons that have nothing to do with the actual code "
            "change. See details below."
        )

    with st.expander("Dependency setup", expanded=any_dependency_failures):
        if not env_result:
            st.caption(
                "Skipped by user."
                if not install_deps
                else "No recognized dependency manifest found; skipped install."
            )
        for ecosystem, result in env_result.items():
            badge_color = {"PASS": "green", "PARTIAL": "orange"}.get(
                result.get("status"), "red"
            )
            st.markdown(f"**{ecosystem}** :{badge_color}-badge[{result.get('status')}]")

            if ecosystem == "environment_agent":
                # No hardcoded Python/Node/Go/Rust manifest matched
                # -- the Environment Agent explored the repo itself
                # and decided what this is and how to build/test it.
                st.caption(f"Ecosystem detected: **{result.get('ecosystem')}**")
                if result.get("test_command"):
                    st.caption(f"Test command: `{result['test_command']}`")
                if result.get("reason"):
                    st.caption(result["reason"])
                if result.get("status") == "FAIL" and result.get("error"):
                    st.code(result["error"])
                continue

            for step in result.get("install_steps", []):
                step_color = {"PASS": "green", "PARTIAL": "orange"}.get(
                    step["status"], "red"
                )
                st.markdown(f"`{step['target']}` :{step_color}-badge[{step['status']}]")
                if step["status"] != "PASS" and step.get("error"):
                    st.code(step["error"])
            if result.get("status") == "FAIL" and result.get("error"):
                st.code(result["error"])


def _start_new_run(
    repo_url, base_branch, spec_text, human_in_the_loop, install_deps,
    github_token, cleanup_after, agent_models, workflow,
):
    user_agents, custom_pipeline, stage_orders = workflow_builder.to_run_config(
        workflow, st.session_state.agent_catalog
    )
    if github_token.strip():
        config.GITHUB_TOKEN = github_token.strip()

    run_id = graph.new_run_id()

    with st.chat_message("assistant", avatar="⚙️"):
        st.markdown(f"**Setting up** — cloning `{repo_url}` @ `{base_branch}`")
        try:
            ws = workspace_module.clone_and_branch(repo_url.strip(), base_branch.strip(), run_id)
        except RuntimeError as e:
            st.error(f"Clone failed: {e}")
            st.stop()

        index = indexer.build_repo_index(ws.root)

        if install_deps:
            with st.spinner("Installing dependencies (can take a while on first run)..."):
                env_status_slot = st.empty()
                env_result = environment_setup.ensure_environment(
                    ws.root, index,
                    on_event=_status_ticker(env_status_slot, "Installing dependencies"),
                    model=agent_models.get("environment"),
                )
                env_status_slot.empty()
        else:
            env_result = {}

        # Persisted, not just rendered here -- this whole "Setting up"
        # message is otherwise NEVER shown again: the chat log below
        # replays completed_events on every rerun, but this clone/setup
        # step happens once, right before the immediate st.rerun() that
        # kicks off the pipeline, and was previously rendered only in
        # this one script pass -- invisible in practice (there one
        # instant, gone the next). setup_summary makes it replay
        # exactly like everything else in the conversation.
        setup_summary = {
            "ws": ws,
            "index": index,
            "agent_models": agent_models,
            "env_result": env_result,
            "install_deps": install_deps,
        }
        st.session_state.setup_summary = setup_summary
        _render_setup_summary(setup_summary)

    st.session_state.run_started = True
    st.session_state.run_active = True
    st.session_state.run_id = run_id
    st.session_state.ws = ws
    st.session_state.spec_text = spec_text
    st.session_state.human_in_the_loop = human_in_the_loop
    st.session_state.cleanup_after = cleanup_after
    st.session_state.pipeline_state = {
        "run_id": run_id,
        "spec_text": spec_text,
        "workspace": ws,
        "repo_index": index,
        "iteration": 0,
        "failure_history": [],
        "stagnation_count": 0,
        "total_steps": 0,
        "history": [],
        "agent_models": agent_models,
        "stage_orders": stage_orders,
    }
    st.session_state.pipeline_state["user_agents"] = user_agents
    st.session_state.pipeline_state["knowledge_paths"] = knowledge.resolve_paths(
        st.session_state.get("knowledge_sources", [])
    )
    st.session_state.pipeline_state["custom_pipeline"] = custom_pipeline
    st.session_state.current_node = custom_pipeline[0]
    st.session_state.completed_events = []
    st.session_state.awaiting_approval = False
    st.session_state.aborted = False
    st.session_state.abort_reason = ""
    st.session_state.finished = False
    st.session_state.error = None


# ------------------------------------------------
# Workflow builder: drag agents into the order you want them to run and
# type orders into any of them. Locked while a run is in progress (the
# run keeps the workflow it started with). Left untouched, it's the
# standard pipeline (opt-in preset), routed exactly as before.
# ------------------------------------------------
if "agent_catalog" not in st.session_state:
    st.session_state.agent_catalog = workflow_builder.load_catalog()
if "workflow" not in st.session_state:
    st.session_state.workflow = []
if "edges" not in st.session_state:
    st.session_state.edges = []
if "workflow_reset" not in st.session_state:
    st.session_state.workflow_reset = 0

# ------------------------------------------------
# Knowledge base: a deliberately tiny button outside the workspace. Folders
# (picked with the OS folder browser, or typed), local locations and git
# repos agents may search as read-only reference material. Persisted to
# knowledge_base.json; see knowledge.py.
# ------------------------------------------------
if "knowledge_sources" not in st.session_state:
    st.session_state.knowledge_sources = knowledge.load()


def _kb_add(location: str) -> None:
    source = knowledge.classify(location)
    if source is None:
        st.session_state.kb_message = f"'{location}' is not a git URL or an existing folder."
    elif source in st.session_state.knowledge_sources:
        st.session_state.kb_message = "Already in the knowledge base."
    else:
        st.session_state.knowledge_sources.append(source)
        knowledge.save(st.session_state.knowledge_sources)
        st.session_state.kb_message = ""


_, _atl_col, _kb_col = st.columns([40, 1, 1])
with _kb_col:
    with st.popover("📚", help="Knowledge base"):
        st.markdown("**Knowledge base**")
        for _i, _src in enumerate(list(st.session_state.knowledge_sources)):
            _row, _rm = st.columns([8, 1])
            _row.caption(f"{'📁' if _src['kind'] == 'folder' else '🔗'} `{_src['location']}`")
            if _rm.button("✕", key=f"kb_rm_{_i}", help="Remove", disabled=_run_active):
                st.session_state.knowledge_sources.pop(_i)
                knowledge.save(st.session_state.knowledge_sources)
                st.rerun()
        if not st.session_state.knowledge_sources:
            st.caption("Nothing added yet.")
        if st.button("Browse for a folder…", key="kb_browse", disabled=_run_active):
            _picked = knowledge.browse_for_folder()
            if _picked:
                _kb_add(_picked)
                st.rerun()
        _typed = st.text_input(
            "Or paste a git repo URL or folder path", key="kb_typed", disabled=_run_active
        )
        if st.button("Add", key="kb_add", disabled=_run_active or not _typed.strip()):
            _kb_add(_typed)
            st.rerun()
        if st.session_state.get("kb_message"):
            st.warning(st.session_state.kb_message)

def _handle_command(cmd) -> bool:
    """Act on a command from a chat box (send / stop / new / answer / credentials / clear_memory).
    True if the page should do a full rerun afterwards."""
    if not cmd or cmd["nonce"] == st.session_state.get("handled_command_nonce"):
        return False
    st.session_state.handled_command_nonce = cmd["nonce"]
    agent_id, kind = cmd["id"], cmd["type"]
    if kind == "stop":
        executor.stop(agent_id)
        return False
    if kind == "new":
        executor.new_chat(agent_id)
        return True
    if kind == "answer":
        return executor.answer_question(agent_id, cmd.get("qid", ""), cmd.get("answers") or [])
    if kind == "credentials":
        return executor.answer_credentials(agent_id, cmd.get("qid", ""), cmd.get("values") or {})
    if kind == "clear_memory":
        executor.clear_memory()
        return True
    if kind == "send":
        agent = next((a for a in st.session_state.agent_catalog if a["id"] == agent_id), None)
        if agent:
            return executor.send(agent, cmd.get("text", ""))
    return False


# The builder is a fragment so it can refresh on its own once a second while
# any agent is working (live steps in each agent's chat) without rerunning the
# whole page; with nothing running it doesn't tick at all. Starting a command
# does a full rerun, which is what turns the ticking on; once everything has
# finished, one more full rerun turns it off.
@st.fragment(run_every=1.0 if executor.any_running() else None)
def _builder_fragment():
    executor.set_context(
        st.session_state.agent_catalog, st.session_state.edges, st.session_state.knowledge_sources
    )
    _catalog, st.session_state.workflow, st.session_state.edges, _cmd = workflow_builder.workflow_builder(
        st.session_state.agent_catalog,
        st.session_state.workflow,
        st.session_state.edges,
        key="workflow_builder",
        reset_token=st.session_state.workflow_reset,
        outputs=executor.snapshot(),
        memory=executor.memory_snapshot(st.session_state.knowledge_sources),
    )
    executor.set_context(
        st.session_state.agent_catalog, st.session_state.edges, st.session_state.knowledge_sources
    )
    if _catalog != st.session_state.agent_catalog:
        st.session_state.agent_catalog = _catalog
        workflow_builder.save_catalog(_catalog)
    if _handle_command(_cmd):
        st.rerun()
    _live = executor.any_running()
    if st.session_state.get("_builder_was_live") and not _live:
        st.session_state._builder_was_live = False
        st.rerun()
    st.session_state._builder_was_live = _live
    if st.button("Clear workspace"):
        for _w in st.session_state.workflow:
            executor.new_chat(_w["id"])
        st.session_state.workflow = []
        st.session_state.edges = []
        st.session_state.workflow_reset += 1
        st.rerun()


# ------------------------------------------------
# Jira & Confluence connection: a second tiny button. Agents created with
# use this account when they need Jira/Confluence (they ask first); see atlassian.py.
# ------------------------------------------------
with _atl_col:
    with st.popover("🔌", help="Jira & Confluence connection"):
        _creds = atlassian.load()
        st.markdown("**Jira & Confluence**")
        if atlassian.is_configured(_creds):
            st.caption(f"Connected account: `{_creds['email']}` on `{_creds['site']}`"
                       + (f" · project `{_creds['project_key']}`" if _creds["project_key"] else "")
                       + " · token saved ✓")
        else:
            st.caption("Not connected yet -- you don't have to do this ahead of time: an agent asks you for it "
                       "the moment you ask it to use Jira or Confluence. A link alone can't grant access, "
                       "though: Atlassian also needs your email and an API token.")
        _link = st.text_input(
            "Jira or Confluence link (any page on your site)",
            value=_creds["site"], key="atl_link",
            placeholder="https://your-team.atlassian.net/jira/software/projects/KEY/list",
        )
        _parsed = atlassian.parse_link(_link)
        _email = st.text_input("Atlassian account email", value=_creds["email"], key="atl_email")
        _token = st.text_input(
            "API token", type="password", key="atl_token",
            placeholder="leave blank to keep the saved token" if _creds["token"] else "paste your API token",
            help=f"Create one at {atlassian.TOKEN_URL}",
        )
        st.caption(f"Create a token here: {atlassian.TOKEN_URL}")
        _c1, _c2 = st.columns(2)
        if _c1.button("Save & test", key="atl_save", disabled=not (_parsed["site"] and _email.strip())):
            _tok = _token.strip() or _creds["token"]
            if not _tok:
                st.warning("Paste an API token.")
            else:
                atlassian.save(_parsed["site"], _email, _tok, _parsed["project_key"] or _creds["project_key"])
                with st.spinner("Checking with Atlassian…"):
                    st.session_state.atl_result = atlassian.test_connection()
        if _c2.button("Disconnect", key="atl_clear", disabled=not atlassian.is_configured(_creds)):
            atlassian.clear()
            st.session_state.pop("atl_result", None)
            st.rerun()
        for _prod, (_ok, _msg) in (st.session_state.get("atl_result") or {}).items():
            (st.success if _ok else st.error)(f"{_prod.title()}: {_msg}")

with st.expander("🧩 Workflow builder — create agents, drag them into the workspace, give them orders", expanded=True):
    _builder_fragment()

workflow = st.session_state.workflow
_wf_errors = workflow_builder.validate(workflow, st.session_state.agent_catalog)
if workflow:
    for _msg in _wf_errors:
        st.error(_msg)
if workflow and not _wf_errors and not _run_active:
    _names = {a["id"]: a["name"] for a in st.session_state.agent_catalog}
    st.caption("Workflow: " + " → ".join(_names[w["id"]] for w in workflow))

# User-defined agents of the run in progress (or just finished) get the
# same display plumbing the built-in stages have.
_run_agents = (st.session_state.get("pipeline_state") or {}).get("user_agents") or {}
for _id, _agent in _run_agents.items():
    NODE_DISPLAY[_id] = _agent["name"]
    STAGE_ICON[_id] = "🧑‍💼"
    STAGE_ACCENT[_id] = "#0F766E"

# ------------------------------------------------
# Live pipeline progress header, above the conversation.
# ------------------------------------------------
pipeline_strip = st.empty()

# ------------------------------------------------
# The conversation: the spec that kicked this run off, as a user
# message, followed by every stage's activity as it happens.
# ------------------------------------------------
if st.session_state.get("chat_spec_message"):
    with st.chat_message("user"):
        st.write(st.session_state.chat_spec_message)

if st.session_state.get("setup_summary"):
    with st.chat_message("assistant", avatar="⚙️"):
        _render_setup_summary(st.session_state.setup_summary)

if st.session_state.get("run_started"):
    # Replay everything already completed in a PREVIOUS script
    # execution (i.e. before this rerun) immediately, so nothing
    # already shown ever disappears across an Approve/Suggest-changes/
    # Override rerun or a new chat message.
    seen_counts = {}
    for node_name, node_result in st.session_state.completed_events:
        seen_counts[node_name] = seen_counts.get(node_name, 0) + 1
        render_stage_message(node_name, node_result, seen_counts[node_name])

    if st.session_state.completed_events:
        with pipeline_strip.container():
            render_pipeline(st.session_state.completed_events[-1][0])

    if st.session_state.run_active and not st.session_state.awaiting_approval:
        while True:
            current_node = st.session_state.current_node
            if current_node is None:
                st.session_state.finished = True
                st.session_state.run_active = False
                break

            running_label = NODE_DISPLAY.get(current_node, current_node)

            with st.chat_message("assistant", avatar=STAGE_ICON.get(current_node, "🤖")):
                label_slot = st.empty()
                label_slot.markdown(f"**{running_label}** — running…")

                # Two independent sources of the same "still working"
                # pulse: llm.py's run_agent calls on_event live whenever
                # Claude actually produces a tool call or narration, and
                # the wall-clock ticker below cycles the same label on a
                # plain timer regardless -- some agents (Requirement,
                # Testing's scorecard call, Release) never use tools and
                # often finish in one quiet turn with nothing for
                # on_event to ever fire on, so without the wall-clock
                # ticker they'd show a static "running…" the whole time.
                st.session_state.pipeline_state["on_event"] = _status_ticker(label_slot, running_label)
                wall_clock_ticker = _WallClockTicker(label_slot, running_label).start()
                try:
                    result = (
                        graph.node_fn_for(current_node, st.session_state.pipeline_state)
                        or NODE_FN[current_node]
                    )(st.session_state.pipeline_state)
                except Exception as e:
                    label_slot.markdown(f"**{running_label}** — ❌ error")
                    st.write(str(e))
                    st.session_state.error = str(e)
                    st.session_state.run_active = False
                    break
                finally:
                    wall_clock_ticker.stop()
                    st.session_state.pipeline_state.pop("on_event", None)

                st.session_state.pipeline_state.update(result)

                pending = st.session_state.pipeline_state.get("pending_human_note")
                if pending and pending.get("stage") == current_node:
                    st.session_state.pipeline_state.pop("pending_human_note", None)

                st.session_state.completed_events.append((current_node, result))
                pass_number = _pass_number(current_node, st.session_state.completed_events)

                label_slot.markdown(f"**{stage_label(current_node, result, pass_number)}**")
                render_stage_body(current_node, result)

            with pipeline_strip.container():
                render_pipeline(current_node)

            if st.session_state.human_in_the_loop and (
                current_node in graph.GATED_NODE_NAMES or current_node in _run_agents
            ):
                st.session_state.awaiting_approval = True
                break

            st.session_state.current_node = next_node(
                current_node, st.session_state.pipeline_state
            )

    # ------------------------------------------------
    # Approval gate, as an assistant message asking a question. The
    # clarificatory layer below is Human approval mode ONLY -- Agentic
    # mode and the CLI never read state["clarifying_questions"] at
    # all, so a question an agent surfaces there has zero effect;
    # every node still had to produce its own best-judgment output
    # regardless (see CLARIFYING_QUESTIONS_GUIDANCE_TEXT).
    # ------------------------------------------------
    if st.session_state.awaiting_approval:
        pending_name = st.session_state.current_node
        pending_result = (
            st.session_state.completed_events[-1][1]
            if st.session_state.completed_events
            else {}
        )
        questions = pending_result.get("clarifying_questions") or []
        gate_n = len(st.session_state.completed_events)
        no_preference = "No preference — go with the agent's own recommendation"

        with st.chat_message("assistant", avatar="⏸️"):
            if questions:
                st.markdown(
                    f"**Your call** — **{NODE_DISPLAY.get(pending_name, pending_name)}** "
                    f"flagged {len(questions)} decision it can't make for you. "
                    f"Answer below, add a note, or approve to go with its own "
                    f"recommendation. Only once you approve does the next agent run."
                )
            else:
                st.markdown(
                    f"**Review required** — reviewing "
                    f"**{NODE_DISPLAY.get(pending_name, pending_name)}** above. "
                    f"Approve to continue, answer/suggest changes to have this "
                    f"stage revise its own output before you're asked again, or "
                    f"override to send the run to a different agent."
                )

            answers = {}
            for i, q in enumerate(questions):
                options = list(q.get("options") or [])
                recommended = q.get("recommended_option")
                choice = st.radio(
                    q.get("question", f"Question {i + 1}"),
                    options + [no_preference],
                    index=len(options),  # default: no answer forced on the user
                    key=f"cq_{gate_n}_{i}",
                    format_func=lambda opt, rec=recommended: (
                        f"{opt} ✨ recommended" if opt == rec else opt
                    ),
                )
                if choice != no_preference:
                    answers[q.get("question", f"Question {i + 1}")] = choice

            note = st.text_area(
                "Additional notes (optional, leave blank to approve as-is)",
                key=f"note_{gate_n}",
                height=80,
            )
            gate_col1, gate_col2 = st.columns(2)
            approve_clicked = gate_col1.button(
                "Approve & continue",
                key=f"approve_{gate_n}",
                type="primary",
                use_container_width=True,
            )
            suggest_clicked = gate_col2.button(
                "Answer & continue" if questions else "Suggest changes",
                key=f"suggest_{gate_n}",
                use_container_width=True,
            )

            natural_next = next_node(pending_name, st.session_state.pipeline_state)
            override_targets = graph.override_targets_for(st.session_state.pipeline_state)
            default_target = natural_next if natural_next in override_targets else pending_name
            default_index = (
                override_targets.index(default_target) if default_target in override_targets else 0
            )

            with st.expander("Override: send this to a different agent", expanded=False):
                st.caption(
                    "Take the decision out of the pipeline's hands: choose which agent "
                    "runs next and tell it what to do. It runs with the state that "
                    "already exists, so a stage whose inputs haven't been produced yet "
                    "can't be picked."
                )
                override_target = st.selectbox(
                    "Next agent",
                    override_targets,
                    index=default_index,
                    format_func=lambda k: NODE_DISPLAY.get(k, k),
                    key=f"override_target_{gate_n}",
                )
                override_instruction = st.text_area(
                    "Instructions for that agent",
                    key=f"override_instruction_{gate_n}",
                    height=80,
                    placeholder=(
                        "e.g. Skip the new abstraction -- add the endpoint directly in "
                        "the existing router."
                    ),
                )
                blockers = graph.override_blockers(
                    override_target, st.session_state.pipeline_state
                )
                for reason_text in blockers:
                    st.error(reason_text)

                release_warning = (
                    graph.release_gate_warning(st.session_state.pipeline_state)
                    if override_target == "release_planning" and not blockers
                    else None
                )
                release_ack = True
                if release_warning:
                    st.warning(
                        f"{release_warning} Sending this to release planning skips the "
                        f"review gate that normally blocks it."
                    )
                    release_ack = st.checkbox(
                        "I understand and want to release anyway",
                        key=f"override_release_ack_{gate_n}",
                    )

                override_clicked = st.button(
                    "Override & continue",
                    key=f"override_{gate_n}",
                    use_container_width=True,
                    disabled=bool(blockers) or not release_ack,
                )

        repo_key = feedback_store.repo_key(
            st.session_state.ws.owner, st.session_state.ws.name
        )

        if approve_clicked:
            st.session_state.awaiting_approval = False
            approval_ledger.record(
                run_id=st.session_state.run_id,
                repo_key=repo_key,
                stage=pending_name,
                decision="APPROVE",
            )
            st.session_state.current_node = next_node(
                pending_name, st.session_state.pipeline_state
            )
            st.rerun()

        if suggest_clicked:
            note_parts = [f"Q: {q}\nA: {a}" for q, a in answers.items()]
            if note.strip():
                note_parts.append(f"Additional note: {note.strip()}")
            combined_note = "\n\n".join(note_parts)

            if not combined_note:
                st.warning(
                    "Answer at least one question or add a note, or click "
                    "Approve to continue as-is."
                )
            else:
                st.session_state.awaiting_approval = False
                approval_ledger.record(
                    run_id=st.session_state.run_id,
                    repo_key=repo_key,
                    stage=pending_name,
                    decision="SUGGEST_CHANGES",
                    note=combined_note,
                )
                st.session_state.pipeline_state["pending_human_note"] = {
                    "stage": pending_name,
                    "note": combined_note,
                }
                # current_node is left unchanged, so the next script
                # execution re-runs this same stage with the note.
                st.rerun()

        if override_clicked:
            if not override_instruction.strip():
                st.warning(
                    "Enter instructions for the agent you're sending this to, or "
                    "use Approve/Suggest changes instead if there's nothing to redirect."
                )
            else:
                st.session_state.awaiting_approval = False
                approval_ledger.record(
                    run_id=st.session_state.run_id,
                    repo_key=repo_key,
                    stage=pending_name,
                    decision="OVERRIDE",
                    note=override_instruction.strip(),
                    reason=(
                        f"Redirected to {override_target}"
                        + (f" despite: {release_warning}" if release_warning else "")
                    ),
                )
                st.session_state.current_node = override_target
                st.session_state.pipeline_state["pending_human_note"] = {
                    "stage": override_target,
                    "note": graph.override_note(pending_name, override_instruction.strip()),
                }
                # Jumping to a stage other than pending_name's natural next
                # is what makes this an override, not a revision --
                # _human_note_for() matches pending_human_note purely by
                # stage name, so whichever stage runs next picks it up.
                st.rerun()

    # ------------------------------------------------
    # Final result, once finished/aborted/errored.
    # ------------------------------------------------
    if st.session_state.finished or st.session_state.aborted or st.session_state.error:
        ws = st.session_state.ws
        final_state = st.session_state.pipeline_state

        run_dir = run_log.dump_run_state(
            st.session_state.run_id, final_state, ws, error=st.session_state.error
        )

        with st.chat_message("assistant", avatar="🏁" if not st.session_state.error else "❌"):
            if st.session_state.aborted:
                st.markdown("**Run stopped**")
                st.write(st.session_state.abort_reason)
            elif st.session_state.error:
                st.markdown("**Pipeline error**")
                st.write(st.session_state.error)
            else:
                outcome = final_state.get("run_outcome", "UNKNOWN")
                success_outcomes = {"APPROVED_RELEASED", "NO_CHANGE_VERIFIED"}
                icon = "✅" if outcome in success_outcomes else "⚠️"
                st.markdown(f"{icon} **{outcome}**")
                st.caption(f"{final_state.get('iteration', 0)} iteration(s)")

            st.caption(f"Workspace: `{ws.root}`")
            st.caption(f"Run log: `{run_dir / 'history.json'}`")

        if st.session_state.cleanup_after:
            workspace_module.cleanup(ws)
            st.caption("Workspace cleaned up.")


if _run_active and not st.session_state.get("run_active", False):
    # run_active flipped True -> False DURING this same script pass
    # (the pipeline just finished/errored/was rejected) -- the
    # sidebar and request box above were already rendered with
    # disabled=_run_active from before that happened, so they're
    # showing stale (locked) state. One more rerun re-reads
    # run_active fresh and re-enables them; the `and not
    # st.session_state.get(...)` guard means this fires exactly once,
    # not on every subsequent rerun.
    st.rerun()
