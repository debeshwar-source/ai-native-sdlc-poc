"""
The Claude-Native SDLC pipeline, wired as a LangGraph state machine.

FIXED sequence for most of the run:

    requirement -> architecture -> [change_required?]
                                       |-> quality -> coding -> testing
                                       '-> testing (no-change verification)
                   testing -> [status?]
                                |-> PASS + change_required -> review (code review)
                                |                                  -> qa_review
                                |-> PASS + no change      -> finalize
                                '-> FAIL                   -> failure_analysis

Two reviewers gate release, in order: `review` (Code Review -- is the
code correct, safe, in scope?) then `qa_review` (QA Review -- does the
test evidence prove every acceptance criterion?). Both route through
the same resolve_next_node, and release is unreachable until BOTH have
passed.

Only failure_analysis, review and qa_review have any routing autonomy at all --
after a test failure, whether the application, the test, or the plan
is at fault is a genuine judgment call; same for whether a diff is
actually ready to release. Every other stage's own next step is a
deterministic fact (a data field, not an LLM's own choice), computed
by architecture_router/testing_router below or by a fixed add_edge.

This replaces an earlier design where all 7 pre-release stages chose
their own next_agent freely among all 9 possible targets -- in
practice that could oscillate across many stages (e.g. review <->
architecture) for a long time before the overall step cap ever caught
it. resolve_next_node() below is now the router ONLY failure_analysis
and review share; each of THEIR OWN output schemas already restricts
next_agent to a small, sane enum (agents/_common.py's
FAILURE_ANALYSIS_NEXT_AGENT_NAMES / REVIEW_NEXT_AGENT_NAMES) -- review
can't even express "send this to failure_analysis" or vice versa. What
resolve_next_node still enforces on top of that narrower schema:

  - release_planning (and therefore release_apply) is unreachable
    without BOTH review's and qa_review's status actually being PASS.
  - coding is unreachable with nothing for it to act on.
  - the retry loop's iteration/stagnation caps still force a stop.
  - an overall step cap guarantees the run terminates regardless.

release_planning -> release_apply -> finalize stays a fixed sequence:
applying an already-approved release plan is mechanical execution,
not a judgment call for an agent to route around.
"""

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

import agents.architecture as architecture_agent
import agents.coding as coding_agent
import agents.custom as custom_agent
import agents.failure_analysis as failure_analysis_agent
import agents.qa_review as qa_review_agent
import agents.quality as quality_agent
import agents.release as release_agent
import agents.requirement as requirement_agent
import agents.review as review_agent
import agents.testing as testing_agent
import config
import context.indexer as indexer
import feedback.store as feedback_store
import tools.github_api as github_api
import tools.issue_tracker as issue_tracker
import tools.manifest_check as manifest_check
import tools.static_analysis as static_analysis
import workspace as workspace_module

MAX_ITERATIONS = config.MAX_ITERATIONS
MAX_STAGNANT_ITERATIONS = config.MAX_STAGNANT_ITERATIONS

# Hard backstop on total agent hops in a single run, independent of
# MAX_ITERATIONS (which only counts genuine coding attempts). Dynamic
# routing means, say, review <-> architecture could in principle
# oscillate without ever touching coding at all -- this guarantees the
# run still terminates regardless of which agents get visited in what
# order. Generous: legitimate bounded back-and-forth stays well under
# it; only a genuinely runaway routing loop hits it.
MAX_TOTAL_STEPS = 40

# Nodes that involve an agent producing a judgment worth a human
# checkpoint, when human-in-the-loop mode is on. release_apply and
# finalize are deliberately excluded -- they only execute a decision
# (release_planning's plan) that was already gated one step earlier.
GATED_NODE_NAMES = {
    "requirement",
    "architecture",
    "quality",
    "coding",
    "testing",
    "failure_analysis",
    "review",
    "qa_review",
    "release_planning",
}

# Every node name either failure_analysis or review could name via
# next_agent, keyed by the enum value their own OUTPUT_SCHEMA uses.
# Each agent's schema only allows a subset of these (see
# agents/_common.py's FAILURE_ANALYSIS_NEXT_AGENT_NAMES /
# REVIEW_NEXT_AGENT_NAMES) -- this dict just translates whichever
# value comes back into the matching graph node name.
NODE_BY_AGENT_NAME = {
    "ARCHITECTURE": "architecture",
    "QUALITY": "quality",
    "CODING": "coding",
    "FAILURE_ANALYSIS": "failure_analysis",
    "REVIEW": "review",
    # Not offered in any agent's next_agent enum (review's PASS is
    # redirected here by resolve_next_node's release invariant), but
    # it must be a valid graph target.
    "QA_REVIEW": "qa_review",
    "RELEASE_PLANNING": "release_planning",
    "FINALIZE": "finalize",
}

# The targets that represent "take another attempt" rather than a
# terminal/release choice -- the iteration/stagnation caps apply to
# these once the run is past its first pass (iteration > 0 means a
# genuine coding attempt already happened at least once). Deliberately
# excludes release_planning/finalize (not retries) and testing
# (testing_router itself is the one place that already applies these
# same caps before ever reaching failure_analysis).
RETRY_LOOP_NODES = {"architecture", "quality", "coding"}


class SDLCState(TypedDict, total=False):
    run_id: str
    spec_text: str
    workspace: Any
    repo_index: Any

    requirement: dict
    architecture: dict
    design: dict
    change_required: bool
    quality: dict
    testing: dict
    failure_analysis: dict
    review: dict
    qa_review: dict
    release_plan: dict
    release_result: dict

    # Set by the Streamlit UI's human-in-the-loop mode when a reviewer
    # asks a stage to revise its own last output, e.g.
    # {"stage": "coding", "note": "..."}. A node consumes this via
    # _human_note_for only when it matches that node's own stage name;
    # the CLI (no HITL) never sets this, so it's always None there and
    # every node behaves exactly as it did before this field existed.
    pending_human_note: dict

    iteration: int
    failure_history: List[dict]
    stagnation_count: int
    run_outcome: str
    history: List[dict]

    # Set ONLY by failure_analysis/review (the only 2 agents with any
    # routing autonomy -- see module docstring) -- their own choice of
    # what should run next, and why. Read by resolve_next_node. Every
    # other node's own next step is a fixed edge or a deterministic
    # router (architecture_router/testing_router), so this is simply
    # absent/stale after any other node runs.
    next_agent: str
    next_agent_reason: str
    # The most recently completed node's own clarifying_questions (see
    # agents/_common.py's with_clarifying_questions_schema) -- read
    # ONLY by streamlit_app.py's approval gate in Human approval mode.
    # Agentic mode and the CLI never look at this field.
    clarifying_questions: List[dict]
    # Hard backstop counting every node visited so far, regardless of
    # which ones -- see MAX_TOTAL_STEPS.
    total_steps: int

    # Optional "still working" ping for a UI status indicator:
    # on_event(stage), called each time Claude produces a tool call or
    # a piece of narration inside that stage's agent call -- no detail
    # about what happened, just that something did. See llm.py's
    # run_agent(on_event=...) and _stage_event_callback below. Absent
    # for the CLI's plain graph.stream() and any run that doesn't set
    # it, so every node's own behavior is unchanged either way.
    on_event: Callable[[str], None]

    # Per-agent Claude model override for this run, keyed by
    # config.AGENT_NAMES (e.g. {"coding": "claude-opus-5-5"}) -- set
    # once at run start (Streamlit's sidebar / cli.py's --agent-model)
    # and read uniformly via _model_for. An agent missing from this
    # dict (or an empty/absent dict entirely) falls back to
    # config.MODEL_ID, so a run that never sets this behaves exactly
    # as it did before this field existed.
    agent_models: Dict[str, str]

    # User-built workflow (Streamlit "Workflow builder" / cli.py
    # --stages): the ordered node names to run, and standing "orders"
    # typed into individual agents, keyed by node name. Both absent on
    # a default run -- custom_pipeline absent means the standard
    # routing above applies unchanged; stage_orders is independent of
    # it and applies to any run. See custom_next_node / _note_for.
    custom_pipeline: List[str]
    stage_orders: Dict[str, str]
    # User-defined agents (workflow builder's "Add agent" tab), keyed by
    # the node id used in custom_pipeline / stage_orders:
    # {id: {name, job_description, skills, model, can_edit}}. When present the
    # run is a plain linear chain of these -- see user_agent_node.
    user_agents: Dict[str, dict]
    # Local directories of the knowledge base (knowledge.resolve_paths):
    # read-only reference material every user-defined agent may search.
    knowledge_paths: List[str]


def _human_note_for(stage: str, state: SDLCState) -> Optional[str]:
    pending = state.get("pending_human_note")
    if pending and pending.get("stage") == stage:
        return pending.get("note")
    return None


def _note_for(stage: str, state: SDLCState) -> Optional[str]:
    """What an agent actually receives as its note: the operator's
    standing orders for this stage (typed into its card in the workflow
    builder; applied on every pass through the stage) followed by any
    one-off human-in-the-loop note. Kept separate from _human_note_for,
    which coding_node still uses on its own to tell a human revision
    apart from a genuine retry attempt."""
    orders = ((state.get("stage_orders") or {}).get(stage) or "").strip()
    human = _human_note_for(stage, state)
    parts = []
    if orders:
        parts.append(f"Standing orders for this stage: {orders}")
    if human:
        parts.append(human)
    return "\n\n".join(parts) or None


def _model_for(state: SDLCState, agent_key: str) -> str:
    return (state.get("agent_models") or {}).get(agent_key) or config.MODEL_ID


def _stage_event_callback(state: SDLCState, stage: str) -> Optional[Callable[[], None]]:
    """Binds the state's generic on_event(stage) ping (if any) to this
    one node's own stage name, giving each agent call the plain
    no-arg on_event() callback llm.run_agent expects."""
    raw = state.get("on_event")
    if raw is None:
        return None
    return lambda: raw(stage)


def append_history(state: SDLCState, entry: dict) -> List[dict]:
    history = list(state.get("history", []))
    history.append(entry)
    return history


def _can_retry(state: SDLCState) -> bool:
    return state.get("iteration", 0) < MAX_ITERATIONS


def _failure_signature(test_result: dict) -> str:
    payload = f"{test_result.get('status')}|{test_result.get('return_code')}|{(test_result.get('error') or '')[:500]}|{(test_result.get('output') or '')[-500:]}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _next_agent_fields(result: dict) -> dict:
    """Every dynamically-routed agent's result carries these two keys
    (agents/_common.py's with_next_agent_schema) -- hoisted into every
    node's own returned state dict so resolve_next_node can read
    state["next_agent"] uniformly regardless of which node just ran."""
    return {
        "next_agent": result.get("next_agent"),
        "next_agent_reason": result.get("next_agent_reason"),
    }


def _clarifying_questions_field(result: dict) -> dict:
    """Every human-gated agent's result carries clarifying_questions
    (agents/_common.py's with_clarifying_questions_schema) -- hoisted
    into every node's own returned state dict, the same way
    _next_agent_fields hoists next_agent, so streamlit_app.py's
    approval gate can read state["clarifying_questions"] uniformly
    regardless of which node just ran. Read ONLY in Human approval
    mode; Agentic mode never looks at this field, so it has no effect
    there."""
    return {"clarifying_questions": result.get("clarifying_questions") or []}


# ============================================================
# NODES
# ============================================================

def requirement_node(state: SDLCState):
    ws = state["workspace"]
    index = state["repo_index"]
    key = feedback_store.repo_key(ws.owner, ws.name)
    prior_runs = feedback_store.load_recent_summaries(key, state["spec_text"])

    requirement = requirement_agent.run(
        state["spec_text"],
        indexer.repo_overview_text(index),
        prior_runs,
        _note_for("requirement", state),
        _stage_event_callback(state, "requirement"),
        _model_for(state, "requirement"),
    )

    return {
        "requirement": requirement,
        "history": append_history(
            state,
            {"stage": "requirement", "status": "DONE", "result": requirement},
        ),
        **_clarifying_questions_field(requirement),
    }


def architecture_node(state: SDLCState):
    if state.get("iteration", 0) > 0:
        # A prior Coding Agent attempt may have left uncommitted writes
        # in the working tree; architecture must reason about the real
        # base state, not a partial attempt from a diagnosis that
        # implicated the architecture decision itself.
        workspace_module.reset_to_base(state["workspace"])

    architecture = architecture_agent.run(
        state["workspace"].root,
        state["repo_index"],
        state["requirement"],
        _note_for("architecture", state),
        _stage_event_callback(state, "architecture"),
        _model_for(state, "architecture"),
    )

    # The design layer is for people, not for downstream agents: kept
    # out of `architecture` so it doesn't bloat every later prompt.
    design = architecture.pop("design", None)

    return {
        "architecture": architecture,
        "design": design,
        "change_required": architecture["change_required"],
        "failure_analysis": {},
        "history": append_history(
            state,
            {"stage": "architecture", "status": "DONE", "result": architecture, "design": design},
        ),
        **_clarifying_questions_field(architecture),
    }


def quality_node(state: SDLCState):
    quality = quality_agent.run(
        state["workspace"].root,
        state["repo_index"],
        state["requirement"],
        state["architecture"],
        state.get("failure_analysis") or None,
        _note_for("quality", state),
        _stage_event_callback(state, "quality"),
        _model_for(state, "quality"),
    )

    return {
        "quality": quality,
        "history": append_history(
            state, {"stage": "quality", "status": "DONE", "result": quality}
        ),
        **_clarifying_questions_field(quality),
    }


def coding_node(state: SDLCState):
    human_note = _human_note_for("coding", state)

    result = coding_agent.run(
        state["workspace"].root,
        state["repo_index"],
        state["requirement"],
        state["architecture"],
        state["quality"],
        state.get("review") or None,
        state.get("failure_analysis") or None,
        _note_for("coding", state),
        _stage_event_callback(state, "coding"),
        _model_for(state, "coding"),
    )

    # A human revising coding's own last output in place (via the
    # HITL "suggest changes" gate) must NOT consume MAX_ITERATIONS --
    # that budget bounds the automated coding<->testing<->
    # failure_analysis retry loop, a different, unsupervised risk.
    # Only advance it on a genuine attempt.
    iteration = state.get("iteration", 0) if human_note else state.get("iteration", 0) + 1

    # Captured now, not computed later by the UI on render: a later
    # coding pass (automated retry or another human revision) changes
    # the working tree, and `git diff` only ever reflects its CURRENT
    # state -- if the UI recomputed this from the live workspace on
    # every rerun, replaying an earlier pass's history entry would
    # silently show a later pass's diff instead of what that pass
    # actually did.
    ws = state["workspace"]
    touched_paths = list(result.get("files_written", {})) + list(result.get("files_deleted", []))
    result["diffs"] = {
        path: workspace_module.diff_against_base(ws, path=path) for path in touched_paths
    }

    return {
        "iteration": iteration,
        # New code invalidates any earlier QA verdict.
        "qa_review": {},
        "history": append_history(
            state,
            {
                "stage": "coding",
                "status": "DONE",
                "iteration": iteration,
                "result": result,
            },
        ),
        **_clarifying_questions_field(result),
    }


def testing_node(state: SDLCState):
    testing = testing_agent.run(
        state["workspace"].root,
        state["requirement"],
        state.get("review") or {},
        state.get("change_required"),
        _note_for("testing", state),
        _stage_event_callback(state, "testing"),
        _model_for(state, "testing"),
    )

    test_result = testing["test_result"]
    signature = _failure_signature(test_result)
    failure_history = list(state.get("failure_history", []))

    if test_result["status"] != "PASS":
        failure_history.append(
            {"iteration": state.get("iteration", 0), "signature": signature}
        )

    stagnation_count = 0
    if test_result["status"] != "PASS" and len(failure_history) >= 2:
        if failure_history[-2]["signature"] == signature:
            stagnation_count = state.get("stagnation_count", 0) + 1

    return {
        "testing": testing,
        "failure_history": failure_history,
        "stagnation_count": stagnation_count,
        "history": append_history(
            state,
            {
                "stage": "testing",
                "status": test_result["status"],
                "iteration": state.get("iteration", 0),
                "result": testing,
            },
        ),
        **_clarifying_questions_field(testing),
    }


def failure_analysis_node(state: SDLCState):
    analysis = failure_analysis_agent.run(
        state["workspace"].root,
        state["repo_index"],
        state["requirement"],
        state["architecture"],
        state["testing"],
        _note_for("failure_analysis", state),
        _stage_event_callback(state, "failure_analysis"),
        _model_for(state, "failure_analysis"),
    )

    return {
        "failure_analysis": analysis,
        "history": append_history(
            state,
            {"stage": "failure_analysis", "status": "DONE", "result": analysis},
        ),
        **_next_agent_fields(analysis),
        **_clarifying_questions_field(analysis),
    }


def review_node(state: SDLCState):
    ws = state["workspace"]
    changed_files = workspace_module.changed_files(ws)
    static_analysis_result = static_analysis.run(ws.root, changed_files)
    dependency_fix_files = (state.get("failure_analysis") or {}).get("dependency_fix_files") or []
    manifest_check_result = manifest_check.run(
        state.get("architecture") or {}, changed_files, extra_declared=dependency_fix_files
    )

    review = review_agent.run(
        state["workspace"].root,
        state["repo_index"],
        state["workspace"],
        state["requirement"],
        state["architecture"],
        state["testing"],
        static_analysis_result,
        manifest_check_result,
        _note_for("review", state),
        _stage_event_callback(state, "review"),
        _model_for(state, "review"),
    )

    return {
        "review": review,
        # A fresh Code Review means QA's verdict (if any) was about an
        # older diff -- it must run again before release.
        "qa_review": {},
        "history": append_history(
            state, {"stage": "review", "status": review["status"], "result": review}
        ),
        **_next_agent_fields(review),
        **_clarifying_questions_field(review),
    }


def qa_review_node(state: SDLCState):
    qa_review = qa_review_agent.run(
        state["workspace"].root,
        state["repo_index"],
        state["workspace"],
        state["requirement"],
        state["architecture"],
        state.get("quality") or {},
        state["testing"],
        state["review"],
        _note_for("qa_review", state),
        _stage_event_callback(state, "qa_review"),
        _model_for(state, "qa_review"),
    )

    return {
        "qa_review": qa_review,
        "history": append_history(
            state, {"stage": "qa_review", "status": qa_review["status"], "result": qa_review}
        ),
        **_next_agent_fields(qa_review),
        **_clarifying_questions_field(qa_review),
    }


def release_planning_node(state: SDLCState):
    ws = state["workspace"]
    changed = workspace_module.changed_files(ws)

    release_plan = release_agent.run(
        state["requirement"],
        state["architecture"],
        state["testing"],
        state["review"],
        changed,
        _note_for("release_planning", state),
        _stage_event_callback(state, "release_planning"),
        _model_for(state, "release"),
        qa_review=state.get("qa_review"),
    )

    return {
        "release_plan": release_plan,
        "history": append_history(
            state,
            {"stage": "release_planning", "status": "DONE", "result": release_plan},
        ),
        **_clarifying_questions_field(release_plan),
    }


def release_apply_node(state: SDLCState):
    ws = state["workspace"]
    plan = state["release_plan"]

    commit_message = f"{plan['release_title']}\n\n{plan['release_summary']}"
    commit_sha = workspace_module.commit_all(ws, commit_message)

    run_dir = config.RUNS_ROOT / state["run_id"]
    diff_text = ""
    diff_stat = ""
    diff_path = None

    if commit_sha:
        diff_text = workspace_module.diff_against_base(ws)
        diff_stat = workspace_module.diff_stat_against_base(ws)
        run_dir.mkdir(parents=True, exist_ok=True)
        diff_path = run_dir / "diff.patch"
        diff_path.write_text(diff_text, encoding="utf-8")

    push_result = (
        workspace_module.push(ws)
        if commit_sha
        else {"status": "SKIPPED", "reason": "Nothing to commit."}
    )

    fork_result = None
    head_owner = None

    if commit_sha and push_result.get("status") == "FAIL":
        # Direct push failed -- most likely the token only has read
        # access to this repo. Fall back to forking it under the
        # token's own account and pushing there instead.
        # create_or_get_fork is idempotent: if the account has already
        # forked this repo, it just returns that existing fork rather
        # than erroring or creating a duplicate.
        direct_push_error = push_result.get("error")
        fork_result = github_api.create_or_get_fork(ws.owner, ws.name)

        if fork_result.get("status") == "PASS":
            fork_push_result = workspace_module.push_to_url(
                ws, workspace_module.authenticated_url(fork_result["clone_url"])
            )
            if fork_push_result.get("status") == "PASS":
                push_result = {
                    "status": "PASS",
                    "branch": ws.working_branch,
                    "via_fork": fork_result["full_name"],
                }
                head_owner = fork_result["fork_owner"]
            else:
                push_result = {
                    "status": "FAIL",
                    "error": (
                        f"Direct push failed ({direct_push_error}); push "
                        f"to fork {fork_result['full_name']} also failed: "
                        f"{fork_push_result.get('error')}"
                    ),
                    "branch": ws.working_branch,
                }
        else:
            push_result = {
                "status": "FAIL",
                "error": (
                    f"Direct push failed ({direct_push_error}); fork "
                    f"fallback also failed: {fork_result.get('error')}"
                ),
                "branch": ws.working_branch,
            }

    pr_result = {"status": "SKIPPED", "reason": "Push did not succeed."}
    if push_result.get("status") == "PASS":
        pr_result = github_api.create_pull_request(
            owner=ws.owner,
            repo=ws.name,
            head_branch=ws.working_branch,
            base_branch=ws.base_branch,
            title=plan["release_title"],
            body=plan["release_summary"] + "\n\n" + plan["feedback_to_requirements"],
            head_owner=head_owner,
        )

    tracker = issue_tracker.LocalMockIssueTracker(run_dir / "issue_tracker")
    wiki = issue_tracker.LocalMockWiki(run_dir / "wiki")

    issue_summary = plan["issue_summary"]
    issue_record = tracker.create_issue(
        issue_summary["title"],
        issue_summary["description"],
        issue_summary["acceptance_criteria"],
        issue_summary.get("status", "DONE"),
    )

    wiki_summary = plan["wiki_summary"]
    wiki_record = wiki.create_page(wiki_summary["title"], wiki_summary["sections"])

    release_result = {
        "commit_sha": commit_sha,
        "push": push_result,
        "fork": fork_result,
        "pull_request": pr_result,
        "issue": issue_record,
        "wiki": wiki_record,
        "diff": diff_text,
        "diff_stat": diff_stat,
        "diff_path": str(diff_path) if diff_path else None,
    }

    return {
        "release_result": release_result,
        "history": append_history(
            state,
            {
                "stage": "release_apply",
                "status": "PASS" if commit_sha else "NOTHING_TO_COMMIT",
                "result": release_result,
            },
        ),
    }


def finalize_node(state: SDLCState):
    run_outcome = _derive_run_outcome(state)

    ws = state["workspace"]
    entry = {
        "run_id": state.get("run_id"),
        "repo_key": feedback_store.repo_key(ws.owner, ws.name),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "spec_text": state.get("spec_text", ""),
        "outcome": run_outcome,
        "iteration_count": state.get("iteration", 0),
        "files_touched": workspace_module.changed_files(ws),
        "review_findings_count": len(state.get("review", {}).get("findings", [])),
        "qa_findings_count": len((state.get("qa_review") or {}).get("findings", [])),
        "release": state.get("release_result"),
    }
    feedback_store.append_run_summary(entry)

    return {
        "run_outcome": run_outcome,
        "history": append_history(
            state, {"stage": "finalize", "status": run_outcome, "result": entry}
        ),
    }


def _derive_run_outcome(state: SDLCState) -> str:
    if state.get("user_agents"):
        return "COMPLETED"

    release_result = state.get("release_result")
    if release_result:
        if release_result["commit_sha"] is None:
            return "NOTHING_TO_COMMIT"
        if release_result["push"].get("status") == "FAIL":
            return "PUSH_FAILED"
        return "APPROVED_RELEASED"

    if not state.get("change_required") and state.get("testing", {}).get(
        "test_result", {}
    ).get("status") == "PASS":
        return "NO_CHANGE_VERIFIED"

    if state.get("review", {}).get("status") == "CHANGES_REQUESTED":
        return "REVIEW_BLOCKED"

    if (state.get("qa_review") or {}).get("status") == "CHANGES_REQUESTED":
        return "QA_BLOCKED"

    if state.get("stagnation_count", 0) >= MAX_STAGNANT_ITERATIONS:
        return "STAGNANT"

    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "EXHAUSTED_ITERATIONS"

    return "INCOMPLETE"


# ============================================================
# DETERMINISTIC ROUTERS
#
# architecture_router and testing_router: no agent choice involved,
# just a data field. These are what make most of the pipeline a fixed
# sequence (see module docstring).
# ============================================================

def architecture_router(state: SDLCState) -> str:
    if state.get("total_steps", 0) >= MAX_TOTAL_STEPS:
        return "finalize"
    return "quality" if state.get("change_required") else "testing"


def testing_router(state: SDLCState) -> str:
    if state.get("total_steps", 0) >= MAX_TOTAL_STEPS:
        return "finalize"

    test_result = (state.get("testing") or {}).get("test_result", {})
    if test_result.get("status") != "PASS":
        # The one place the coding<->testing<->failure_analysis retry
        # loop's iteration/stagnation caps are actually enforced --
        # every failing pass funnels through here on its way to
        # failure_analysis, regardless of which stage sent it back for
        # another attempt.
        if state.get("iteration", 0) > 0:
            if not _can_retry(state):
                return "finalize"
            if state.get("stagnation_count", 0) >= MAX_STAGNANT_ITERATIONS:
                return "finalize"
        return "failure_analysis"

    if not state.get("change_required"):
        return "finalize"  # NO_CHANGE_VERIFIED: nothing to review or release.
    return "review"


# ============================================================
# DYNAMIC ROUTER
#
# Shared by ONLY failure_analysis and review -- the two agents with
# any routing autonomy (see module docstring). Reads that node's own
# `next_agent` choice (already restricted to a small, sane enum by its
# own OUTPUT_SCHEMA) and enforces the few hard invariants neither
# agent's choice can override. Never raises over a malformed or
# missing choice -- falls back to the safest generic step
# (ARCHITECTURE) instead, since a routing hiccup should degrade the
# run, not crash it.
# ============================================================

def resolve_next_node(state: SDLCState) -> str:
    if state.get("total_steps", 0) >= MAX_TOTAL_STEPS:
        return "finalize"

    requested_name = str(state.get("next_agent") or "").upper().strip()
    node = NODE_BY_AGENT_NAME.get(requested_name, "architecture")

    # Hard invariant: release_planning (and therefore release_apply)
    # is unreachable unless review's status is actually PASS --
    # regardless of what review's own next_agent said (its own
    # _validate already keeps these in sync in the normal case; this
    # is the backstop for anything that slips through).
    #
    # Two reviewers gate release, in order: Code Review, then QA Review.
    # Whichever is still outstanding is where a release request lands.
    # (An empty/missing verdict has no status, so counts as outstanding.)
    if node == "release_planning":
        if (state.get("review") or {}).get("status") != "PASS":
            node = "review"
        elif (state.get("qa_review") or {}).get("status") != "PASS":
            node = "qa_review"

    # Hard invariant: coding needs something concrete to act on -- an
    # architecture-named file to touch, or a dependency-manifest fix
    # failure_analysis diagnosed. Never invoke it with neither; route
    # to architecture instead so a real plan gets produced first.
    if node == "coding":
        architecture = state.get("architecture") or {}
        dependency_fix_files = (state.get("failure_analysis") or {}).get("dependency_fix_files") or []
        has_targets = bool(
            architecture.get("files_to_modify")
            or architecture.get("files_to_create")
            or architecture.get("files_to_delete")
            or dependency_fix_files
        )
        if not has_targets:
            node = "architecture"

    # Hard invariant: once we're past the first pass through the run,
    # the same iteration/stagnation caps that always bounded the
    # coding<->testing<->failure_analysis retry loop still apply to a
    # requested retry (architecture/quality/coding), regardless of
    # which of the two dynamic agents asked for it. Not applied to
    # release_planning/finalize -- those aren't another attempt.
    if node in RETRY_LOOP_NODES and state.get("iteration", 0) > 0:
        if not _can_retry(state):
            return "finalize"
        if state.get("stagnation_count", 0) >= MAX_STAGNANT_ITERATIONS:
            return "finalize"

    return node


# State keys only the harness (this module) may write. Counters that
# bound the run must never be influenced by anything an agent produced:
# a node's returned dict is built field-by-field from named agent
# outputs (never a spread of the raw agent result), and this guard makes
# that a checked invariant, not a convention -- a node that tries to set
# one fails loudly instead of silently resetting its own step budget.
# (iteration / stagnation_count / failure_history are also computed in
# code, from test results and attempt counts -- never read from agent
# output -- but individual nodes legitimately set those.)
HARNESS_OWNED_KEYS = frozenset({"total_steps"})


# ============================================================
# HUMAN OVERRIDE
#
# At a human-in-the-loop gate the operator may override the pipeline's
# own routing: pick any next agent and tell it what to do. A human's
# call isn't an agent's choice, so resolve_next_node's invariants don't
# apply to it -- but the target still has to be runnable (its inputs
# must exist), which these helpers check, and the one gate that exists
# to protect a release (both reviews passing) is surfaced as an explicit
# warning the UI makes the human acknowledge, then records in the audit
# ledger. The step cap still applies (nodes are step-counted as usual).
# ============================================================

OVERRIDE_TARGETS = [
    "requirement", "architecture", "quality", "coding", "testing",
    "failure_analysis", "review", "qa_review", "release_planning",
]

# State each target's node indexes directly (a missing key would crash
# the run), keyed by node name.
_OVERRIDE_NEEDS = {
    "requirement": (),
    "architecture": ("requirement",),
    "quality": ("requirement", "architecture"),
    "coding": ("requirement", "architecture", "quality"),
    "testing": ("requirement",),
    "failure_analysis": ("requirement", "architecture", "testing"),
    "review": ("requirement", "architecture", "testing"),
    "qa_review": ("requirement", "architecture", "testing", "review"),
    "release_planning": ("requirement", "architecture", "testing", "review"),
}
_NEEDS_A_CHANGE = {"quality", "coding", "review", "qa_review", "release_planning"}


def override_blockers(target: str, state: "SDLCState") -> List[str]:
    """Reasons `target` can't run yet, as plain sentences; empty means
    it can. Never raises."""
    if target in (state.get("user_agents") or {}):
        return []  # user-defined agents take only the spec + upstream output
    if target not in _OVERRIDE_NEEDS:
        return [f"'{target}' is not a stage you can send the run to."]
    reasons = [
        f"{need.replace('_', ' ').title()} hasn't produced output yet."
        for need in _OVERRIDE_NEEDS[target]
        if not state.get(need)
    ]
    if target in _NEEDS_A_CHANGE and state.get("architecture") and not state.get("change_required"):
        reasons.append("Architecture decided no change is required, so there is nothing for this stage to act on.")
    return reasons


def override_targets_for(state: "SDLCState") -> List[str]:
    return list(state["custom_pipeline"]) if state.get("user_agents") else list(OVERRIDE_TARGETS)


def release_gate_warning(state: "SDLCState") -> Optional[str]:
    """Why sending the run to release planning now would bypass a
    review gate, or None if both reviews have passed."""
    missing = []
    if (state.get("review") or {}).get("status") != "PASS":
        missing.append("Code Review")
    if (state.get("qa_review") or {}).get("status") != "PASS":
        missing.append("QA Review")
    if not missing:
        return None
    return f"{' and '.join(missing)} {'has' if len(missing) == 1 else 'have'} not passed."


def override_note(from_stage: str, instruction: str) -> str:
    """The note handed to the overridden-to stage. Worded for an agent
    that did NOT just produce the output being discussed -- the generic
    human-note wrapper says "revise your own last output", which would
    mislead a stage the run was redirected to."""
    return (
        f"The human operator overrode the pipeline's routing and sent the run to you "
        f"directly (from the '{from_stage}' stage). Follow their instruction: {instruction}"
    )


def with_step_counter(node_fn):
    """Wraps a dynamically-routed node so every visit counts against
    MAX_TOTAL_STEPS, without every node function needing to compute
    this itself. Not applied to release_planning/release_apply/
    finalize -- that fixed tail runs at most once per run regardless
    of routing, so it needs no loop protection."""

    def wrapped(state: SDLCState):
        result = dict(node_fn(state))
        leaked = HARNESS_OWNED_KEYS & result.keys()
        if leaked:
            raise RuntimeError(
                f"Node tried to set harness-owned state key(s) {sorted(leaked)}; "
                f"run-bounding counters are written only by with_step_counter."
            )
        result["total_steps"] = state.get("total_steps", 0) + 1
        return result

    return wrapped


# ============================================================
# GRAPH
# ============================================================

def build_graph():
    graph = StateGraph(SDLCState)

    graph.add_node("requirement", with_step_counter(requirement_node))
    graph.add_node("architecture", with_step_counter(architecture_node))
    graph.add_node("quality", with_step_counter(quality_node))
    graph.add_node("coding", with_step_counter(coding_node))
    graph.add_node("testing", with_step_counter(testing_node))
    graph.add_node("failure_analysis", with_step_counter(failure_analysis_node))
    graph.add_node("review", with_step_counter(review_node))
    graph.add_node("qa_review", with_step_counter(qa_review_node))
    graph.add_node("release_planning", release_planning_node)
    graph.add_node("release_apply", release_apply_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "requirement")
    graph.add_edge("requirement", "architecture")

    graph.add_conditional_edges(
        "architecture", architecture_router, {"quality": "quality", "testing": "testing", "finalize": "finalize"}
    )
    graph.add_edge("quality", "coding")
    graph.add_edge("coding", "testing")
    graph.add_conditional_edges(
        "testing", testing_router,
        {"review": "review", "failure_analysis": "failure_analysis", "finalize": "finalize"},
    )

    # The only three nodes with any routing autonomy -- see module
    # docstring. dynamic_targets is deliberately the full set both
    # share (rather than each getting only its own schema's narrower
    # subset): resolve_next_node's own invariants, not the edge
    # declaration, are what should decide what's actually reachable.
    dynamic_targets = {name: name for name in NODE_BY_AGENT_NAME.values()}
    graph.add_conditional_edges("failure_analysis", resolve_next_node, dynamic_targets)
    graph.add_conditional_edges("review", resolve_next_node, dynamic_targets)
    graph.add_conditional_edges("qa_review", resolve_next_node, dynamic_targets)

    # Applying an already-approved release plan is mechanical
    # execution, not a judgment call -- fixed, not agent-routed.
    graph.add_edge("release_planning", "release_apply")
    graph.add_edge("release_apply", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


def user_agent_node(node_id: str):
    """Graph node for one user-defined agent (agents/custom.py). Each
    sees what the agents before it in the workflow produced."""

    def node(state: SDLCState):
        agent = state["user_agents"][node_id]
        upstream = [
            {"name": state["user_agents"][h["stage"]]["name"], **h["result"]}
            for h in state.get("history", [])
            if h.get("stage") in state["user_agents"] and h["stage"] != node_id
        ]
        result = custom_agent.run(
            state["workspace"].root,
            indexer.repo_overview_text(state["repo_index"]),
            state["spec_text"],
            agent,
            upstream,
            _note_for(node_id, state),
            _stage_event_callback(state, node_id),
            agent.get("model") or _model_for(state, node_id),
            state.get("knowledge_paths") or None,
        )
        return {
            "history": append_history(
                state, {"stage": node_id, "status": "DONE", "result": result}
            ),
        }

    return node


def node_fn_for(node: str, state: SDLCState):
    """The step-counted callable for a node name, built-in or user-defined."""
    if node in (state.get("user_agents") or {}):
        return with_step_counter(user_agent_node(node))
    return None


# ============================================================
# USER-BUILT WORKFLOW
#
# The Streamlit workflow builder (and cli.py --stages) lets the user pick
# which agents run and in what order. That is a LINEAR sequence, not a
# free graph: the user's order decides what runs next going forward,
# while the loops that already existed keep working wherever the agents
# involved are present -- failure_analysis/review/qa_review still pick a
# retry target (resolve_next_node), testing still routes a failure to
# failure_analysis. A retry target the user left out of the sequence
# simply ends the run (finalize) instead of running an agent they didn't
# choose. A default-order sequence is NOT custom -- callers leave
# custom_pipeline unset then, and the standard routing above applies.
# ============================================================

DEFAULT_PIPELINE = [
    "requirement", "architecture", "quality", "coding", "testing",
    "failure_analysis", "review", "qa_review", "release_planning",
]

# Runs only when testing fails (testing_router sends it there) -- never
# as a plain step in the forward sequence, where it would have no
# failure to analyze.
_CONDITIONAL_STAGES = {"failure_analysis"}
_REVIEWERS = ("review", "qa_review")


def validate_custom_pipeline(stages: List[str]) -> List[str]:
    """Reasons `stages` can't run as an ordered pipeline, as plain
    sentences; empty means it can. A stage's inputs (_OVERRIDE_NEEDS)
    must come from stages placed before it. Never raises."""
    if not stages:
        return ["Add at least one agent to the workflow."]
    errors, seen = [], []
    for stage in stages:
        if stage not in _OVERRIDE_NEEDS:
            errors.append(f"'{stage}' is not a known agent.")
            continue
        if stage in seen:
            errors.append(f"{_title(stage)} appears more than once.")
            continue
        for need in _OVERRIDE_NEEDS[stage]:
            if need not in seen:
                errors.append(f"{_title(stage)} needs {_title(need)} to run before it.")
        seen.append(stage)
    return errors


def custom_pipeline_warnings(stages: List[str]) -> List[str]:
    """Valid-but-risky choices worth surfacing to the user."""
    warnings = []
    if "release_planning" in stages:
        skipped = [_title(r) for r in _REVIEWERS if r not in stages]
        if skipped:
            warnings.append(
                f"Release Planning is in this workflow without {' and '.join(skipped)} -- "
                f"nothing will review the change before it is released."
            )
    if "failure_analysis" in stages and "coding" not in stages:
        warnings.append(
            "Failure Analysis can only send a failing test back to agents that are in "
            "the workflow; without Coding, a failure ends the run."
        )
    return warnings


def _title(stage: str) -> str:
    return stage.replace("_", " ").title()


def _next_in_sequence(current: str, state: "SDLCState") -> str:
    """The next stage after `current` in the user's order that can
    actually run on the state so far (e.g. skips Coding when
    Architecture decided no change is required); finalize if none."""
    stages = state["custom_pipeline"]
    for nxt in stages[stages.index(current) + 1:]:
        if nxt in _CONDITIONAL_STAGES:
            continue
        if not override_blockers(nxt, state):
            return nxt
    return "finalize"


def custom_next_node(current: str, state: "SDLCState") -> Optional[str]:
    """Routing for a user-built workflow: what runs after `current`.
    Same role as the per-node routers wired in build_graph(), and used
    by both build_custom_graph() and the Streamlit driver."""
    if current == "finalize":
        return None
    if state.get("user_agents"):
        stages = state["custom_pipeline"]
        if current not in stages or state.get("total_steps", 0) >= MAX_TOTAL_STEPS:
            return "finalize"
        idx = stages.index(current) + 1
        return stages[idx] if idx < len(stages) else "finalize"
    if current == "release_planning":
        return "release_apply"
    if current == "release_apply":
        return "finalize"
    if state.get("total_steps", 0) >= MAX_TOTAL_STEPS:
        return "finalize"

    stages = state["custom_pipeline"]

    if current == "testing":
        if (state.get("testing") or {}).get("test_result", {}).get("status") != "PASS":
            target = testing_router(state)
            return target if target == "failure_analysis" and target in stages else "finalize"
        return _next_in_sequence(current, state)

    if current in ("failure_analysis", "review", "qa_review"):
        requested = str(state.get("next_agent") or "").upper().strip()
        # resolve_next_node makes release wait on BOTH reviewers; here
        # only the reviewers the user actually included gate it.
        if requested == "RELEASE_PLANNING" and "release_planning" in stages:
            if all((state.get(r) or {}).get("status") == "PASS" for r in _REVIEWERS if r in stages):
                if state.get("total_steps", 0) < MAX_TOTAL_STEPS:
                    return "release_planning"
        target = resolve_next_node(state)
        if target == "finalize" or (target in stages and not override_blockers(target, state)):
            return target
        return "finalize"

    return _next_in_sequence(current, state)


def build_custom_graph(stages: List[str]):
    """build_graph() for a user-chosen sequence. The state passed to the
    returned graph must carry custom_pipeline=stages."""
    errors = validate_custom_pipeline(stages)
    if errors:
        raise ValueError("; ".join(errors))

    graph = StateGraph(SDLCState)
    node_fns = {
        "requirement": requirement_node, "architecture": architecture_node,
        "quality": quality_node, "coding": coding_node, "testing": testing_node,
        "failure_analysis": failure_analysis_node, "review": review_node,
        "qa_review": qa_review_node,
    }
    names = list(stages) + (["release_apply"] if "release_planning" in stages else []) + ["finalize"]
    for stage in stages:
        if stage == "release_planning":
            graph.add_node(stage, release_planning_node)
        else:
            graph.add_node(stage, with_step_counter(node_fns[stage]))
    if "release_planning" in stages:
        graph.add_node("release_apply", release_apply_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, stages[0])
    targets = {name: name for name in names}
    for name in names:
        if name == "finalize":
            graph.add_edge("finalize", END)
        else:
            graph.add_conditional_edges(name, lambda s, n=name: custom_next_node(n, s), targets)
    return graph.compile()


def new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{uuid.uuid4().hex[:6]}"
