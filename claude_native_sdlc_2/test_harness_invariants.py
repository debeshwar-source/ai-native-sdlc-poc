"""Run-bounding counters and routing invariants must hold no matter what
an agent says. Plain asserts -- run with `python3 test_harness_invariants.py`
or pytest."""

import orchestrator.graph as g
from agents._common import (
    MAX_OPTION_CHARS,
    MAX_QUESTION_CHARS,
    format_work_order,
    validate_clarifying_questions,
)
from agents.architecture import _default_work_order
from agents.qa_review import _apply_coverage_gate, _validate as qa_validate, criterion_ids


def test_node_cannot_set_total_steps():
    hostile = g.with_step_counter(lambda state: {"total_steps": 0})
    try:
        hostile({"total_steps": 39})
    except RuntimeError as exc:
        assert "harness-owned" in str(exc)
    else:
        raise AssertionError("a node overwrote total_steps")


def test_step_counter_increments():
    node = g.with_step_counter(lambda state: {})
    assert node({"total_steps": 5})["total_steps"] == 6


def test_step_cap_beats_any_agent_choice():
    for choice in ("CODING", "RELEASE_PLANNING", "ARCHITECTURE", "bogus", None):
        state = {"total_steps": g.MAX_TOTAL_STEPS, "next_agent": choice}
        assert g.resolve_next_node(state) == "finalize", choice


def test_release_needs_a_passing_review():
    state = {"total_steps": 1, "next_agent": "RELEASE_PLANNING", "review": {"status": "CHANGES_REQUESTED"}}
    assert g.resolve_next_node(state) == "review"


def test_release_needs_a_passing_qa_review_too():
    base = {"total_steps": 1, "next_agent": "RELEASE_PLANNING", "review": {"status": "PASS"}}
    assert g.resolve_next_node({**base}) == "qa_review"                                   # QA not run yet
    assert g.resolve_next_node({**base, "qa_review": {}}) == "qa_review"                  # reset by new code
    assert g.resolve_next_node({**base, "qa_review": {"status": "CHANGES_REQUESTED"}}) == "qa_review"
    assert g.resolve_next_node({**base, "qa_review": {"status": "PASS"}}) == "release_planning"


def test_failing_code_review_outranks_passing_qa():
    state = {"total_steps": 1, "next_agent": "RELEASE_PLANNING",
             "review": {"status": "CHANGES_REQUESTED"}, "qa_review": {"status": "PASS"}}
    assert g.resolve_next_node(state) == "review"


def test_graph_has_qa_review_node_and_both_gate_release():
    compiled = g.build_graph()
    assert "qa_review" in compiled.get_graph().nodes
    assert "qa_review" in g.GATED_NODE_NAMES


REQ = {"acceptance_criteria": ["Returns 200 on /health", "Rejects empty body with 400"]}


def _qa(coverage, findings=None, next_agent="RELEASE_PLANNING", status="PASS"):
    result = {"status": status, "criteria_coverage": coverage, "findings": findings or [],
              "next_agent": next_agent}
    _apply_coverage_gate(result, criterion_ids(REQ))
    qa_validate(result)
    return result


def test_qa_uncovered_criterion_blocks_and_routes_to_quality():
    r = _qa([{"criterion_id": "C1", "status": "COVERED", "evidence": "test_health::test_ok asserts 200"},
             {"criterion_id": "C2", "status": "UNCOVERED", "evidence": "no test"}])
    assert r["status"] == "CHANGES_REQUESTED"
    assert any(f["id"] == "qa-coverage:C2" and f["severity"] == "BLOCKING" for f in r["findings"])
    assert r["next_agent"] == "QUALITY"


def test_qa_unassessed_criterion_counts_as_uncovered():
    r = _qa([{"criterion_id": "C1", "status": "COVERED", "evidence": "test_ok"}])
    assert r["status"] == "CHANGES_REQUESTED"
    assert [c["status"] for c in r["criteria_coverage"]] == ["COVERED", "UNCOVERED"]


def test_qa_partial_and_uncited_cover_are_follow_ups_not_blockers():
    r = _qa([{"criterion_id": "C1", "status": "PARTIAL", "evidence": "only status code"},
             {"criterion_id": "C2", "status": "COVERED", "evidence": ""}])
    assert r["status"] == "PASS"
    assert {f["severity"] for f in r["findings"]} == {"FOLLOW_UP"}


def test_qa_ignores_made_up_criterion_ids_and_model_cannot_pass_a_gap():
    r = _qa([{"criterion_id": "C1", "status": "COVERED", "evidence": "t"},
             {"criterion_id": "C9", "status": "COVERED", "evidence": "t"},
             {"criterion_id": "C2", "status": "UNCOVERED", "evidence": "none"}], status="PASS")
    assert len(r["criteria_coverage"]) == 2 and r["status"] == "CHANGES_REQUESTED"


def test_qa_all_covered_passes():
    r = _qa([{"criterion_id": "C1", "status": "COVERED", "evidence": "a"},
             {"criterion_id": "C2", "status": "COVERED", "evidence": "b"}])
    assert r["status"] == "PASS" and r["next_agent"] == "RELEASE_PLANNING" and not r["findings"]


def test_garbage_choice_degrades_not_crashes():
    assert g.resolve_next_node({"total_steps": 1, "next_agent": "rm -rf"}) == "architecture"


def test_questions_are_shortened_to_decision_shape():
    long_q = "Should we ship now? " + "Background detail. " * 60
    result = {"clarifying_questions": [
        {"question": long_q, "options": ["A " * 100, "B"], "recommended_option": "A " * 100}
    ]}
    validate_clarifying_questions(result, "t")
    q = result["clarifying_questions"][0]
    assert len(q["question"]) <= MAX_QUESTION_CHARS
    assert q["question"].startswith("Should we ship now?")
    assert all(len(o) <= MAX_OPTION_CHARS for o in q["options"])
    assert q["recommended_option"] in q["options"]


def test_work_order_defaults_and_rendering():
    arch = {"change_required": True, "changes": ["Add /health"], "files_to_modify": ["main.py"],
            "files_to_create": [], "files_to_delete": [], "existing_conventions": ["FastAPI routers"]}
    _default_work_order(arch)
    assert arch["objective"] == "Add /health"
    assert arch["boundaries"] and arch["definition_of_done"]
    text = format_work_order(arch)
    for part in ("OBJECTIVE", "OUTPUT", "TOOLS & REFERENCES", "BOUNDARIES & DEFINITION OF DONE", "modify main.py"):
        assert part in text




def test_end_to_end_qa_failure_loops_back_before_release():
    """Stubbed nodes through the real compiled graph: QA's first verdict
    sends the run back through Quality/Coding/Testing/Code Review, and
    release only happens after QA passes on the second pass."""
    visited = []
    qa_calls = {"n": 0}

    def stub(name, update_fn=lambda s: {}):
        def node(state):
            visited.append(name)
            return update_fn(state)
        return node

    def qa(state):
        qa_calls["n"] += 1
        if qa_calls["n"] == 1:
            return {"qa_review": {"status": "CHANGES_REQUESTED"}, "next_agent": "QUALITY"}
        return {"qa_review": {"status": "PASS"}, "next_agent": "RELEASE_PLANNING"}

    originals = {n: getattr(g, n) for n in (
        "requirement_node", "architecture_node", "quality_node", "coding_node", "testing_node",
        "review_node", "qa_review_node", "release_planning_node", "release_apply_node", "finalize_node")}
    try:
        g.requirement_node = stub("requirement")
        g.architecture_node = stub("architecture", lambda s: {"change_required": True})
        g.quality_node = stub("quality")
        g.coding_node = stub("coding", lambda s: {"qa_review": {}, "iteration": s.get("iteration", 0) + 1})
        g.testing_node = stub("testing", lambda s: {"testing": {"test_result": {"status": "PASS"}}})
        g.review_node = stub("review", lambda s: {"review": {"status": "PASS"}, "qa_review": {},
                                                  "next_agent": "RELEASE_PLANNING"})
        g.qa_review_node = stub("qa_review", qa)
        g.release_planning_node = stub("release_planning")
        g.release_apply_node = stub("release_apply")
        g.finalize_node = stub("finalize")
        # The stubs above are patched module globals; build_graph reads them at call time.
        # (architecture's files_to_modify must exist for the coding invariant -- QUALITY->CODING
        # is a fixed edge so it is unaffected.)
        list(g.build_graph().stream({"run_id": "t", "iteration": 0}, stream_mode="updates"))
    finally:
        for name, fn in originals.items():
            setattr(g, name, fn)

    assert visited == [
        "requirement", "architecture", "quality", "coding", "testing", "review", "qa_review",
        "quality", "coding", "testing", "review", "qa_review",
        "release_planning", "release_apply", "finalize",
    ], visited


def test_override_blocks_stages_whose_inputs_dont_exist():
    fresh = {"requirement": {"feature": "x"}}
    assert g.override_blockers("requirement", {}) == []
    assert g.override_blockers("architecture", fresh) == []
    assert any("Architecture" in r for r in g.override_blockers("quality", fresh))
    assert any("Review" in r for r in g.override_blockers("release_planning", fresh))
    assert g.override_blockers("finalize", fresh)          # not a valid target
    assert g.override_blockers("coding", {})               # nothing exists yet


def test_override_blocks_change_stages_when_no_change_needed():
    state = {"requirement": {"f": 1}, "architecture": {"a": 1}, "testing": {"t": 1},
             "review": {"status": "PASS"}, "change_required": False}
    assert any("no change" in r for r in g.override_blockers("coding", state))
    assert g.override_blockers("testing", state) == []


def test_override_can_reach_every_stage_once_state_exists():
    full = {"requirement": {"f": 1}, "architecture": {"a": 1}, "quality": {"q": 1}, "testing": {"t": 1},
            "review": {"status": "CHANGES_REQUESTED"}, "change_required": True}
    for target in g.OVERRIDE_TARGETS:
        assert g.override_blockers(target, full) == [], target


def test_release_gate_warning_names_the_missing_reviewer():
    assert g.release_gate_warning({"review": {"status": "PASS"}, "qa_review": {"status": "PASS"}}) is None
    assert "Code Review" in g.release_gate_warning({"review": {"status": "CHANGES_REQUESTED"},
                                                    "qa_review": {"status": "PASS"}})
    both = g.release_gate_warning({})
    assert "Code Review" in both and "QA Review" in both


def test_override_note_is_worded_for_an_agent_that_didnt_write_the_output():
    note = g.override_note("review", "Use the existing router")
    assert "overrode" in note and "'review'" in note and "Use the existing router" in note
    assert "revise your own" not in note


def test_override_targets_are_all_real_graph_nodes():
    nodes = set(g.build_graph().get_graph().nodes)
    assert set(g.OVERRIDE_TARGETS) <= nodes


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
