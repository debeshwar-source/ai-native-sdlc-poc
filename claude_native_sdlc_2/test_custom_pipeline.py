"""User-built workflow: validation, routing, and standing orders."""

import json
import orchestrator.graph as g

FULL = list(g.DEFAULT_PIPELINE)
CHANGE = {"requirement": {"x": 1}, "architecture": {"files_to_modify": ["a.py"]}, "change_required": True}


def test_validation_requires_inputs_to_come_first():
    assert g.validate_custom_pipeline(FULL) == []
    assert g.validate_custom_pipeline(["requirement", "architecture", "testing"]) == []
    errors = g.validate_custom_pipeline(["coding", "requirement"])
    assert any("Coding needs Architecture" in e for e in errors)
    assert g.validate_custom_pipeline([]) != []
    assert g.validate_custom_pipeline(["requirement", "requirement"]) != []
    assert g.validate_custom_pipeline(["requirement", "nope"]) != []


def test_forward_follows_the_users_order_and_skips_what_cannot_run():
    state = {"custom_pipeline": ["requirement", "architecture", "testing", "coding"], **CHANGE}
    # Coding needs Quality's output, which isn't in the workflow -> can't be reached.
    assert g.custom_next_node("requirement", state) == "architecture"
    assert g.custom_next_node("architecture", state) == "testing"
    state_no_change = {**state, "change_required": False}
    assert g.custom_next_node("architecture", {**state_no_change}) == "testing"


def test_no_change_skips_change_only_stages():
    state = {"custom_pipeline": ["requirement", "architecture", "quality", "testing"],
             "requirement": {"x": 1}, "architecture": {}, "change_required": False}
    assert g.custom_next_node("architecture", state) == "testing"


def test_failed_test_goes_to_failure_analysis_only_if_included():
    failing = {"total_steps": 1, "testing": {"test_result": {"status": "FAIL"}}, **CHANGE}
    with_fa = {**failing, "custom_pipeline": ["requirement", "architecture", "quality", "coding", "testing", "failure_analysis"]}
    without = {**failing, "custom_pipeline": ["requirement", "architecture", "quality", "coding", "testing"]}
    assert g.custom_next_node("testing", with_fa) == "failure_analysis"
    assert g.custom_next_node("testing", without) == "finalize"


def test_retry_target_outside_the_workflow_ends_the_run():
    stages = ["requirement", "architecture", "quality", "coding", "testing", "failure_analysis"]
    state = {"custom_pipeline": stages, "total_steps": 3, "quality": {"q": 1},
             "testing": {"t": 1}, "next_agent": "CODING", **CHANGE}
    assert g.custom_next_node("failure_analysis", state) == "coding"
    state["custom_pipeline"] = ["requirement", "architecture", "quality", "testing", "failure_analysis"]
    assert g.custom_next_node("failure_analysis", state) == "finalize"


def test_release_waits_only_on_reviewers_in_the_workflow():
    stages = ["requirement", "architecture", "quality", "coding", "testing", "review", "release_planning"]
    state = {"custom_pipeline": stages, "total_steps": 5, "next_agent": "RELEASE_PLANNING",
             "quality": {"q": 1}, "testing": {"t": 1}, "review": {"status": "PASS"}, **CHANGE}
    assert g.custom_next_node("review", state) == "release_planning"   # no QA Review in the workflow
    state["review"] = {"status": "CHANGES_REQUESTED"}
    assert g.custom_next_node("review", state) != "release_planning"
    assert g.custom_pipeline_warnings(stages)                          # QA Review missing -> warned


def test_release_tail_and_step_cap():
    state = {"custom_pipeline": FULL, "total_steps": 1}
    assert g.custom_next_node("release_planning", state) == "release_apply"
    assert g.custom_next_node("release_apply", state) == "finalize"
    assert g.custom_next_node("finalize", state) is None
    assert g.custom_next_node("requirement", {**state, "total_steps": g.MAX_TOTAL_STEPS}) == "finalize"


def test_standing_orders_reach_the_agent_note_alongside_a_human_note():
    state = {"stage_orders": {"coding": "  Use dataclasses.  "},
             "pending_human_note": {"stage": "coding", "note": "Rename it."}}
    note = g._note_for("coding", state)
    assert "Use dataclasses." in note and "Rename it." in note
    assert g._note_for("review", state) is None
    assert g._human_note_for("coding", state) == "Rename it."   # iteration logic unaffected


def test_custom_graph_builds_with_only_chosen_nodes():
    compiled = g.build_custom_graph(["requirement", "architecture", "testing"])
    nodes = set(compiled.get_graph().nodes)
    assert {"requirement", "architecture", "testing", "finalize"} <= nodes
    assert "coding" not in nodes and "release_apply" not in nodes


# ---- user-defined agents (workflow builder's "Add agent" tab) ----

def _user_state():
    import workflow_builder as wb
    catalog = [
        {"id": "a1", "name": "Researcher", "job_description": "Find things", "skills": "grep", "model": "claude-opus-5-5", "can_edit": False},
        {"id": "a2", "name": "Writer", "job_description": "Write docs", "skills": "", "can_edit": True},
    ]
    stages = [{"id": "a1", "orders": "be brief"}, {"id": "a2", "orders": ""}]
    assert wb.validate(stages, catalog) == []
    agents, pipeline, orders = wb.to_run_config(stages, catalog)
    return {"user_agents": agents, "custom_pipeline": pipeline, "stage_orders": orders,
            "spec_text": "do it", "repo_index": object(), "workspace": type("W", (), {"root": "/r"})(),
            "history": [], "total_steps": 0}


def test_user_agents_run_as_a_plain_linear_chain(monkeypatch):
    state = _user_state()
    assert state["stage_orders"] == {"a1": "be brief"}
    assert g.custom_next_node("a1", state) == "a2"
    assert g.custom_next_node("a2", state) == "finalize"
    assert g.custom_next_node("finalize", state) is None
    assert g.override_blockers("a2", state) == []
    assert g.override_targets_for(state) == ["a1", "a2"]
    assert g._derive_run_outcome(state) == "COMPLETED"


def test_user_agent_node_passes_orders_and_upstream_output(monkeypatch):
    import context.indexer as indexer
    monkeypatch.setattr(indexer, "repo_overview_text", lambda _i: "overview")
    seen = []

    def fake_run(root, overview, spec, agent, upstream, note, on_event, model, knowledge_paths):
        seen.append((agent["name"], [u["name"] for u in upstream], note, model))
        return {"summary": f"{agent['name']} done", "output": "x", "files_changed": []}

    monkeypatch.setattr(g.custom_agent, "run", fake_run)
    state = _user_state()
    for node_id in state["custom_pipeline"]:
        update = g.node_fn_for(node_id, state)(state)
        state.update(update)
    assert seen[0] == ("Researcher", [], "Standing orders for this stage: be brief", "claude-opus-5-5")
    assert seen[1] == ("Writer", ["Researcher"], None, g.config.MODEL_ID)   # no model chosen -> default
    assert state["total_steps"] == 2


def test_builtin_names_are_not_user_agents():
    assert g.node_fn_for("coding", {"user_agents": {"a1": {}}}) is None


def test_custom_agent_prompt_carries_name_job_skills_and_edit_rights():
    from agents.custom import build_system_prompt
    p = build_system_prompt("Auditor", "Audit deps", "OWASP", can_edit=False)
    assert "Auditor" in p and "Audit deps" in p and "OWASP" in p and "read-only" in p
    assert "create and edit files" in build_system_prompt("W", "j", "", can_edit=True)


# ---- knowledge base ----

def test_knowledge_classify_and_persist(tmp_path, monkeypatch):
    import knowledge
    monkeypatch.setattr(knowledge, "KB_PATH", tmp_path / "kb.json")
    assert knowledge.classify("https://github.com/o/r.git") == {"kind": "repo", "location": "https://github.com/o/r.git"}
    assert knowledge.classify(str(tmp_path))["kind"] == "folder"
    assert knowledge.classify(str(tmp_path / "missing")) is None
    assert knowledge.classify("   ") is None
    knowledge.save([{"kind": "folder", "location": str(tmp_path)}])
    assert knowledge.load() == [{"kind": "folder", "location": str(tmp_path)}]
    assert knowledge.resolve_paths(knowledge.load()) == [str(tmp_path)]
    assert knowledge.resolve_paths([{"kind": "folder", "location": str(tmp_path / "gone")}]) == []


def test_knowledge_folders_are_readable_but_never_writable(tmp_path):
    from agents._common import make_repo_scoped_permission
    repo, kb = tmp_path / "repo", tmp_path / "kb"
    repo.mkdir(); kb.mkdir()
    check = make_repo_scoped_permission(repo, extra_read_roots=[str(kb)])
    assert type(check("Read", {"file_path": str(kb / "a.md")})).__name__ == "PermissionResultAllow"
    assert type(check("Write", {"file_path": str(kb / "a.md")})).__name__ == "PermissionResultDeny"
    assert type(check("Read", {"file_path": str(tmp_path / "other.md")})).__name__ == "PermissionResultDeny"
    assert type(check("Read", {"file_path": str(repo / "x.py")})).__name__ == "PermissionResultAllow"

# ---- chat-style agents (executor.py) ----

import threading
import time


def _wait(pred, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def _fresh_executor(monkeypatch, tmp_path, fake_run):
    import executor, knowledge
    monkeypatch.setattr(executor, "WORKDIR_ROOT", tmp_path)
    monkeypatch.setattr(knowledge, "resolve_paths", lambda s: ["/kb"])
    monkeypatch.setattr(executor, "_chats", {})
    monkeypatch.setattr(executor, "_memory", [])
    monkeypatch.setattr(executor, "MEMORY_PATH", tmp_path / "memory.json")
    monkeypatch.setattr(executor, "_ctx", {"catalog": [], "edges": [], "knowledge": []})
    monkeypatch.setattr(executor.custom_agent, "run", fake_run)
    return executor


AGENT = {"id": "a1", "name": "W", "job_description": "j", "skills": "k", "model": "claude-opus-5-5", "can_edit": True}


def test_send_runs_in_background_and_streams_the_trace_live(monkeypatch, tmp_path):
    release = threading.Event()
    seen = {}

    def fake_run(root, overview, spec, agent, upstream, note, on_event, model, kb, history, on_trace, cancel, ask=None, **kw):
        if not seen:                                    # the queued second message runs this too; keep the first call
            seen.update(root=root, spec=spec, model=model, kb=kb, history=history, upstream=upstream)
        on_trace({"type": "tool_use", "tool": "Read", "input": "x"})
        release.wait(3)
        return {"summary": "s", "output": "o", "files_changed": ["a.md"], "_trace": []}

    ex = _fresh_executor(monkeypatch, tmp_path, fake_run)
    assert ex.send(AGENT, "do X", [{"name": "R", "summary": "s0", "output": "o0"}], []) is True
    assert ex.is_running("a1") and ex.any_running()
    assert ex.send(AGENT, "again", [], []) is True                       # a chat: delivered to the working agent right away
    assert [t["status"] for t in ex.snapshot()["a1"]["turns"]] == ["running"]
    assert [m["text"] for m in ex.snapshot()["a1"]["turns"][0]["messages"]] == ["again"]
    assert _wait(lambda: ex.snapshot()["a1"]["turns"][0]["trace"])        # visible while still running
    assert ex.snapshot()["a1"]["turns"][0]["status"] == "running"
    release.set()
    assert _wait(lambda: not ex.any_running())
    turn = ex.snapshot()["a1"]["turns"][0]
    assert turn["status"] == "done" and turn["output"] == "o" and turn["files_changed"] == ["a.md"] and "seconds" in turn
    assert seen["root"] == tmp_path / "a1" and seen["spec"] == "do X" and seen["model"] == "claude-opus-5-5"
    assert seen["kb"] == ["/kb"] and seen["history"] == [] and seen["upstream"][0]["name"] == "R"
    assert ex.send(AGENT, "   ", [], []) is False                         # empty order ignored


def test_follow_up_continues_the_conversation(monkeypatch, tmp_path):
    histories = []

    def fake_run(*a, **k):
        histories.append([t["orders"] for t in a[9]])
        return {"summary": "s", "output": "o", "files_changed": []}

    ex = _fresh_executor(monkeypatch, tmp_path, fake_run)
    ex.send(AGENT, "first", [], []); assert _wait(lambda: not ex.is_running("a1"))
    ex.send(AGENT, "second", [], []); assert _wait(lambda: not ex.is_running("a1"))
    assert histories == [[], ["first"]]
    ex.new_chat("a1")
    assert "a1" not in ex.snapshot()


def test_stop_cancels_and_errors_are_reported_not_raised(monkeypatch, tmp_path):
    import llm

    def stoppable(*a, **k):
        cancel = a[11]
        cancel.wait(3)
        raise llm.Cancelled("stopped")

    ex = _fresh_executor(monkeypatch, tmp_path, stoppable)
    ex.send(AGENT, "go", [], [])
    ex.stop("a1")
    assert _wait(lambda: not ex.is_running("a1"))
    assert ex.snapshot()["a1"]["turns"][0]["status"] == "stopped"

    def boom(*a, **k):
        raise RuntimeError("boom")

    ex = _fresh_executor(monkeypatch, tmp_path, boom)
    ex.send(AGENT, "go", [], [])
    assert _wait(lambda: not ex.is_running("a1"))
    turn = ex.snapshot()["a1"]["turns"][0]
    assert turn["status"] == "error" and "boom" in turn["error"]


def test_upstream_is_the_latest_answer_of_the_agents_wired_into_this_one(monkeypatch, tmp_path):
    ex = _fresh_executor(monkeypatch, tmp_path, lambda *a, **k: {"summary": "sa", "output": "oa", "files_changed": []})
    for aid in ("a", "c"):
        ex.send({"id": aid, "name": aid.upper()}, "go", [], []); assert _wait(lambda: not ex.is_running(aid))
    catalog = [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}, {"id": "c", "name": "C"}]
    edges = [{"from": "a", "to": "b"}, {"from": "c", "to": "b"}, {"from": "a", "to": "c"}]
    assert ex.upstream_for("b", edges, catalog) == [
        {"name": "A", "summary": "sa", "output": "oa"}, {"name": "C", "summary": "sa", "output": "oa"}]
    assert ex.upstream_for("c", edges, catalog) == [{"name": "A", "summary": "sa", "output": "oa"}]
    assert ex.upstream_for("a", edges, catalog) == []                   # nothing wired into it
    assert ex.upstream_for("b", [{"from": "b", "to": "x"}], catalog) == []   # a wire out is not a wire in
    # a wired-in agent that hasn't answered yet contributes nothing
    assert ex.upstream_for("b", [{"from": "zzz", "to": "b"}], catalog) == []


def test_send_passes_wired_in_output_and_every_agent_gets_the_knowledge_base(monkeypatch, tmp_path):
    seen = {}

    def run(root, overview, spec, agent, upstream, note, on_event, model, kb, history, on_trace, cancel, ask=None, **kw):
        seen[agent["name"]] = (upstream, kb)
        return {"summary": "sum-" + agent["name"], "output": "out-" + agent["name"], "files_changed": []}

    ex = _fresh_executor(monkeypatch, tmp_path, run)
    cat = [dict(AGENT, id="a1", name="First"), dict(AGENT, id="a2", name="Second"), dict(AGENT, id="a3", name="Loner")]
    ex.set_context(cat, [{"from": "a1", "to": "a2"}], [{"kind": "folder", "location": "/kb"}])
    ex.send(cat[0], "do the first thing"); assert _wait(lambda: not ex.is_running("a1"))
    ex.send(cat[1], "build on it"); assert _wait(lambda: not ex.is_running("a2"))
    ex.send(cat[2], "unrelated"); assert _wait(lambda: not ex.is_running("a3"))
    assert seen["First"][0] == [] and seen["Second"][0] == [{"name": "First", "summary": "sum-First", "output": "out-First"}]
    assert seen["Loner"][0] == []
    assert all(kb == ["/kb"] for _, kb in seen.values())                 # the knowledge base reaches every agent
    hist = ex.memory_snapshot([])["history"]
    assert [h["context_from"] for h in hist] == [[], ["First"], []]


def test_llm_cancel_interrupts_a_waiting_run_and_is_not_retried(monkeypatch):
    import asyncio, llm
    calls = []

    async def slow_query(prompt, options):
        calls.append(1)
        await asyncio.sleep(30)
        yield None

    monkeypatch.setattr(llm, "query", slow_query)
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    started = time.time()
    try:
        llm.run_agent(agent_name="X", system_prompt="s", user_prompt="u", output_schema={}, cancel=cancel)
        assert False, "should have been cancelled"
    except llm.Cancelled:
        pass
    assert time.time() - started < 5 and len(calls) == 1


# ---- orchestrator: background memory ----

def test_memory_records_every_input_and_output_and_survives_clearing_chats(monkeypatch, tmp_path):
    outs = iter([{"summary": "s1", "output": "out one", "files_changed": ["a.md"]}, RuntimeError("boom")])

    def run(*a, **k):
        r = next(outs)
        if isinstance(r, Exception):
            raise r
        return r

    ex = _fresh_executor(monkeypatch, tmp_path, run)
    cat = [dict(AGENT, id="a1", name="Researcher"), dict(AGENT, id="a2", name="Writer")]
    ex.send(cat[0], "research X", [], []); assert _wait(lambda: not ex.is_running("a1"))
    ex.send(cat[1], "write it up", [], []); assert _wait(lambda: not ex.is_running("a2"))
    kb = [{"kind": "folder", "location": "/kb"}]
    mem = ex.memory_snapshot(kb)
    assert mem["knowledge"] == kb
    first, second = mem["history"]
    assert (first["agent"], first["input"], first["status"], first["output"]) == ("Researcher", "research X", "done", "out one")
    assert first["files_changed"] == ["a.md"] and "seconds" in first
    assert (second["agent"], second["input"], second["status"]) == ("Writer", "write it up", "error") and "boom" in second["error"]

    ex.new_chat("a1")                                   # clearing a chat doesn't erase the orchestrator's memory
    assert len(ex.memory_snapshot([])["history"]) == 2
    assert "a1" not in ex.snapshot()
    ex.clear_memory()
    assert ex.memory_snapshot([])["history"] == []


def test_memory_is_persisted_and_a_run_that_was_in_flight_is_marked_interrupted(monkeypatch, tmp_path):
    import json
    ex = _fresh_executor(monkeypatch, tmp_path, lambda *a, **k: {"summary": "s", "output": "o", "files_changed": []})
    ex.send(AGENT, "go", [], []); assert _wait(lambda: not ex.is_running("a1"))
    saved = json.loads((tmp_path / "memory.json").read_text())["history"]
    assert saved[0]["input"] == "go" and saved[0]["status"] == "done"

    (tmp_path / "memory.json").write_text(json.dumps({"history": [{"agent": "W", "input": "x", "status": "running"}]}))
    ex._memory.clear(); ex._load_memory()
    assert ex._memory[0]["status"] == "interrupted"


def test_parked_orchestrator_agent_still_cleans_its_output():
    from agents.orchestrator import clean
    out = clean({"reply": " hi ", "dispatch": [{"agent_id": "ghost", "orders": "x"}],
                 "questions": [{"question": "Pick", "options": ["a", "b"], "recommended_option": "zzz", "agent_id": None}]}, ["a1"])
    assert out["reply"] == "hi" and out["dispatch"] == [] and out["questions"][0]["recommended_option"] is None


# ---- agents ask the user directly (ask_user) ----

def test_clean_questions_repairs_and_drops():
    from agents.custom import clean_questions
    out = clean_questions([
        {"question": " Which  db? ", "options": ["PG", " SQLite ", "", "x", "y", "z"], "recommended_option": "PG"},
        {"question": "Yes?", "options": ["yes"]},
        {"question": "", "options": ["a", "b"]},
        {"question": "Pick", "options": ["a", "b"], "recommended_option": "zzz"},
        "junk",
    ])
    assert [q["question"] for q in out] == ["Which db?", "Pick"]
    assert out[0]["options"] == ["PG", "SQLite", "x", "y"] and out[0]["recommended_option"] == "PG"
    assert out[1]["recommended_option"] is None
    assert clean_questions(None) == []


def test_ask_user_tool_blocks_on_the_users_answers_and_returns_them():
    import asyncio
    from agents.custom import _make_ask_tool
    seen = {}

    def ask(questions):
        seen["q"] = questions
        return ["PG", ""]

    t = _make_ask_tool(ask)
    out = asyncio.run(t.handler({"questions": [
        {"question": "Which db?", "options": ["PG", "SQLite"]}, {"question": "Tests?", "options": ["yes", "no"]}]}))
    text = out["content"][0]["text"]
    assert "Q: Which db?\nA: PG" in text and "Q: Tests?\nA: no preference" in text
    assert seen["q"][0]["question"] == "Which db?"
    bad = asyncio.run(t.handler({"questions": [{"question": "x", "options": ["only one"]}]}))
    assert bad["is_error"] is True


def test_agent_waits_for_the_user_and_continues_from_the_answer(monkeypatch, tmp_path):
    produced = []

    def run(root, overview, spec, agent, upstream, note, on_event, model, kb, history, on_trace, cancel, ask=None, **kw):
        answers = ask([{"question": "Which db?", "options": ["PG", "SQLite"], "recommended_option": "PG"}])
        produced.append(answers)
        return {"summary": "used " + answers[0], "output": "done", "files_changed": []}

    ex = _fresh_executor(monkeypatch, tmp_path, run)
    ex.send(AGENT, "build it", [], [])
    assert _wait(lambda: ex.snapshot().get("a1", {"turns": [{}]})["turns"][0].get("waiting"))
    turn = ex.snapshot()["a1"]["turns"][0]
    assert turn["status"] == "running" and turn["waiting"]["questions"][0]["question"] == "Which db?"
    assert produced == [] and "output" not in turn                      # no output before it is answered
    qid = turn["waiting"]["id"]

    assert ex.answer_question("a1", "wrong-id", ["PG"]) is False
    assert ex.answer_question("a1", qid, [""]) is False                  # something must be answered
    assert ex.answer_question("a1", qid, ["Neither -- DuckDB"]) is True   # free text is fine
    assert _wait(lambda: not ex.is_running("a1"))
    done = ex.snapshot()["a1"]["turns"][0]
    assert done["status"] == "done" and done["summary"] == "used Neither -- DuckDB" and done["waiting"] is None
    assert done["qa"] == [{"questions": [{"question": "Which db?", "options": ["PG", "SQLite"], "recommended_option": "PG"}],
                           "answers": ["Neither -- DuckDB"]}]
    assert ex.memory_snapshot([])["history"][0]["qa"] == done["qa"]       # the orchestrator remembers the exchange
    assert ex.answer_question("a1", qid, ["PG"]) is False                # no longer waiting


def test_stopping_an_agent_that_is_waiting_for_the_user(monkeypatch, tmp_path):
    import llm

    def run(*a, ask=None, **k):
        ask([{"question": "Q?", "options": ["a", "b"]}])
        return {"summary": "s", "output": "o", "files_changed": []}

    ex = _fresh_executor(monkeypatch, tmp_path, run)
    ex.send(AGENT, "go", [], [])
    assert _wait(lambda: ex.snapshot().get("a1", {"turns": [{}]})["turns"][0].get("waiting"))
    ex.stop("a1")
    assert _wait(lambda: not ex.is_running("a1"))
    # run() let the Cancelled raised inside ask propagate, as the real agent loop does
    assert ex.snapshot()["a1"]["turns"][0]["status"] == "stopped"
    assert ex._waiting == {}


# ---- Jira & Confluence (atlassian.py) ----

def test_atlassian_parse_link_finds_site_and_project():
    import atlassian
    kan = "https://mythik-team-ydk5npyq.atlassian.net/jira/software/projects/KAN/list?jql=project%20%3D%20KAN%20ORDER%20BY%20cf%5B10019%5D%20ASC,"
    assert atlassian.parse_link(kan) == {"site": "https://mythik-team-ydk5npyq.atlassian.net", "project_key": "KAN"}
    assert atlassian.parse_link("x.atlassian.net/wiki/spaces/ENG/pages/1") == {"site": "https://x.atlassian.net", "project_key": ""}
    assert atlassian.parse_link("https://x.atlassian.net/browse/ABC-12")["project_key"] == "ABC"
    assert atlassian.parse_link("https://x.atlassian.net/jira?jql=project%20%3D%20ZED")["project_key"] == "ZED"
    assert atlassian.parse_link("nonsense") == {"site": "", "project_key": ""}
    assert atlassian.parse_link("") == {"site": "", "project_key": ""}


def test_atlassian_credentials_are_stored_privately_and_mode_controls_write_tools(tmp_path, monkeypatch):
    import atlassian, os, stat
    monkeypatch.setattr(atlassian, "CREDENTIALS_PATH", tmp_path / "c.json")
    assert atlassian.is_configured() is False
    atlassian.save("https://x.atlassian.net/", " me@x.com ", " tok ", "KAN")
    assert stat.S_IMODE(os.stat(tmp_path / "c.json").st_mode) == 0o600
    creds = atlassian.load()
    assert creds == {"site": "https://x.atlassian.net", "email": "me@x.com", "token": "tok", "project_key": "KAN"}
    assert atlassian.is_configured(creds)
    read = atlassian.mcp_server_config("read", creds)
    write = atlassian.mcp_server_config("write", creds)
    assert read["command"] == "uvx" and read["args"] == ["mcp-atlassian"]
    assert read["env"]["READ_ONLY_MODE"] == "true" and write["env"]["READ_ONLY_MODE"] == "false"
    assert read["env"]["CONFLUENCE_URL"] == "https://x.atlassian.net/wiki" and read["env"]["JIRA_API_TOKEN"] == "tok"
    assert "READ-ONLY" in atlassian.operator_prompt("read", creds) and "KAN" in atlassian.operator_prompt("read", creds)
    assert "create and update" in atlassian.operator_prompt("write", creds)
    atlassian.clear()
    assert atlassian.is_configured() is False


def test_atlassian_connection_test_reports_each_product(tmp_path):
    import atlassian, base64, http.server, threading

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            ok = self.headers.get("Authorization") == "Basic " + base64.b64encode(b"me@x.com:good").decode()
            if not ok:
                self.send_response(401); self.end_headers(); return
            if self.path.startswith("/wiki"):
                self.send_response(404); self.end_headers(); return
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(b'{"displayName": "Mo Mitra"}')
        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    site = f"http://127.0.0.1:{srv.server_port}"
    try:
        good = atlassian.test_connection({"site": site, "email": "me@x.com", "token": "good", "project_key": ""})
        assert good["jira"] == (True, "Connected as Mo Mitra")
        assert good["confluence"][0] is False and "Not found" in good["confluence"][1]
        bad = atlassian.test_connection({"site": site, "email": "me@x.com", "token": "wrong", "project_key": ""})
        assert bad["jira"] == (False, "Rejected: wrong email or API token.")
    finally:
        srv.shutdown()
    down = atlassian.test_connection({"site": "http://127.0.0.1:1", "email": "a", "token": "b", "project_key": ""}, timeout=2)
    assert down["jira"][0] is False and "Could not reach" in down["jira"][1]


# ---- Jira & Confluence, on demand ----

def _jira_stub(good_token="good"):
    import base64, http.server, threading

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            ok = self.headers.get("Authorization") == "Basic " + base64.b64encode(f"me@x.com:{good_token}".encode()).decode()
            self.send_response(200 if ok else 401)
            self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(b'{"displayName": "Mo"}')
        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"



def _atl_agent(**kw):
    return dict(AGENT, **kw)


def _agent_that_calls_the_gate(calls, mode="read", reason="list the open issues"):
    """A fake agent run that behaves like Claude calling atlassian_task midway."""
    def run(root, overview, spec, agent, upstream, note, on_event, model, kb, history, on_trace, cancel, ask=None, atlassian_gate=None, **kw):
        calls.append(("started", spec))
        if atlassian_gate is None:
            return {"summary": "no jira tool", "output": "-", "files_changed": []}
        ok, message = atlassian_gate(mode, reason)
        calls.append(("gate", ok, message))
        return {"summary": "gate=" + str(ok), "output": message or "done", "files_changed": []}
    return run


def test_jira_is_not_asked_for_until_the_agent_actually_needs_it(monkeypatch, tmp_path):
    import atlassian
    monkeypatch.setattr(atlassian, "CREDENTIALS_PATH", tmp_path / "c.json")
    calls = []
    ex = _fresh_executor(monkeypatch, tmp_path, lambda *a, **k: calls.append(k.get("atlassian_gate") is not None) or
                         {"summary": "stories drafted", "output": "1. ...", "files_changed": []})
    ex.send(_atl_agent(), "draft five stories for the login feature", [], [])
    assert _wait(lambda: not ex.is_running("a1"))
    turn = ex.snapshot()["a1"]["turns"][0]
    assert turn["status"] == "done" and turn.get("waiting") is None and calls == [True]   # tool available, nothing asked


def test_agent_asks_for_jira_credentials_only_when_it_calls_the_tool(monkeypatch, tmp_path):
    import atlassian
    monkeypatch.setattr(atlassian, "CREDENTIALS_PATH", tmp_path / "c.json")
    srv, site = _jira_stub()
    calls = []
    ex = _fresh_executor(monkeypatch, tmp_path, _agent_that_calls_the_gate(calls, "read", "list the open issues in KAN"))
    try:
        ex.send(_atl_agent(), "show me the open issues", [], [])
        waiting = lambda: (ex.snapshot().get("a1", {"turns": [{}]})["turns"][0].get("waiting") or {})
        assert _wait(lambda: waiting().get("kind") == "form")
        w = waiting()
        assert "list the open issues in KAN" in w["note"]                      # says why it needs access
        assert [f["name"] for f in w["fields"]] == ["site", "email", "token"] and w["fields"][2]["type"] == "password"
        assert ex.answer_credentials("a1", "nope", {"site": site, "email": "me@x.com", "token": "x"}) is False
        assert ex.answer_credentials("a1", w["id"], {"site": site, "email": "me@x.com", "token": ""}) is False
        assert ex.answer_question("a1", w["id"], ["x"]) is False

        bad = site + "/jira/software/projects/KAN/list"
        assert ex.answer_credentials("a1", w["id"], {"site": bad, "email": "me@x.com", "token": "bad"})
        assert _wait(lambda: waiting().get("error") and waiting()["id"] != w["id"])
        assert "wrong email or API token" in waiting()["error"] and not atlassian.is_configured()
        assert ex.answer_credentials("a1", waiting()["id"], {"site": bad, "email": "me@x.com", "token": "good"})
        assert _wait(lambda: not ex.is_running("a1"))
        assert atlassian.load() == {"site": site, "email": "me@x.com", "token": "good", "project_key": "KAN"}
        assert ("gate", True, "") in calls                                     # read: connected, no extra permission asked
        blob = json.dumps([ex.snapshot(), ex.memory_snapshot([])])
        assert "good" not in blob and "bad" not in blob                        # the token is never in the chat, trace or memory
    finally:
        srv.shutdown()


def test_giving_up_on_the_connection_tells_the_agent_and_stopping_cancels(monkeypatch, tmp_path):
    import atlassian
    monkeypatch.setattr(atlassian, "CREDENTIALS_PATH", tmp_path / "c.json")
    calls = []
    ex = _fresh_executor(monkeypatch, tmp_path, _agent_that_calls_the_gate(calls))
    ex.send(_atl_agent(), "go", [], [])
    assert _wait(lambda: (ex.snapshot().get("a1", {"turns": [{}]})["turns"][0].get("waiting") or {}).get("kind") == "form")
    ex.stop("a1")                                                              # the user doesn't want to connect
    assert _wait(lambda: not ex.is_running("a1"))
    assert ex.snapshot()["a1"]["turns"][0]["status"] == "stopped" and ex._waiting == {}


def test_a_message_during_the_credentials_form_declines_it_and_is_never_taken_as_the_token(monkeypatch, tmp_path):
    import atlassian
    monkeypatch.setattr(atlassian, "CREDENTIALS_PATH", tmp_path / "c.json")
    calls = []
    ex = _fresh_executor(monkeypatch, tmp_path, _agent_that_calls_the_gate(calls))
    ex.send(_atl_agent(), "first", [], [])
    assert _wait(lambda: (ex.snapshot().get("a1", {"turns": [{}]})["turns"][0].get("waiting") or {}).get("kind") == "form")
    assert ex.send(_atl_agent(), "not now, just show me the stories as text", [], []) is True
    assert _wait(lambda: not ex.any_running())
    ok_flag, message = calls[-1][1], calls[-1][2]
    assert ok_flag is False and "not now, just show me the stories as text" in message and "chose not to connect" in message
    assert not atlassian.is_configured() and len(ex.snapshot()["a1"]["turns"]) == 1

def test_writes_need_permission_each_time_unless_allowed_for_the_conversation(monkeypatch, tmp_path):
    import atlassian
    monkeypatch.setattr(atlassian, "CREDENTIALS_PATH", tmp_path / "c.json")
    atlassian.save("https://x.atlassian.net", "me@x.com", "tok", "KAN")      # already connected
    calls = []
    ex = _fresh_executor(monkeypatch, tmp_path, _agent_that_calls_the_gate(calls, "write", "create the 5 stories in KAN"))
    waiting = lambda: (ex.snapshot().get("a1", {"turns": [{}]})["turns"][-1].get("waiting") or {})

    def run_and_answer(choice_index=None, free_text=None):
        ex.send(_atl_agent(), "put them in jira", [], [])
        assert _wait(lambda: waiting().get("questions"))
        q = waiting()
        assert "wants to CHANGE Jira/Confluence" in q["questions"][0]["question"] and "create the 5 stories in KAN" in q["questions"][0]["question"]
        assert q["questions"][0]["options"] == ["Allow this once", "Allow for the rest of this conversation", "Don't allow"]
        if free_text:
            assert ex.send(_atl_agent(), free_text, [], [])
        else:
            assert ex.answer_question("a1", q["id"], [q["questions"][0]["options"][choice_index]])
        assert _wait(lambda: not ex.is_running("a1"))

    run_and_answer(2)                                                         # Don't allow
    assert calls[-1][:2] == ("gate", False) and "did not allow" in calls[-1][2]
    run_and_answer(0)                                                         # once
    assert calls[-1] == ("gate", True, "")
    run_and_answer(0)                                                         # asked again
    run_and_answer(1)                                                         # for the conversation
    assert calls[-1] == ("gate", True, "")
    n = len(calls)
    ex.send(_atl_agent(), "and one more", [], [])                              # no question this time
    assert _wait(lambda: not ex.is_running("a1")) and calls[n:] == [("started", "and one more"), ("gate", True, "")]
    ex.new_chat("a1")                                                          # a fresh conversation forgets the grant
    run_and_answer(0)
    ex.new_chat("a1")
    run_and_answer(free_text="Not yet -- only create the first two")           # words instead of a choice: nothing changes
    assert calls[-1][:2] == ("gate", False) and "Not yet -- only create the first two" in calls[-1][2]


def test_an_agent_set_to_never_has_no_jira_tool(monkeypatch, tmp_path):
    seen = []
    ex = _fresh_executor(monkeypatch, tmp_path, lambda *a, **k: seen.append(k.get("atlassian_gate")) or {"summary": "s", "output": "o", "files_changed": []})
    ex.send(_atl_agent(atlassian="never"), "go", [], []); assert _wait(lambda: not ex.is_running("a1"))
    ex.send(_atl_agent(atlassian="read"), "go", [], []); assert _wait(lambda: not ex.is_running("a1"))   # old saved values mean "on request"
    assert seen[0] is None and callable(seen[1])


def test_atlassian_task_tool_gates_then_runs_the_helper(monkeypatch):
    import asyncio, agents.custom as custom
    ran = []
    monkeypatch.setattr(custom, "_run_atlassian", lambda task, mode, on_trace, cancel: ran.append((task, mode)) or "Created KAN-12")
    verdicts = iter([(False, "The user did not allow changes."), (True, ""), (True, ""), (True, "")])
    gate_calls = []
    t = custom._make_atlassian_tool(lambda mode, reason: gate_calls.append((mode, reason)) or next(verdicts), None, None)
    out = asyncio.run(t.handler({"task": "create story X", "write": True, "reason": "user asked"}))
    assert out["content"][0]["text"] == "The user did not allow changes." and ran == []          # refused: helper never runs
    out = asyncio.run(t.handler({"task": "create story X", "write": True}))
    assert out["content"][0]["text"] == "Created KAN-12" and ran == [("create story X", "write")]
    asyncio.run(t.handler({"task": "list issues", "write": False}))
    assert ran[-1] == ("list issues", "read") and gate_calls[-1][0] == "read"
    assert asyncio.run(t.handler({"task": "  "}))["is_error"] is True
    monkeypatch.setattr(custom, "_run_atlassian", lambda *a: (_ for _ in ()).throw(RuntimeError("401")))
    err = asyncio.run(t.handler({"task": "x", "write": False}))
    assert err["is_error"] is True and "401" in err["content"][0]["text"]


def test_atlassian_tool_and_prompt_are_attached_only_when_a_gate_is_given(monkeypatch, tmp_path):
    import agents.custom as custom, llm
    seen = {}

    class R:
        result = {"summary": "s", "output": "o"}; tool_calls = []; trace = []

    monkeypatch.setattr(llm, "run_agent", lambda **kw: seen.update(kw) or R())
    custom.run(tmp_path, "", "x", {"name": "A", "job_description": "j"}, [], atlassian_gate=lambda m, r: (True, ""))
    assert "user_tools_atl" in seen["mcp_servers"] and "mcp__user_tools_atl__atlassian_task" in seen["tools"]
    assert "JIRA & CONFLUENCE (on request)" in seen["system_prompt"] and "draft them in your answer first" in seen["system_prompt"]
    seen.clear()
    custom.run(tmp_path, "", "x", {"name": "A", "job_description": "j"}, [])
    assert not seen["mcp_servers"] and "JIRA & CONFLUENCE" not in seen["system_prompt"]


# ---- it is a chat: messages at any time ----

def test_a_message_sent_while_an_agent_waits_on_questions_is_taken_as_its_reply(monkeypatch, tmp_path):
    got = []

    def run(root, overview, spec, agent, upstream, note, on_event, model, kb, history, on_trace, cancel, ask=None, **kw):
        reply = ask([{"question": "Which db?", "options": ["PG", "SQLite"]}, {"question": "Tests?", "options": ["yes", "no"]}])
        got.append(reply)
        return {"summary": "s", "output": "o", "files_changed": []}

    ex = _fresh_executor(monkeypatch, tmp_path, run)
    ex.send(AGENT, "build it", [], [])
    assert _wait(lambda: ex.snapshot().get("a1", {"turns": [{}]})["turns"][0].get("waiting"))
    assert ex.send(AGENT, "Actually skip the database, just use a JSON file", [], []) is True   # typed, not clicked
    assert _wait(lambda: not ex.is_running("a1")) and len(ex.snapshot()["a1"]["turns"]) == 1     # no extra turn: it answered
    answers, free_text = got[0]
    assert free_text == "Actually skip the database, just use a JSON file" and answers == ["", ""]
    turn = ex.snapshot()["a1"]["turns"][0]
    assert turn["status"] == "done" and turn["qa"][0]["free_text"] == free_text
    assert ex.memory_snapshot([])["history"][0]["qa"][0]["free_text"] == free_text


def test_free_text_reply_reaches_the_agent_through_the_ask_tool():
    import asyncio
    from agents.custom import _make_ask_tool
    t = _make_ask_tool(lambda qs: (["", ""], "use a JSON file instead"))
    text = asyncio.run(t.handler({"questions": [{"question": "Which db?", "options": ["PG", "SQLite"]}]}))["content"][0]["text"]
    assert "own words" in text and "use a JSON file instead" in text and "follow it" in text


def test_a_message_sent_while_the_agent_works_reaches_it_immediately_not_after(monkeypatch, tmp_path):
    release = threading.Event()
    got = []

    def run(root, overview, spec, agent, upstream, note, on_event, model, kb, history, on_trace, cancel, inbox=None, **kw):
        seen_inbox = []
        end = time.time() + 3
        while time.time() < end and len(seen_inbox) < 2:       # like the real agent loop: take what the user says while working
            try:
                seen_inbox.append(inbox.get(timeout=0.05))
            except Exception:
                pass
        got.append((spec, seen_inbox, [h["orders"] for h in history]))
        return {"summary": "s", "output": "o", "files_changed": []}

    ex = _fresh_executor(monkeypatch, tmp_path, run)
    ex.send(AGENT, "first", [], [])
    assert _wait(lambda: ex.is_running("a1"))
    ex.send(AGENT, "actually do it differently", [], []); ex.send(AGENT, "and keep it short", [], [])
    assert _wait(lambda: not ex.any_running())
    assert got == [("first", ["actually do it differently", "and keep it short"], [])]       # delivered to the SAME run, in order
    turns = ex.snapshot()["a1"]["turns"]
    assert [t["status"] for t in turns] == ["done"] and [m["text"] for m in turns[0]["messages"]] == ["actually do it differently", "and keep it short"]
    assert ex.memory_snapshot([])["history"][0]["messages"] == ["actually do it differently", "and keep it short"]


def test_a_message_that_arrives_as_the_run_finishes_is_not_lost(monkeypatch, tmp_path):
    specs = []

    def run(root, overview, spec, agent, upstream, note, on_event, model, kb, history, on_trace, cancel, inbox=None, **kw):
        specs.append(spec)
        if len(specs) == 1:
            ex.send(AGENT, "too late for the first run", [], [])   # lands in the inbox but the run never reads it
        return {"summary": "s", "output": "o", "files_changed": []}

    ex = _fresh_executor(monkeypatch, tmp_path, run)
    ex.send(AGENT, "first", [], [])
    assert _wait(lambda: len(specs) == 2 and not ex.any_running())
    assert specs == ["first", "too late for the first run"]


def test_stopping_an_agent_works_while_it_has_unread_messages(monkeypatch, tmp_path):
    import llm

    def run(root, overview, spec, agent, upstream, note, on_event, model, kb, history, on_trace, cancel, **kw):
        cancel.wait(3)
        raise llm.Cancelled("stopped")

    ex = _fresh_executor(monkeypatch, tmp_path, run)
    ex.send(AGENT, "go", [], [])
    assert _wait(lambda: ex.is_running("a1"))
    ex.send(AGENT, "hello?", [], [])
    ex.stop("a1")
    assert _wait(lambda: not ex.any_running())
    assert ex.snapshot()["a1"]["turns"][0]["status"] == "stopped"

def test_the_llm_streams_user_messages_into_a_running_agent_and_replays_them_on_retry(monkeypatch):
    import asyncio, llm, queue as q
    prompts = []
    rounds = []

    async def fake_query(prompt, options):
        rounds.append(1)
        if isinstance(prompt, str):
            prompts.append(prompt)
        else:
            first = await prompt.__anext__()
            prompts.append(first["message"]["content"])
            if len(rounds) == 1:
                second = await prompt.__anext__()                # the user's mid-task message arrives
                prompts.append(second["message"]["content"])
        if len(rounds) == 1:
            raise RuntimeError("flaky generation")               # first attempt fails after the user spoke
        yield llm.ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False, num_turns=1,
                                session_id="s", total_cost_usd=0.0, usage=None, result="", structured_output={"summary": "ok"})

    monkeypatch.setattr(llm, "query", fake_query)
    inbox = q.Queue(); inbox.put("change of plan")
    out = llm.run_agent(agent_name="X", system_prompt="s", user_prompt="do it", output_schema={}, inbox=inbox)
    assert out.result == {"summary": "ok"}
    assert prompts[0] == "do it" and prompts[1] == "change of plan"                       # delivered mid-run
    assert prompts[2].startswith("do it") and "change of plan" in prompts[2]              # replayed after the retry restarted


