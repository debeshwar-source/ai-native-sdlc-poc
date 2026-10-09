"""Chat-style execution of workspace agents, plus the Orchestrator's memory.

Every agent in the workspace is a conversation. Sending it an order starts a
background thread that runs it (agents/custom.py) and records every step as it
happens, so the UI can show the work live and let the user stop it. A follow-up
order continues the same conversation.

The Orchestrator lives in the background and, for now, only remembers: the
knowledge base, and every input given to every agent with the output it
produced, in the order things happened (memory_snapshot()). That record is
append-only, kept on disk (orchestrator_memory.json) and deliberately separate
from the agent chats, so clearing a chat or the workspace doesn't erase it.

There is no repository: an agent works in its own scratch directory
(agent_workdir/<agent id>/, the only place it may write, and only if it was
created with "can edit files") and may read the knowledge base.

State is held in memory in this process (shared by every Streamlit rerun),
guarded by one lock; snapshot() is the JSON-able view the UI renders.
"""

import copy
import json
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import agents.custom as custom_agent
import atlassian
import config
import knowledge
import llm

WORKDIR_ROOT = Path(__file__).parent / "agent_workdir"
MEMORY_PATH = Path(__file__).parent / "orchestrator_memory.json"

NO_REPO_OVERVIEW = (
    "(There is no repository for this task. You work in an empty scratch directory; "
    "reference material, if any, is in the knowledge base below.)"
)

class _Declined(Exception):
    """The user answered the Jira connection form with a message instead."""


_lock = threading.RLock()
# agent id -> {"turns": [turn, ...], "cancel": Event | None}
_chats: Dict[str, Dict[str, Any]] = {}
# agent id -> the question the agent is waiting on: {"id", "event", "answers"}
_waiting: Dict[str, Dict[str, Any]] = {}
# The Orchestrator's memory: one entry per order given to an agent, oldest first.
_memory: List[Dict[str, Any]] = []
# The workspace as the UI last reported it (set_context), the defaults for send().
_ctx: Dict[str, Any] = {"catalog": [], "edges": [], "knowledge": []}


def _load_memory() -> None:
    try:
        data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        entries = [e for e in data.get("history", []) if isinstance(e, dict)]
    except (OSError, ValueError, AttributeError):
        return
    for e in entries:
        if e.get("status") == "running":   # the process that was running it is gone
            e["status"] = "interrupted"
    _memory[:] = entries


def _save_memory() -> None:
    """Best-effort: a failed write must never break an agent run."""
    try:
        MEMORY_PATH.write_text(json.dumps({"history": _memory}, indent=1), encoding="utf-8")
    except OSError:
        pass


_load_memory()


def _chat(chat_id: str) -> Dict[str, Any]:
    return _chats.setdefault(chat_id, {"turns": [], "cancel": None})


def set_context(catalog: List[Dict[str, Any]], edges: List[Dict[str, str]], knowledge_sources: List[Dict[str, str]]) -> None:
    """The workspace as the UI last reported it: the agent catalog, the wires
    between agents, and the knowledge base -- what send() falls back to."""
    with _lock:
        _ctx.update(catalog=copy.deepcopy(catalog), edges=copy.deepcopy(edges), knowledge=copy.deepcopy(knowledge_sources))


def _running_turn(chat: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return next((t for t in chat["turns"] if t["status"] == "running"), None)


def is_running(agent_id: str) -> bool:
    with _lock:
        return _running_turn(_chat(agent_id)) is not None


def any_running() -> bool:
    with _lock:
        return any(_running_turn(c) is not None for c in _chats.values())


def snapshot() -> Dict[str, Any]:
    """{agent id: {"turns": [...]}} -- a deep copy, safe to hand to the UI."""
    with _lock:
        return {aid: {"turns": copy.deepcopy(c["turns"])} for aid, c in _chats.items() if c["turns"]}


def latest_answers() -> Dict[str, Optional[Dict[str, Any]]]:
    """Each agent's most recent finished turn, for upstream context."""
    with _lock:
        return {
            aid: next((t for t in reversed(c["turns"]) if t["status"] == "done"), None)
            for aid, c in _chats.items()
        }


def upstream_for(agent_id: str, edges: List[Dict[str, str]], catalog) -> List[Dict[str, Any]]:
    """Latest answers of the agents wired into `agent_id` (an edge
    {"from": X, "to": agent_id}), in the order the wires were drawn. An agent
    that hasn't answered yet contributes nothing."""
    names = {a["id"]: a["name"] for a in catalog}
    answers = latest_answers()
    upstream = []
    for edge in edges:
        if edge["to"] != agent_id:
            continue
        turn = answers.get(edge["from"])
        if turn:
            upstream.append({"name": names.get(edge["from"], edge["from"]),
                             "summary": turn["summary"], "output": turn["output"]})
    return upstream


def memory_snapshot(knowledge_sources: List[Dict[str, str]]) -> Dict[str, Any]:
    """Everything the Orchestrator knows, for display: the knowledge base and
    the history of inputs given to agents and the outputs they produced."""
    with _lock:
        return {"knowledge": copy.deepcopy(knowledge_sources), "history": copy.deepcopy(_memory)}


def clear_memory() -> None:
    with _lock:
        _memory.clear()
        _save_memory()


def answer_question(agent_id: str, question_id: str, answers: List[Optional[str]]) -> bool:
    """The user's answers (one per question; empty = no preference) to what
    `agent_id` is waiting on. The agent then carries on from them. False if
    it isn't waiting on that question, or nothing was actually answered."""
    with _lock:
        slot = _waiting.get(agent_id)
        turn = _running_turn(_chat(agent_id))
        waiting = turn and turn.get("waiting")
        if not slot or not waiting or waiting["id"] != question_id or waiting.get("kind") == "form":
            return False
        if len(answers) != len(waiting["questions"]) or not any((a or "").strip() for a in answers):
            return False
        slot["answers"] = [(a or "").strip() for a in answers]
        slot["event"].set()
        return True


def answer_credentials(agent_id: str, question_id: str, values: Dict[str, str]) -> bool:
    """The user's reply to a credentials form (see _ensure_atlassian). The
    agent's thread validates it; False here only if it isn't waiting on that
    form or a field is empty."""
    with _lock:
        slot = _waiting.get(agent_id)
        turn = _running_turn(_chat(agent_id))
        waiting = turn and turn.get("waiting")
        if not slot or not waiting or waiting["id"] != question_id or waiting.get("kind") != "form":
            return False
        values = {k: str((values or {}).get(k) or "").strip() for k in ("site", "email", "token")}
        if not all(values.values()):
            return False
        slot["values"] = values
        slot["event"].set()
        return True


def new_chat(agent_id: str) -> None:
    """Clear an agent's conversation (stopping it first if running). The
    Orchestrator's memory of what it did is kept."""
    stop(agent_id)
    with _lock:
        _chats[agent_id] = {"turns": [], "cancel": None}


def stop(agent_id: str) -> None:
    """Stop the running order, and drop any messages still queued behind it."""
    with _lock:
        chat = _chat(agent_id)
        chat["turns"][:] = [t for t in chat["turns"] if t["status"] != "queued"]
        cancel = chat["cancel"]
    if cancel is not None:
        cancel.set()


def send(
    agent: Dict[str, Any],
    orders: str,
    upstream: Optional[List[Dict[str, Any]]] = None,
    knowledge_sources: Optional[List[Dict[str, str]]] = None,
) -> bool:
    """A message from the user to `agent`, at any time -- it is a chat:
    - idle: it starts a run in the background;
    - waiting on questions it asked (ask_user): the message is taken as the
      user's reply to them, in their own words;
    - waiting on the Jira connection form: the message declines the form and
      the agent carries on from what the user said;
    - working: the message is delivered to the running agent immediately, the
      way a message typed to Claude mid-task is, so it can change course.
    False only if the message is empty."""
    orders = orders.strip()
    if not orders:
        return False
    with _lock:
        chat = _chat(agent["id"])
        running = _running_turn(chat)
        if running:
            waiting, slot = running.get("waiting"), _waiting.get(agent["id"])
            if waiting and slot and waiting.get("kind") == "form":
                slot["declined"] = orders
                slot["event"].set()
                return True
            if waiting and slot:
                slot["free_text"] = orders
                slot["answers"] = [""] * len(waiting["questions"])
                slot["event"].set()
                return True
            inbox = chat.get("inbox")
            if inbox is not None:
                running.setdefault("messages", []).append({"text": orders, "at": time.time()})
                for e in _memory:
                    if e.get("agent_id") == agent["id"] and e.get("status") == "running":
                        e.setdefault("messages", []).append(orders)
                inbox.put(orders)
                return True
            chat["turns"].append({"orders": orders, "status": "queued", "trace": [], "started": time.time()})
            return True
        turn = {"orders": orders, "status": "running", "trace": [], "started": time.time()}
        chat["turns"].append(turn)
    _launch(agent, turn, upstream, knowledge_sources)
    return True


def _launch(agent: Dict[str, Any], turn: Dict[str, Any], upstream, knowledge_sources) -> None:
    """Run `turn` (already in the agent's chat, status "running") on a thread;
    when it ends, the next queued message, if any, is started."""
    orders = turn["orders"]
    with _lock:
        chat = _chat(agent["id"])
        history = [t for t in chat["turns"] if t["status"] == "done"]
        cancel = threading.Event()
        inbox: "queue.Queue[str]" = queue.Queue()
        chat["cancel"] = cancel
        chat["inbox"] = inbox
        if upstream is None:
            upstream = upstream_for(agent["id"], _ctx["edges"], _ctx["catalog"])
        if knowledge_sources is None:
            knowledge_sources = _ctx["knowledge"]
        turn["started"] = time.time()
        entry = {"agent_id": agent["id"], "agent": agent["name"], "input": orders,
                 "context_from": [u["name"] for u in upstream],
                 "status": "running", "started": turn["started"], "output": None}
        _memory.append(entry)
        _save_memory()

    def _on_trace(event: Dict[str, Any]) -> None:
        with _lock:
            turn["trace"].append(event)

    def _ask(questions: List[Dict[str, Any]]) -> List[str]:
        """Called from the agent's thread by its ask_user tool: show the
        questions in the agent's chat and block until the user answers."""
        slot = {"event": threading.Event(), "answers": None, "free_text": None}
        qid = uuid.uuid4().hex[:8]
        with _lock:
            _waiting[agent["id"]] = slot
            turn["waiting"] = {"id": qid, "questions": questions}
        try:
            while not slot["event"].wait(0.2):
                if cancel.is_set():
                    raise llm.Cancelled("stopped while waiting for the user")
        finally:
            with _lock:
                _waiting.pop(agent["id"], None)
                turn["waiting"] = None
        answers, free_text = slot["answers"], slot["free_text"]
        with _lock:
            record = {"questions": questions, "answers": answers}
            if free_text:
                record["free_text"] = free_text
            turn.setdefault("qa", []).append(record)
            entry.setdefault("qa", []).append(record)
            _save_memory()
        return (answers, free_text) if free_text else answers

    def _ensure_atlassian(reason: str) -> None:
        """The agent has just asked to use Jira & Confluence and no account is
        connected: ask the user for one, right in the agent's chat. The token is
        checked against Jira before it is saved, and is never put in the chat,
        the trace or the Orchestrator's memory."""
        if atlassian.is_configured():
            return
        saved = atlassian.load()
        error = ""
        for _attempt in range(3):
            slot = {"event": threading.Event(), "values": None, "declined": None}
            qid = uuid.uuid4().hex[:8]
            with _lock:
                _waiting[agent["id"]] = slot
                turn["waiting"] = {
                    "id": qid, "kind": "form", "title": "Connect Jira & Confluence",
                    "note": (f"{agent['name']} needs to reach your Jira & Confluence to: {reason}\n"
                             "Paste a link from your site, your Atlassian email and an API token "
                             "(create one at " + atlassian.TOKEN_URL + ")."),
                    "error": error,
                    "fields": [
                        {"name": "site", "label": "Jira or Confluence link", "type": "text",
                         "value": saved["site"], "placeholder": "https://your-team.atlassian.net/jira/software/projects/KEY/list"},
                        {"name": "email", "label": "Atlassian account email", "type": "text", "value": saved["email"]},
                        {"name": "token", "label": "API token", "type": "password", "value": ""},
                    ],
                }
            try:
                while not slot["event"].wait(0.2):
                    if cancel.is_set():
                        raise llm.Cancelled("stopped while waiting for the user")
            finally:
                with _lock:
                    _waiting.pop(agent["id"], None)
                    turn["waiting"] = None
            if slot.get("declined"):
                raise _Declined(slot["declined"])
            v = slot["values"]
            parsed = atlassian.parse_link(v["site"])
            if not parsed["site"]:
                error = "That doesn't look like an Atlassian link or site URL."
                continue
            creds = {"site": parsed["site"], "email": v["email"], "token": v["token"],
                     "project_key": parsed["project_key"] or saved["project_key"]}
            ok, message = atlassian.test_connection(creds)["jira"]
            if ok:
                atlassian.save(creds["site"], creds["email"], creds["token"], creds["project_key"])
                _on_trace({"type": "text", "text": f"Connected to Jira & Confluence at {creds['site']} ({message})."})
                return
            error = message
        raise RuntimeError("Couldn't connect to Jira & Confluence: " + error)

    def _atlassian_gate(mode: str, reason: str):
        """Called (from the agent's thread) when the agent first needs Jira or
        Confluence -- not before. Connects an account if none is, and for any
        change asks the user's permission. Returns (ok, message for the agent)."""
        try:
            _ensure_atlassian(reason)
        except llm.Cancelled:
            raise
        except _Declined as said:
            return False, ("The user chose not to connect Jira/Confluence just now and said instead: "
                           f"\"{said}\". Do not call the Jira tool again unless they ask; follow what they said.")
        except RuntimeError as exc:
            return False, str(exc) + " Tell the user you could not reach Jira/Confluence."
        if mode != "write":
            return True, ""
        with _lock:
            if _chat(agent["id"]).get("write_ok"):
                return True, ""
        once, convo, deny = "Allow this once", "Allow for the rest of this conversation", "Don't allow"
        reply = _ask([{
            "question": f"{agent['name']} wants to CHANGE Jira/Confluence: {reason}. Allow it?",
            "options": [once, convo, deny], "recommended_option": once,
        }])
        answers, free_text = reply if isinstance(reply, tuple) else (reply, "")
        if free_text:
            return False, ("The user did not allow or deny it, but replied in their own words: "
                           f"\"{free_text}\". Make no change now; follow what they said.")
        if answers and answers[0] == convo:
            with _lock:
                _chat(agent["id"])["write_ok"] = True
            return True, ""
        if answers and answers[0] == once:
            return True, ""
        return False, "The user did not allow changes to Jira/Confluence. Do not retry; tell them nothing was changed."

    def _work() -> None:
        try:
            workdir = WORKDIR_ROOT / agent["id"]
            workdir.mkdir(parents=True, exist_ok=True)
            result = custom_agent.run(
                workdir, NO_REPO_OVERVIEW, orders,
                {"name": agent["name"], "job_description": agent.get("job_description", ""),
                 "skills": agent.get("skills", ""), "can_edit": bool(agent.get("can_edit"))},
                upstream, None, None, agent.get("model") or config.MODEL_ID,
                knowledge.resolve_paths(knowledge_sources) or None,
                history, _on_trace, cancel, ask=_ask,
                atlassian_gate=None if agent.get("atlassian") == "never" else _atlassian_gate,
                inbox=inbox,
            )
            update = {
                "status": "done", "summary": result.get("summary", ""),
                "output": result.get("output", ""), "files_changed": result.get("files_changed", []),
            }
        except llm.Cancelled:
            update = {"status": "stopped"}
        except Exception as exc:  # noqa: BLE001 -- shown to the user in the chat
            update = {"status": "error", "error": str(exc)}
        with _lock:
            turn.update(update)
            turn["seconds"] = round(time.time() - turn["started"])
            entry.update(
                status=update["status"], seconds=turn["seconds"],
                output=update.get("output") if update["status"] == "done" else None,
                summary=update.get("summary"), files_changed=update.get("files_changed", []),
                error=update.get("error"),
            )
            _save_memory()
            chat_now = _chat(agent["id"])
            chat_now["inbox"] = None
            # A message that arrived just as the run was finishing was never seen by the agent:
            # it starts the next order rather than being lost.
            while True:
                try:
                    late = inbox.get_nowait()
                except queue.Empty:
                    break
                if update["status"] != "stopped":      # after a Stop the user is in charge: nothing runs on its own
                    chat_now["turns"].append({"orders": late, "status": "queued", "trace": [], "started": time.time()})
            nxt = next((t for t in chat_now["turns"] if t["status"] == "queued"), None)
            if nxt is not None:
                nxt["status"] = "running"
        if nxt is not None:
            # edits made to the agent in the meantime apply to the queued message
            latest = next((a for a in _ctx["catalog"] if a["id"] == agent["id"]), agent)
            _launch(latest, nxt, None, None)

    threading.Thread(target=_work, daemon=True, name=f"agent-{agent['id']}").start()
