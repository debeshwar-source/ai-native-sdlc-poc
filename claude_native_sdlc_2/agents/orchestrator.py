"""
Orchestrator Agent. (PARKED: not wired into the app for now -- the Orchestrator
currently only keeps memory, see executor.memory_snapshot. Kept for when it
gets a job again.)

Always present, never removed. It holds the total context of the task at
hand and is the one layer of communication between the user and the
workspace agents: the user talks to it, it hands orders to the agents,
and when an agent has open questions they come to it, and it surfaces to
the user whatever it can't settle from context -- as multiple-choice
questions with a recommended option (the user can always answer in their
own words instead). One structured call, no tools: it reasons over the
conversation and the agents' work, it does not do the work.
"""

import threading
from typing import Any, Callable, Dict, List, Optional

import llm

MAX_CHARS_PER_AGENT = 3000
MAX_QUESTIONS = 4

SYSTEM_PROMPT = """
You are the Orchestrator of a user-assembled team of AI agents. You hold the
total context of the task at hand, and you are the ONLY channel between the
user and the agents: the user talks to you, you talk to the agents. You do
not do the agents' work yourself -- you direct it, keep track of it, and keep
the user informed and in control.

============================================================
HOW YOU WORK
============================================================
- Be concise and direct. Plain language, no filler.
- Hand work to agents through `dispatch`: each entry is an agent_id from the
  ROSTER and the exact orders for it. Give an agent only what it needs, and
  include the relevant context from the task and from other agents' work --
  agents cannot see each other or this conversation. Only dispatch when
  there is something for an agent to do now; dispatching nothing is normal.
- When an agent has OPEN QUESTIONS, first try to settle each one from what
  you already know (the task, the user's earlier answers, other agents'
  output). If you can, answer it by dispatching orders to that agent that
  state the answer. Only what you genuinely cannot settle goes to the user.
- Ask the user the way a careful colleague does: multiple choice. Each entry
  in `questions` is ONE short decision with 2-4 concrete, mutually exclusive
  options (never a bare yes/no) and the option you recommend in
  `recommended_option`. Set `agent_id` to the agent whose work the answer
  unblocks, or null if it is your own question. The user can always pick
  something else and answer in their own words, so do not add an "other"
  option yourself and never pad with questions to seem thorough. Do not ask
  what you could decide sensibly yourself, and never ask the same thing twice.
- `reply` is what the user reads: say what happened and what you did or are
  waiting on. If you are asking questions, say briefly why they matter. Do not
  repeat the questions' text in the reply.
- Ground everything in the ROSTER, the TASK CONTEXT and the AGENT WORK shown
  below. Never invent what an agent did or found.
"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string", "description": "what the user should read, in markdown"},
        "dispatch": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"agent_id": {"type": "string"}, "orders": {"type": "string"}},
                "required": ["agent_id", "orders"],
            },
        },
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "recommended_option": {"type": "string"},
                    "agent_id": {"type": ["string", "null"]},
                },
                "required": ["question", "options", "recommended_option", "agent_id"],
            },
        },
    },
    "required": ["reply", "dispatch", "questions"],
}


def _roster(agents: List[Dict[str, Any]]) -> str:
    if not agents:
        return "(no agents are in the workspace yet)"
    return "\n".join(
        f"- id={a['id']} | {a['name']} | job: {a.get('job_description', '')} | skills: {a.get('skills') or '-'}"
        f" | {'can edit files' if a.get('can_edit') else 'read-only'}"
        for a in agents
    )


def _agent_work(work: Dict[str, List[Dict[str, Any]]], names: Dict[str, str]) -> str:
    blocks = []
    for agent_id, turns in work.items():
        lines = []
        for t in turns:
            lines.append(f"  orders: {t['orders']}")
            if t["status"] == "done":
                lines.append(f"  result: {(t.get('output') or t.get('summary') or '')[:MAX_CHARS_PER_AGENT]}")
            else:
                lines.append(f"  ({t['status']})")
        blocks.append(f"[{names.get(agent_id, agent_id)}]\n" + "\n".join(lines))
    return "\n\n".join(blocks) or "(no agent has done anything yet)"


def build_prompt(
    agents: List[Dict[str, Any]],
    transcript: str,
    work: Dict[str, List[Dict[str, Any]]],
    open_questions: str,
    new_item: str,
) -> str:
    names = {a["id"]: a["name"] for a in agents}
    return f"""
============================================================
ROSTER (the agents in the workspace you can direct)
============================================================
{_roster(agents)}

============================================================
TASK CONTEXT (this conversation so far, oldest first)
============================================================
{transcript or '(nothing yet)'}

============================================================
AGENT WORK SO FAR
============================================================
{_agent_work(work, names)}

============================================================
QUESTIONS ALREADY WAITING ON THE USER (do not ask these again)
============================================================
{open_questions or '(none)'}

============================================================
WHAT JUST HAPPENED -- respond to this
============================================================
{new_item}
"""


def clean(result: Dict[str, Any], agent_ids: List[str]) -> Dict[str, Any]:
    """Repairs/drops malformed entries rather than failing the run over a
    secondary field; dispatches to unknown agents are dropped."""
    dispatch = [
        {"agent_id": d["agent_id"], "orders": d["orders"].strip()}
        for d in (result.get("dispatch") or [])
        if isinstance(d, dict) and d.get("agent_id") in agent_ids
        and isinstance(d.get("orders"), str) and d["orders"].strip()
    ]
    questions = []
    for q in result.get("questions") or []:
        if not isinstance(q, dict) or not q.get("question"):
            continue
        options = [str(o).strip() for o in (q.get("options") or []) if str(o).strip()]
        if len(options) < 2:
            continue
        rec = q.get("recommended_option")
        agent_id = q.get("agent_id")
        questions.append({
            "question": " ".join(str(q["question"]).split()),
            "options": options[:4],
            "recommended_option": rec if rec in options[:4] else None,
            "agent_id": agent_id if agent_id in agent_ids else None,
        })
    return {"reply": str(result.get("reply") or "").strip(), "dispatch": dispatch,
            "questions": questions[:MAX_QUESTIONS]}


def run(
    agents: List[Dict[str, Any]],
    transcript: str,
    work: Dict[str, List[Dict[str, Any]]],
    open_questions: str,
    new_item: str,
    model: Optional[str] = None,
    on_trace: Optional[Callable[[Dict[str, Any]], None]] = None,
    cancel: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    agent_result = llm.run_agent(
        agent_name="Orchestrator",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_prompt(agents, transcript, work, open_questions, new_item),
        output_schema=OUTPUT_SCHEMA,
        required_keys=["reply", "dispatch", "questions"],
        model=model,
        on_trace=on_trace,
        cancel=cancel,
    )
    result = clean(agent_result.result, [a["id"] for a in agents])
    result["_trace"] = agent_result.trace
    return result
