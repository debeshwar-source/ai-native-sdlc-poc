"""
User-defined agent.

Unlike the other agents in this package, nothing about this one is fixed
in code: the user supplies its name, job description and skills in the
workflow builder's "Add agent" tab, and this module turns those into a
system prompt and runs it over the repo. It reads the repo (Read/Grep/
Glob, confined to the repo root) and sees what the agents before it in
the user's workflow produced. It may edit files only if the user ticked
"can edit files" on it.
"""

import asyncio
import threading
from typing import Any, Callable, Dict, List, Optional

from claude_agent_sdk import create_sdk_mcp_server, tool

import atlassian
import llm
from agents._common import format_note, make_repo_scoped_permission, written_paths_from_tool_calls

# Per upstream output, so a long chain of agents can't blow up the prompt.
MAX_UPSTREAM_CHARS = 6000

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "2-3 sentence summary of what you did and concluded",
        },
        "output": {
            "type": "string",
            "description": (
                "your full deliverable, in markdown -- what the next agent in the "
                "workflow (and the user) should read"
            ),
        },
    },
    "required": ["summary", "output"],
}


def build_system_prompt(name: str, job_description: str, skills: str, can_edit: bool) -> str:
    skills_text = skills.strip() or "(none listed)"
    edit_text = (
        "You may create and edit files inside your working directory when your job calls for it."
        if can_edit
        else "You are read-only: you can read and search your working directory (and any knowledge base) but not change anything."
    )
    return f"""
You are "{name}", one agent in a user-assembled workflow of AI agents. The user defined you and is talking
to you directly, as in a chat. The job and skills below are your role and expertise: they set what you do by
default and what you are best at. But this is a conversation -- answer follow-up questions, explain your
reasoning, take corrections and redirections, and discuss anything the user raises, the way a capable colleague
would, whether or not it falls inside your job description. Do not refuse a reasonable request just because it
is outside your role; if you are the wrong specialist for part of it, say so briefly and still help as far as
you sensibly can. Do not take actions beyond what the user asked for. For a plain conversational message, answer
it directly in `output` (and a one-line `summary`); no report format is needed.

============================================================
YOUR JOB
============================================================
{job_description.strip()}

============================================================
YOUR SKILLS
============================================================
{skills_text}

============================================================
RULES
============================================================
- {edit_text}
- Ground what you say in the files you can read (Read/Grep/Glob) and in the upstream agents' output;
  do not invent files, APIs or results.
- Follow any "OPERATOR INSTRUCTIONS" in the prompt: they come from the user and apply to you.
- Other agents may be connected downstream of you and will receive your output as context; you do not choose or name them.

============================================================
ASKING THE USER
============================================================
You have an `ask_user` tool that puts questions straight to the user and waits for their
answers. Use it when a decision or fact that only the user can supply would change what you
do -- and call it BEFORE doing the work that depends on the answer. Never guess at such a
thing, and do NOT produce your final output while a question of that kind is still unanswered:
ask first, then continue from the answers.
- Do not ask what you can find out yourself (Read/Grep/Glob) or decide sensibly on your own.
- Put related questions in ONE call (at most 4), each ONE short sentence stating the decision,
  with 2-4 concrete, mutually exclusive options (never a bare yes/no) and the option you
  recommend in `recommended_option`. The user may answer in their own words instead of picking
  an option; treat a free-text answer as a real alternative, not an error.
- If a question was already answered in this conversation, it is settled: use the answer.
"""


ASK_USER_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 4},
                    "recommended_option": {"type": "string"},
                },
                "required": ["question", "options"],
            },
        },
    },
    "required": ["questions"],
}


def clean_questions(questions: Any) -> List[Dict[str, Any]]:
    """Repairs/drops malformed entries of an ask_user call (at most 4)."""
    cleaned = []
    for q in questions if isinstance(questions, list) else []:
        if not isinstance(q, dict) or not str(q.get("question") or "").strip():
            continue
        options = [str(o).strip() for o in (q.get("options") or []) if str(o).strip()][:4]
        if len(options) < 2:
            continue
        rec = q.get("recommended_option")
        cleaned.append({"question": " ".join(str(q["question"]).split()), "options": options,
                        "recommended_option": rec if rec in options else None})
    return cleaned[:4]


def _make_ask_tool(ask: Callable[[List[Dict[str, Any]]], List[str]]):
    """`ask` blocks until the user has answered (executor.py); it runs in a
    worker thread so the agent's event loop stays free."""

    @tool(
        "ask_user",
        "Ask the user one or more multiple-choice questions and wait for their answers. "
        "Use it before doing work that depends on a decision only the user can make.",
        ASK_USER_SCHEMA,
    )
    async def ask_user(args: Dict[str, Any]) -> Dict[str, Any]:
        questions = clean_questions(args.get("questions"))
        if not questions:
            return {"content": [{"type": "text", "text": "No valid questions (each needs 2-4 options)."}],
                    "is_error": True}
        reply = await asyncio.get_running_loop().run_in_executor(None, ask, questions)
        answers, free_text = reply if isinstance(reply, tuple) else (reply, "")
        if free_text:
            # They chose to reply in their own words rather than pick options.
            text = ("The user did not pick from the options; they replied in their own words:\n\n"
                    f"\"{free_text}\"\n\nTreat this as their answer to whichever of your questions it addresses, "
                    "and if it changes or redirects the task, follow it.")
            return {"content": [{"type": "text", "text": text}]}
        lines = [f"Q: {q['question']}\nA: {a or 'no preference -- use your own best judgment'}"
                 for q, a in zip(questions, answers)]
        return {"content": [{"type": "text", "text": "The user answered:\n\n" + "\n\n".join(lines)}]}

    return ask_user


ATLASSIAN_TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "task": {"type": "string", "description": "What to do in Jira/Confluence, in full detail (project key, issue type, titles, descriptions...)"},
        "write": {"type": "boolean", "description": "true only if the task creates or changes anything"},
        "reason": {"type": "string", "description": "One sentence for the user on why you need this"},
    },
    "required": ["task", "write"],
}


def _run_atlassian(task: str, mode: str, on_trace, cancel) -> str:
    """The helper run: an agent with the Atlassian tools attached, doing one task."""
    from claude_agent_sdk import PermissionResultAllow

    creds = atlassian.load()

    result = llm.run_agent(
        agent_name="Jira & Confluence",
        system_prompt=atlassian.operator_prompt(mode, creds),
        user_prompt=task,
        output_schema={"type": "object", "properties": {"result": {"type": "string"}}, "required": ["result"]},
        required_keys=["result"],
        tools=[],
        can_use_tool=lambda t, i: PermissionResultAllow(),
        mcp_servers={"atlassian": atlassian.mcp_server_config(mode, creds)},
        on_trace=on_trace,
        cancel=cancel,
    )
    return result.result["result"]


def _make_atlassian_tool(gate: Callable[[str, str], Any], on_trace, cancel):
    """`gate(mode, reason)` (executor.py) blocks until the user has connected an
    account and, for writes, allowed the change; it returns (ok, message)."""

    @tool(
        "atlassian_task",
        "Do something in Jira or Confluence (search/read issues and pages, create stories, write pages...). "
        "Asks the user for a connection and for permission itself when needed.",
        ATLASSIAN_TASK_SCHEMA,
    )
    async def atlassian_task(args: Dict[str, Any]) -> Dict[str, Any]:
        task = str(args.get("task") or "").strip()
        if not task:
            return {"content": [{"type": "text", "text": "No task given."}], "is_error": True}
        mode = "write" if args.get("write") else "read"
        loop = asyncio.get_running_loop()
        ok, message = await loop.run_in_executor(None, gate, mode, str(args.get("reason") or task)[:300])
        if not ok:
            return {"content": [{"type": "text", "text": message}]}
        try:
            text = await loop.run_in_executor(None, _run_atlassian, task, mode, on_trace, cancel)
        except llm.Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001 -- reported to the agent, which tells the user
            return {"content": [{"type": "text", "text": f"The Jira/Confluence request failed: {exc}"}], "is_error": True}
        return {"content": [{"type": "text", "text": text}]}

    return atlassian_task


def _format_upstream(upstream: List[Dict[str, Any]]) -> str:
    if not upstream:
        return "(none -- no other agent is connected into you)"
    blocks = []
    for item in upstream:
        text = (item.get("output") or "")[:MAX_UPSTREAM_CHARS]
        blocks.append(f"### {item['name']}\nSummary: {item.get('summary', '')}\n\n{text}")
    return "\n\n".join(blocks)


def _format_conversation(turns: Optional[List[Dict[str, Any]]]) -> str:
    """Earlier exchanges with the user in this same chat, so a follow-up
    order can refer back to what the agent already did."""
    if not turns:
        return ""
    blocks = [
        f"User: {t['orders']}"
        + "".join(f"\nUser (while you were working): {m['text']}" for m in t.get("messages", []))
        + f"\nYou: {(t.get('output') or t.get('summary') or '')[:MAX_UPSTREAM_CHARS]}"
        for t in turns
    ]
    return f"""
============================================================
THIS CONVERSATION SO FAR (the new orders below follow on from it)
============================================================

{chr(10).join(blocks)}
"""


def _format_knowledge(paths: Optional[List[str]]) -> str:
    if not paths:
        return ""
    listing = "\n".join(f"- {p}" for p in paths)
    return f"""
============================================================
KNOWLEDGE BASE (read-only reference material outside the repo)
============================================================

Search and read these directories when they help; never write to them:
{listing}
"""


def run(
    root,
    repo_overview: str,
    spec_text: str,
    agent: Dict[str, Any],
    upstream: List[Dict[str, Any]],
    human_note: Optional[str] = None,
    on_event: Optional[Callable[[], None]] = None,
    model: Optional[str] = None,
    knowledge_paths: Optional[List[str]] = None,
    conversation: Optional[List[Dict[str, Any]]] = None,
    on_trace: Optional[Callable[[Dict[str, Any]], None]] = None,
    cancel: Optional[threading.Event] = None,
    ask: Optional[Callable[[List[Dict[str, Any]]], List[str]]] = None,
    atlassian_gate: Optional[Callable[[str, str], Any]] = None,
    inbox: Any = None,
) -> Dict[str, Any]:
    if not spec_text or not spec_text.strip():
        raise ValueError("Spec text cannot be empty.")

    can_edit = bool(agent.get("can_edit"))
    prompt = f"""
============================================================
WORKING CONTEXT (context only)
============================================================

{repo_overview}

============================================================
OUTPUT FROM THE AGENTS CONNECTED INTO YOU (their latest results)
============================================================

{_format_upstream(upstream)}
{_format_conversation(conversation)}{_format_knowledge(knowledge_paths)}{format_note(human_note)}
============================================================
THE USER'S ORDERS
============================================================

{spec_text}
"""
    servers: Dict[str, Any] = {}
    if ask:
        servers["user_tools"] = create_sdk_mcp_server("user_tools", tools=[_make_ask_tool(ask)])
    if atlassian_gate:
        servers["user_tools_atl"] = create_sdk_mcp_server(
            "user_tools_atl", tools=[_make_atlassian_tool(atlassian_gate, on_trace, cancel)])
        system_extra = atlassian.prompt_section()
    else:
        system_extra = ""

    agent_result = llm.run_agent(
        agent_name=agent["name"],
        system_prompt=build_system_prompt(
            agent["name"], agent.get("job_description", ""), agent.get("skills", ""), can_edit
        ) + system_extra,
        user_prompt=prompt,
        output_schema=OUTPUT_SCHEMA,
        required_keys=["summary", "output"],
        cwd=root,
        tools=["Read", "Grep", "Glob"] + (["Write", "Edit"] if can_edit else [])
        + (["mcp__user_tools__ask_user"] if ask else [])
        + (["mcp__user_tools_atl__atlassian_task"] if atlassian_gate else []),
        mcp_servers=servers or None,
        can_use_tool=make_repo_scoped_permission(root, extra_read_roots=knowledge_paths),
        on_event=on_event,
        model=model,
        add_dirs=knowledge_paths,
        on_trace=on_trace,
        cancel=cancel,
        inbox=inbox,
    )
    result = dict(agent_result.result)
    result["files_changed"] = written_paths_from_tool_calls(agent_result.tool_calls, root)
    result["_tool_calls"] = agent_result.tool_calls
    result["_trace"] = agent_result.trace
    return result
