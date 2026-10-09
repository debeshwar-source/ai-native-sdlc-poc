"""
Coding Agent.

Agentic: reads only what it needs, writes only the files the
Architecture Agent named, and can self-check with a scoped run_tests
tool before declaring itself finished. Deletion is a real, explicit
tool (not a Bash rm) -- same reasoning as the original Bedrock build:
an explicit, auditable delete_file call, never worked around by
emptying a file's contents or leaving it in place.
"""

from pathlib import Path
from typing import Callable, Any, Dict, List, Optional

import context.indexer as indexer
import llm
import tools.test_runner as test_runner
from agents._common import (
    CLARIFYING_QUESTIONS_GUIDANCE_TEXT,
    format_dict,
    format_note,
    format_requirement,
    format_work_order,
    is_dependency_manifest_path,
    make_repo_scoped_permission,
    validate_clarifying_questions,
    with_clarifying_questions_schema,
    written_paths_from_tool_calls,
)
from agents.quality import is_test_path
from claude_agent_sdk import create_sdk_mcp_server, tool

SYSTEM_PROMPT = """
You are the Coding Agent in a Claude-Native SDLC system.

Your job is to implement (or correct) the change described by the
Architecture Agent's plan, in the REAL repository, using your tools.

You may write ONLY to the files the Architecture Agent listed under
files_to_modify/files_to_create, and delete ONLY the files listed
under files_to_delete, using the delete_file tool. Any other write
path is rejected. Before modifying an existing file, read it first --
never guess its current contents.

You have a real delete_file tool. If the architecture plan lists a
file under files_to_delete, actually call delete_file on it -- do
not work around it by leaving the file in place, emptying its
contents, or explaining in your summary that deletion isn't possible.

Rules:
- Make the smallest change that satisfies the requirement and the
  architecture plan. Do not refactor unrelated code.
- Do not modify test files. If a test looks wrong, say so in your
  summary -- do not weaken it to make it pass.
- Preserve existing behavior, conventions, and style you observe in
  the surrounding code.
- Use run_tests to self-check your work before finishing, and keep
  iterating if it fails, unless FAILURE ANALYSIS below tells you the
  failure is not your responsibility (e.g. a test defect).

If FAILURE ANALYSIS is supplied, this is a correction pass: use it
as a diagnosis, but verify it yourself against the actual code and
test output before acting on it.

If FAILURE ANALYSIS names dependency_fix_files, you are ALSO
authorized to write to exactly those existing dependency-manifest
file(s) for this pass, on top of your normal scope -- but ONLY to
apply the specific fix it diagnosed (e.g. adding one missing package),
never as an opportunity to touch anything else in that file or make
an unrelated change.

When you believe the implementation is complete, provide your final
summary.

This is a fixed pipeline: the formal Testing stage runs next
automatically once you finish -- that's what actually gates review,
even after your own self-check passed. You do not choose or name it.
""" + CLARIFYING_QUESTIONS_GUIDANCE_TEXT

OUTPUT_SCHEMA = with_clarifying_questions_schema({
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "self_check_passed": {"type": "boolean"},
    },
    "required": ["summary", "self_check_passed"],
})


def _make_tools(root: Path, allowed_delete_paths: set):
    @tool("delete_file", "Delete a file that the architecture plan listed under files_to_delete.", {"path": str})
    async def delete_file(args: Dict[str, Any]) -> Dict[str, Any]:
        rel_path = args["path"]
        if rel_path not in allowed_delete_paths:
            return {
                "content": [{
                    "type": "text",
                    "text": f"'{rel_path}' is not in files_to_delete; refusing to delete it.",
                }],
                "is_error": True,
            }
        target = root / rel_path
        if not target.is_file():
            return {
                "content": [{"type": "text", "text": f"'{rel_path}' does not exist."}],
                "is_error": True,
            }
        target.unlink()
        return {"content": [{"type": "text", "text": f"Deleted {rel_path}."}]}

    @tool("run_tests", "Run the repository's test suite and see the results. Use this to self-check your work before finishing.", {})
    async def run_tests(_args: Dict[str, Any]) -> Dict[str, Any]:
        result = test_runner.run_tests(root)
        text = (
            f"status={result['status']} command={result['command']}\n"
            f"--- stdout ---\n{result['output']}\n"
            f"--- stderr ---\n{result['error']}"
        )[:6000]
        return {"content": [{"type": "text", "text": text}]}

    return [delete_file, run_tests]


def run(
    root: Path,
    index: indexer.RepoIndex,
    requirement: Dict[str, Any],
    architecture: Dict[str, Any],
    quality: Dict[str, Any],
    review: Optional[Dict[str, Any]] = None,
    failure_analysis: Optional[Dict[str, Any]] = None,
    human_note: Optional[str] = None,
    on_event: Optional[Callable[[], None]] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    planned_write_paths = set(architecture.get("files_to_modify", []) or []) | set(
        architecture.get("files_to_create", []) or []
    )
    planned_delete_paths = set(architecture.get("files_to_delete", []) or [])

    # A correctly-diagnosed ENVIRONMENT_ERROR whose fix is a missing
    # dependency-manifest entry would otherwise have no path to actually
    # get applied: Architecture never named the manifest (it isn't part
    # of the feature's own plan), so without this, the run dead-ends at
    # FINALIZE/INCOMPLETE even though Failure Analysis found the exact
    # fix. Filtered against the fixed allowlist regardless of what
    # Failure Analysis said -- an LLM's diagnosis is trusted for WHICH
    # manifest, never for whether an arbitrary path is safe to widen
    # this agent's write scope to.
    dependency_fix_paths = {
        path
        for path in ((failure_analysis or {}).get("dependency_fix_files") or [])
        if is_dependency_manifest_path(path)
    }

    # Test-shaped paths are Quality's exclusive domain, regardless of
    # what Architecture listed -- the Coding Agent must never be able
    # to touch them, even accidentally.
    allowed_write_paths = {
        path for path in planned_write_paths if not is_test_path(path)
    } | dependency_fix_paths
    allowed_delete_paths = {
        path for path in planned_delete_paths if not is_test_path(path)
    }

    if not allowed_write_paths and not allowed_delete_paths:
        raise ValueError(
            "Coding Agent invoked with no non-test files_to_modify/"
            "files_to_create/files_to_delete."
        )

    context_sections = [
        f"""
============================================================
REQUIREMENT
============================================================

{format_requirement(requirement)}

============================================================
REPOSITORY OVERVIEW
============================================================

{indexer.repo_overview_text(index)}

============================================================
WORK ORDER (what you are being asked to do, and when you are done)
============================================================
{format_work_order(architecture)}
============================================================
ARCHITECTURE PLAN
============================================================

{format_dict(architecture)}

============================================================
TEST SUITE (already written -- do not modify)
============================================================

{format_dict(quality.get('test_files', {}))}
"""
    ]

    if review:
        context_sections.append(
            f"""
============================================================
REVIEW FINDINGS TO ADDRESS
============================================================

{format_dict(review)}
"""
        )

    if failure_analysis:
        context_sections.append(
            f"""
============================================================
FAILURE ANALYSIS (correction pass)
============================================================

{format_dict(failure_analysis)}
"""
        )

    if human_note:
        context_sections.append(format_note(human_note))

    files_to_touch_section = (
        f"""
============================================================
FILES YOU MAY WRITE (create or modify)
============================================================

{chr(10).join(f"- {path}" for path in sorted(allowed_write_paths)) or "(none)"}
"""
        if allowed_write_paths
        else ""
    )

    files_to_delete_section = (
        f"""
============================================================
FILES YOU MAY DELETE (with delete_file)
============================================================

{chr(10).join(f"- {path}" for path in sorted(allowed_delete_paths))}

Actually call delete_file on each of these -- do not just describe
deleting them.
"""
        if allowed_delete_paths
        else ""
    )

    user_prompt = (
        "\n".join(context_sections)
        + files_to_touch_section
        + files_to_delete_section
        + """

Read what you need, make your changes with Write/Edit and/or
delete_file, self-check with run_tests, then provide your final
summary.
"""
    )

    coding_tools = _make_tools(root, allowed_delete_paths)
    server = create_sdk_mcp_server("coding_tools", tools=coding_tools)
    tool_names = [f"mcp__coding_tools__{t.name}" for t in coding_tools]

    agent_result = llm.run_agent(
        agent_name="Coding Agent",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        output_schema=OUTPUT_SCHEMA,
        required_keys=["summary", "self_check_passed"],
        cwd=root,
        tools=["Read", "Write", "Edit", "Grep", "Glob", *tool_names],
        can_use_tool=make_repo_scoped_permission(root, is_write_allowed=lambda p: p in allowed_write_paths),
        mcp_servers={"coding_tools": server},
        on_event=on_event,
        model=model,
    )

    files_written: Dict[str, str] = {}
    for rel_path in written_paths_from_tool_calls(agent_result.tool_calls, root):
        full_path = root / rel_path
        if full_path.is_file():
            files_written[rel_path] = full_path.read_text(encoding="utf-8", errors="replace")

    # The tool call recorded in the stream always carries the SDK's
    # fully-qualified name (mcp__<server>__<tool>), regardless of which
    # form was granted access in `tools` above -- match on that, not the
    # bare tool name.
    files_deleted: List[str] = [
        call["input"]["path"]
        for call in agent_result.tool_calls
        if call.get("tool") == "mcp__coding_tools__delete_file"
        and call.get("input", {}).get("path") in allowed_delete_paths
        and not (root / call["input"]["path"]).exists()
    ]

    if not files_written and not files_deleted:
        # On a correction pass (review findings or a failure diagnosis
        # supplied), the agent may correctly conclude there is nothing
        # left to act on -- e.g. the findings it was asked to address
        # don't actually describe a real defect. That's a legitimate
        # outcome, not a crash: the graph's own iteration/stagnation
        # bounds are what protect against this looping forever, not an
        # exception here. On a first attempt (nothing to react to),
        # writing nothing really is a bug worth failing loudly on.
        if not review and not failure_analysis and not human_note:
            raise ValueError(
                "Coding Agent finished without writing or deleting any "
                "files on its first attempt."
            )

    result = dict(agent_result.result)
    validate_clarifying_questions(result, "Coding Agent")
    result["files_written"] = files_written
    result["files_deleted"] = files_deleted
    result["_tool_calls"] = agent_result.tool_calls
    return result
