"""
Environment Agent.

Fallback for any repo whose ecosystem tools/test_runner.py doesn't
already have a hardcoded install/test recipe for (Python, Node, Go,
Rust): explores the repo read-only and decides how to install its
dependencies and run its test suite, the same way Claude Code itself
has no language whitelist -- it just figures out what a given repo
needs by looking at it.

Safety boundary: the agent chooses WHICH tool (from a fixed allowlist
of known build/package executables, see ALLOWED_EXECUTABLES below)
and what arguments, but can never name an arbitrary executable -- no
shell, no curl/wget, no rm, no sudo. `_sanitize_command` enforces this
on the agent's raw output before environment_setup.py ever executes
anything it returns; a command naming a disallowed executable comes
back as an empty list (treated as "couldn't determine one"), never
silently narrowed or partially executed.

This still trusts the repo's OWN declared install/test process the
same way tools/test_runner.py's existing `pip install -r
requirements.txt` already does -- an allowed tool can still be pointed
at a malicious manifest (a poisoned Gemfile, a hostile Makefile
target). The allowlist's job is narrower and specific: prevent the
agent from ever being talked into running something that ISN'T the
repo's own build/test tooling at all.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import context.indexer as indexer
import llm
from agents._common import make_repo_scoped_permission

# Every build/package/test tool this system knows to be exactly that
# and nothing else -- adding a new ecosystem means adding its tool
# name(s) here, not writing new command-detection code.
ALLOWED_EXECUTABLES = {
    "pip", "pip3", "python", "python3",
    "npm", "yarn", "pnpm", "node",
    "go",
    "cargo",
    "mvn", "mvnw",
    "gradle", "gradlew",
    "bundle", "gem", "rake", "ruby", "rspec",
    "composer", "php", "phpunit",
    "dotnet",
    "make",
    "swift",
    "mix", "elixir",
    "sbt", "scala",
    "stack", "cabal", "ghc",
}

MAX_COMMAND_LENGTH = 12

SYSTEM_PROMPT = """
You are the Environment Agent in a Claude-Native SDLC system.

This repository's language/ecosystem doesn't match any of the ones
this system already knows how to handle out of the box (Python,
JavaScript/TypeScript/Node, Go, Rust). Your job is to figure out, by
actually exploring the real repository, how to (1) install its
declared dependencies and (2) run its test suite.

Use Read/Grep/Glob to find the real manifest/build file (pom.xml,
build.gradle[.kts], Gemfile, composer.json, *.csproj/*.sln, Makefile,
mix.exs, build.sbt, Package.swift, etc.) and any documented build/
test instructions (README, CONTRIBUTING, CI config like
.github/workflows/*.yml). Never guess a command that doesn't match
something you actually found in the repo.

Return install_command and test_command as argv lists (e.g.
["bundle", "install"], never "bundle install" as one string), using
ONLY the repo's own real, documented tool -- the exact command you'd
find in its own README or CI config, nothing invented. If a wrapper
script exists (mvnw, gradlew), prefer it over a bare system-wide
install of the same tool. If you cannot determine a safe, real way to
install or test this repo, return an empty list for that command and
explain why in `notes` -- do not fabricate a plausible-looking
command just to fill the field.
"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "ecosystem": {
            "type": "string",
            "description": "short label, e.g. 'Java (Maven)', 'Ruby (Bundler)', 'Unknown'",
        },
        "install_command": {"type": "array", "items": {"type": "string"}},
        "test_command": {"type": "array", "items": {"type": "string"}},
        "notes": {
            "type": "string",
            "description": "what you found and why, or why no safe command could be determined",
        },
    },
    "required": ["ecosystem", "install_command", "test_command", "notes"],
}


def _sanitize_command(command: Optional[List[Any]]) -> List[str]:
    """The one enforcement point between the agent's raw output and
    anything ever actually executing: reject (return empty) unless
    the first token names a real, known build/test tool -- allowing a
    bare "gradlew"/"mvnw" or a "./gradlew"/"./mvnw" wrapper-script
    form equally, since both are the same real, repo-committed
    script."""
    if not command or not isinstance(command, list):
        return []
    if len(command) > MAX_COMMAND_LENGTH:
        return []

    executable = str(command[0])
    bare = executable[2:] if executable.startswith("./") else executable
    if bare not in ALLOWED_EXECUTABLES:
        return []

    return [str(part) for part in command]


def run(
    root: Path,
    index: "indexer.RepoIndex",
    on_event: Optional[Callable[[], None]] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    user_prompt = f"""
============================================================
REPOSITORY OVERVIEW
============================================================

{indexer.repo_overview_text(index)}

Explore whatever manifest/build/CI files you need, then provide your
final install_command/test_command decision.
"""

    agent_result = llm.run_agent(
        agent_name="Environment Agent",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        output_schema=OUTPUT_SCHEMA,
        required_keys=["ecosystem", "install_command", "test_command", "notes"],
        cwd=root,
        tools=["Read", "Grep", "Glob"],
        can_use_tool=make_repo_scoped_permission(root),
        on_event=on_event,
        model=model,
    )

    result = agent_result.result
    result["install_command"] = _sanitize_command(result.get("install_command"))
    result["test_command"] = _sanitize_command(result.get("test_command"))
    result["_tool_calls"] = agent_result.tool_calls
    return result
