# Claude-Native SDLC

The same stages as `../ai_native_sdlc/` -- Requirement, Architecture,
Quality/TDD, Coding, Testing, Failure Analysis, Code Review, QA Review, Release &
Operations -- against a real, cloned repository, but driven by **Claude
Code itself** (via the Claude Agent SDK) instead of Amazon Bedrock, and
**dynamically routed** rather than run in a fixed sequence (see "Dynamic
agent routing" below). No `ANTHROPIC_API_KEY` or AWS credentials
required: it authenticates exactly like an interactive Claude Code
session.

## Why this exists, and how it differs from `ai_native_sdlc`

`ai_native_sdlc` calls Bedrock's Converse API directly, hand-rolling an
agentic tool-use loop (`read_file`/`write_file`/`delete_file`/`list_files`/
`search_code`/`run_tests` as Bedrock tool definitions) and a JSON-repair
loop for structured output.

This system instead drives the Claude Agent SDK's `query()` -- the same
engine that runs this Claude Code CLI -- which:

- **Runs the agentic loop itself.** Exploration, edits, and self-checking
  happen inside one `query()` call; there's no hand-rolled tool-call loop
  to maintain.
- **Uses Claude Code's own built-in tools** (Read, Write, Edit, Grep,
  Glob) instead of custom Bedrock tool definitions, plus two small custom
  tools for the Coding Agent (`delete_file`, `run_tests`) where a built-in
  tool would be too broad (there's no built-in "delete one file" tool,
  and general `Bash` access was deliberately not granted -- same
  reasoning as the original build: explicit, auditable operations, not a
  shell).
- **Gets structured JSON "for free"** via `output_format` (schema-
  validated by the CLI itself), rather than a custom `finish` tool +
  JSON-repair loop.
- **Scopes file access with a `can_use_tool` permission callback**
  instead of pre-filtering which paths a tool is even allowed to see --
  every Write/Edit call is checked against the exact paths Architecture
  named (or, for Quality, against the test-path pattern) before it's
  allowed to execute, and every path-taking tool call (read or write) is
  confined to inside the cloned repo.

Everything that isn't about talking to the model -- `workspace.py` (git
operations), `context/indexer.py` (repo indexing), `tools/test_runner.py`
(language-aware test execution), `tools/github_api.py` (real PR
creation), `tools/issue_tracker.py` (local-mock Jira/Confluence),
`feedback/store.py` (cross-run feedback), the LangGraph wiring in
`orchestrator/graph.py`, and the Streamlit UI -- is the same code as
`ai_native_sdlc`, including its human-in-the-loop "suggest changes"
revision cycle and per-file coding diffs. The two systems are otherwise
completely independent; this one doesn't import from the other.

## Prerequisite: be logged into Claude Code

```bash
claude login
```

(If you're reading this from inside a Claude Code session, you're
already logged in -- nothing further to do.) The bundled CLI the SDK
drives reuses that same session; there's no separate API key to
configure. `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` also work if set,
but aren't required.

## Usage

```bash
cd claude_native_sdlc
pip install -r requirements.txt

python3 cli.py \
  --repo https://github.com/OWNER/REPO.git \
  --branch main \
  --spec "Add a GET /health endpoint that returns {status: ok}"

# or from a file
python3 cli.py --repo ... --branch main --spec @spec.txt

# UI, with human-in-the-loop suggest-changes/approve/reject gates
streamlit run streamlit_app.py
```

Set `GITHUB_TOKEN` (or `GH_TOKEN`) in the environment to allow pushing
the working branch and opening a real pull request -- this is a GitHub
credential, unrelated to the Claude auth above.

Set `CLAUDE_SDLC_MODEL_ID` to use a different Claude model (default
`claude-sonnet-5`).

## Design layer (existing vs proposed architecture)

The Architecture stage also produces a design: a component-level picture
of the repo as it is today and as it would be after the change, shown as
two tabs ("Existing architecture" / "Proposed architecture") on the
Architecture stage in the UI, each with a plain-language overview and a
Graphviz diagram. (`design.py`)

The model proposes the components (modules / services / layers -- not
files) and how they depend on each other, from what it actually read.
Everything checkable is then checked in code:

- an existing component must cite real files in the repo, or it is
  dropped -- no invented components;
- each component's status (unchanged / modified / new / removed) is
  *computed* from Architecture's own `files_to_modify` / `files_to_create`
  / `files_to_delete`, never taken from the model, so the proposed
  picture can't disagree with the plan Coding will execute;
- any planned file the model didn't place in a component is added under a
  catch-all, so every planned change is visible;
- relationships to unknown components are dropped.

If the model gives nothing usable, a deterministic fallback groups the
repo by top-level directory, so there is always a picture. The design is
advisory and never fails a run. It's kept in state as `design`, separate
from `architecture`, so it doesn't add to every later agent's prompt.

This replaces the old post-clone "Repository structure" file-tree graph,
which was removed; the plain-text file list is still available there.

## Two reviewers: Code Review and QA Review

Release is gated by two independent reviewers, run in order, each with
a different question:

- **Code Review** (`agents/review.py`, node `review`) -- is the code
  correct, safe and in scope? Includes the deterministic Semgrep /
  pip-audit / declared-file gates described below.
- **QA Review** (`agents/qa_review.py`, node `qa_review`) -- does the
  *evidence* prove the requirement is met? Every acceptance criterion
  is given an id (C1..Cn) and assessed COVERED / PARTIAL / UNCOVERED
  against the tests that actually ran. It also checks the Architecture
  work order's `definition_of_done` and flags weak or vacuous tests and
  missing negative/edge cases. It does not repeat Code Review's style/
  security/scope checks.

QA has its own deterministic gate, same pattern as Code Review's: a
criterion marked UNCOVERED -- or never assessed at all -- becomes a
BLOCKING finding regardless of the model's own status, so a model can't
pass a coverage gap by omission. PARTIAL (and "COVERED" with no test
cited) is a FOLLOW_UP.

Routing: a passing Code Review hands off to QA Review (its release
request is redirected there by `resolve_next_node`); QA then routes like
Review does -- QUALITY for a missing/weak test, CODING for wrong
behavior, ARCHITECTURE if the plan can't satisfy the requirement.
`release_planning` is unreachable until **both** have PASS. A new Code
Review or new code resets QA's verdict, so QA always judges the
current diff. A run stopped by QA ends as `QA_BLOCKED`. Each reviewer's
model is independently choosable (`--agent-model qa_review=...`).

## Dynamic agent routing

This used to be a fixed LangGraph pipeline: quality always went to
coding-or-verification, testing always went to review-or-retry, and so
on, each transition decided by a bespoke router function reading the
previous stage's output. It no longer is. Every dynamically-routed
agent (Requirement, Architecture, Quality, Coding, Testing, Failure
Analysis, Code Review, QA Review) is now made aware of every OTHER agent in the system
-- what it does, when it's normally the right next step -- via a
shared `AGENT_DIRECTORY_TEXT` block in `agents/_common.py`, included in
every one of their system prompts. Each agent's own structured output
now includes `next_agent` (which of the nine stages, including
FINALIZE, should run next) and `next_agent_reason` (why, in one
sentence) -- its own judgment call, not a fixed edge.

`orchestrator/graph.py`'s `resolve_next_node()` is the single router
every one of those seven nodes shares. It reads that choice and applies
a handful of hard invariants no agent's choice can override:

- `RELEASE_PLANNING` (and therefore `release_apply`) is unreachable
  without a passing test run AND a passing review, regardless of which
  agent named it or why -- redirected to whichever of the two is still
  outstanding.
- `CODING` is unreachable with nothing for it to act on (no files
  named by Architecture, no review/failure-analysis feedback to react
  to) -- redirected to `TESTING` instead.
- The existing per-run iteration cap (`MAX_ITERATIONS`) and stagnation
  cap (`MAX_STAGNANT_ITERATIONS`, repeating the identical test failure)
  still force a stop, same as before, once the run is past its first
  pass through.
- A new, independent `MAX_TOTAL_STEPS` (40) hard-caps the total number
  of agent hops in a single run. The caps above are specific to the
  coding<->testing<->failure_analysis retry loop; dynamic routing means
  e.g. Review and Architecture could in principle oscillate without
  ever touching Coding at all, so this is the general backstop that
  guarantees the run terminates regardless of which agents get visited
  in what order.

`release_planning -> release_apply -> finalize` stays a fixed sequence
on purpose: applying an already-approved release plan (commit, push,
open the PR) is mechanical execution, not a judgment call worth
letting an agent route around.

Both `cli.py` (via the compiled LangGraph state machine) and
`streamlit_app.py` (which drives nodes one at a time itself, for the
human-in-the-loop UI) share the exact same `resolve_next_node` --
there's one routing implementation, not two that could drift apart.

## Dependency install & test execution

`tools/test_runner.py` installs the TARGET repository's own declared
dependencies (into an isolated venv at `.ai_sdlc_venv/`, never the
host's own site-packages) and runs its test suite, both via plain host
subprocesses -- no Docker or other container runtime involved, and
none required. The CLI and Streamlit UI both ask before installing
(`--install-deps` / the "Install this repo's dependencies" checkbox)
rather than doing it automatically, since it's a real action (network,
disk, time -- can take minutes on a dependency-heavy repo).

## Claude logs in the UI

Every agent call goes through `llm.py`'s `run_agent()`, which now
captures the full underlying Claude Agent SDK transcript -- Claude's
own narration text, each tool call, and each tool's actual result, in
the order they really happened -- not just the final structured
result. Each agent exposes it as `_log` in its own returned dict (the
SDK's internal `StructuredOutput` bookkeeping call is filtered out;
it's not a real tool call, and the result it "returns" is already
shown separately).

The Streamlit UI renders this per stage via `render_claude_log()`, in
the same "Exploration"/"Plan"/"Diagnosis" expanders that already held
a bare list of tool names -- so expanding one now shows what a live
Claude Code session itself would: the file it read and what was in
it, the edit it made and the confirmation, the self-check test run
and its actual pytest output, interleaved with any reasoning text,
not just "Read(file_path=...)" with no result.

**It streams live, not just after the stage finishes.** `run_agent()`
takes an `on_event` callback, invoked the instant each entry happens
inside the SDK's own event loop, not batched until the call returns.
`orchestrator/graph.py`'s `SDLCState` carries an optional `on_event(stage,
entry)` through to whichever agent a node calls (`_stage_event_callback`
binds it to that node's own stage name); `streamlit_app.py`'s live
execution loop sets it to a closure that re-renders an `st.empty()`
placeholder on every single entry. Streamlit pushes a UI delta to the
browser the moment a widget call happens, even mid-script inside a
blocking call -- confirmed by polling a real run every 1.5s and
watching the visible tool-call count grow one at a time (2, 3, 4, 5...)
while the agent was still working, not jump from 0 to N once it
returned. The placeholder is cleared once the stage completes, since
`render_stage_body` then shows the same transcript again inside its
own (collapsed) expander -- the live view and the finished view are
deliberately different, not doubled up.

## Works with any repo, not just Python/Node/Go/Rust

`tools/test_runner.py`'s install/test detection is hardcoded for
Python, Node, Go, and Rust -- fast and free, but a blind spot for
everything else. `environment_setup.py` closes it: when none of those
four match at all, it calls `agents/environment.py`'s **Environment
Agent**, which explores the repo itself (Read/Grep/Glob -- its own
manifest/build file, README, CI config) and decides how to install and
test it, the same way Claude Code has no language whitelist and just
figures out what a given repo needs.

`tools/test_runner.py` itself stays exactly what its docstring says
("deterministic, no LLM calls") -- it never imports the agent.
`environment_setup.py` is the one place that decides when to escalate,
and hands the result back through `test_runner.py`'s own cache/
execution primitives (`set_fallback_test_command`, `run_command`).

Safety boundary, since this reintroduces a form of "run a command an
LLM decided on" that the Docker sandbox used to contain: the agent can
only pick a command whose first token is a real, known build/test tool
(`agents/environment.py`'s `ALLOWED_EXECUTABLES` -- `bundle`, `mvn`,
`gradlew`, `composer`, `dotnet`, `make`, etc.), never an arbitrary
executable. `_sanitize_command` enforces this on every returned
command before anything executes; a suggestion outside the allowlist
comes back empty (treated as "no command determined"), not narrowed or
partially run. Verified against an adversarial repo whose README tried
to instruct the agent to `curl | bash` an external script -- the model
itself refused it as prompt injection, and separately, the allowlist
was confirmed (by direct unit test) to reject `curl`/`rm`/`sudo`/
`bash -c` regardless of what any model decides.

This still trusts the repo's own declared build process the same way
`pip install -r requirements.txt` already does -- an allowed tool can
still be pointed at a malicious manifest. The allowlist's job is
narrower: make sure the agent can never be talked into running
something that isn't the repo's own build/test tooling at all.

## Deterministic static/security analysis in Review

The Review Agent's read of the diff is an LLM judgment call; alongside
it, `tools/static_analysis.py` runs two deterministic, no-LLM checks
on every review pass, in their own dedicated venv (`.ai_sdlc_analysis_venv/`,
also plain host subprocesses):

- **Semgrep**, scoped to the changed files -- our own small
  hardcoded-secret ruleset (`tools/semgrep_rules/secrets.yml`: AWS/
  GitHub/Slack/Stripe token formats, private-key headers), plus
  Semgrep's general `p/security-audit` + `p/owasp-top-ten` registry
  rulesets.
- **pip-audit**, checking the repo's own `requirements*.txt`/
  `pyproject.toml` against the PyPI Advisory DB for known-vulnerable
  dependency versions -- but ONLY when this change's own diff touched
  a dependency manifest. Unlike Semgrep, pip-audit has no way to scope
  itself to just the diff; it audits the WHOLE manifest. Gating on
  that unconditionally means any repo with pre-existing pinned
  vulnerabilities (extremely common -- e.g. a repo pinning an older
  `torch`) can never pass review at all, for a problem this change
  didn't cause and the Coding Agent has no path-scoped permission to
  fix anyway. So this only runs, and only gates, when the change
  itself added/touched a dependency.

General security-audit/OWASP findings are handed to the Review Agent
as context -- that ruleset has a real false-positive rate, so it's the
model's judgment call whether to raise one. Hardcoded secrets and
known CVEs are different: both have a near-zero false-positive rate,
so they're appended to the Review Agent's own findings as BLOCKING
*regardless of what the model concluded* -- the same orchestrator-
level-invariant pattern already used to force NO_GO on a failed test
run. Since `resolve_next_node` only routes to `release_planning` once
both testing and review have passed, this is what actually gates
`release_apply`: a real secret or known CVE blocks release even if
the LLM review missed or dismissed it.

## Tree-sitter repo map

`context/indexer.py`'s "repository overview" used to be just a file
tree + manifest contents -- useful, but it meant Architecture/Quality/
Coding/Review/Failure Analysis all rediscovered the codebase's actual
structure from scratch via live Read/Grep/Glob calls on every single
run, which burns turns on a large repo.

It now also builds a repo map, adapting the technique from Aider's
`repomap.py`: tree-sitter-parse every file in a language we have a
grammar for (Python, JavaScript, TypeScript, Go, Rust -- each
package's own bundled `TAGS_QUERY`, the same tag-query convention
GitHub code navigation and Aider itself use, no hand-written queries),
build a weighted graph of which files reference definitions in which
other files, and rank both files and their definitions with a plain
PageRank. The result is handed to every agent that receives a
`RepoIndex` (previously just Architecture and Quality; now Coding,
Review, and Failure Analysis too, which is what actually closes the
gap -- they took the parameter before but never used it) as a dense
"here's what actually matters here" summary, ranked by how much of the
codebase actually calls/uses each thing -- most useful on a real,
large repo, where it consistently surfaces the handful of genuinely
central files (a logging setup, a base repository class, a shared
model) at the top.

Deterministic, no LLM calls, and never fatal: the tree-sitter packages
are an optional import (`pip install -r requirements.txt` pulls them
in) -- if they're missing, or a repo's language isn't one of the five
above, `repo_map_text` is just empty and the overview falls back to
exactly what it was before this existed.

## Layout

```
config.py              runtime configuration (model id, paths, iteration bounds -- no AWS/API-key config)
llm.py                 Claude Agent SDK wrapper: run_agent(), the single entry point every agent calls
workspace.py            real git operations: clone, branch, commit, push, diff (same as ai_native_sdlc)
cli.py                   entrypoint
environment_setup.py       ties test_runner.py's hardcoded detection to the Environment Agent fallback (see above)

context/indexer.py        repo indexing + tree-sitter repo map (see above)
tools/test_runner.py        language-aware test execution (pytest/npm/go/cargo) (see above)
tools/static_analysis.py       Semgrep + pip-audit, feeding/gating Review (see above)
tools/github_api.py           real PR creation via the GitHub REST API
tools/issue_tracker.py          Jira/Confluence-shaped output; local-file mock

agents/environment.py    agentic, Read/Grep/Glob only -- fallback ecosystem detection for any repo (see above)
agents/requirement.py    single structured call, no tools
agents/architecture.py    agentic, Read/Grep/Glob only
agents/quality.py          agentic, Read + Write/Edit (test-shaped paths only, via can_use_tool)
agents/coding.py            agentic, Read/Write/Edit (architecture's named paths only) + custom delete_file/run_tests tools
agents/failure_analysis.py   agentic, Read/Grep/Glob only -- diagnoses which artifact is responsible for a failure
agents/review.py              agentic, Read/Grep/Glob + a precomputed diff handed to it directly
agents/testing.py               deterministic test run + one-shot LLM scorecard/Go-No-Go
agents/release.py                one-shot, no tools -- release plan + Jira/Confluence summaries
agents/_common.py                 shared prompt/schema helpers, the can_use_tool permission-scoping
                                    helpers, and the AGENT_DIRECTORY_TEXT/with_next_agent_schema
                                    dynamic-routing helpers every agent shares (see above)

orchestrator/graph.py    LangGraph wiring: 10-node pipeline, dynamically routed (see above) via one
                          shared resolve_next_node() rather than a bespoke router per node; still
                          includes the human-in-the-loop "suggest changes" revision cycle and
                          per-pass coding diffs
feedback/store.py          continuous feedback loop (JSONL, keyed by repo)
```

## Runtime artifacts (git-ignored, created on demand)

- `workspaces/<run_id>/` -- the cloned repo for each run (kept after
  the run so you can inspect it; pass `--cleanup` to delete it)
- `.repo_cache/<owner>__<name>/` -- one persistent local bare clone per
  repo, fetched incrementally and reused across runs instead of
  re-downloaded from GitHub every time
- `runs/<run_id>/` -- `history.json` (full stage-by-stage log) plus the
  mock `issue_tracker/` and `wiki/` outputs for that run
- `feedback/history.jsonl` -- the cross-run feedback log

## Workflow builder (your own agents)

The Streamlit page has a **🧩 Workflow builder** expander. The catalog starts
empty; nothing is predefined.

1. **+ Add agent** tab: give an agent a *name*, *job description* and *skills*
   (and optionally let it edit files; otherwise it is read-only). It is saved to
   the catalog (`agent_catalog.json`, so it survives restarts).
2. **Catalog** tab: lists agents by name. Drag one into the **Workspace** (or
   double-click it). Drag cards to reorder; × removes one.
3. Type **orders** into any card in the workspace; they are added to that
   agent's prompt on every run.

A run executes the workspace top to bottom. Each agent reads the repo
(Read/Grep/Glob, confined to the repo root), sees the earlier agents' output,
and hands its own to the next (`agents/custom.py`). With human-in-the-loop on,
every step pauses for approval / suggested changes / override as usual. The
built-in Requirement/Architecture/Coding/... agents are no longer offered in
the UI; they remain available via the CLI (`--stages`, `--order`).

### Knowledge base

The tiny **📚** button above the workspace opens the knowledge base: add a
folder (**Browse for a folder…** opens the OS folder picker on the machine
running the app, or paste a path) or a git repo URL. Sources persist to
`knowledge_base.json`. When a run starts, folders are used in place and repos are
shallow-cloned into `.knowledge_cache/`; every agent can then search and read them
(read-only, never written to) alongside the repo being worked on.

### The Orchestrator

The Orchestrator runs in the background and, for now, only remembers. The small
**◎ Orchestrator** button above the workspace opens its memory in place of the workspace:
the knowledge base, and a history of every order given to every agent with the output it
produced (time, duration, status, files written). The history is kept on disk
(`orchestrator_memory.json`), so it survives restarts, and clearing an agent's chat or the
workspace doesn't erase it; **Clear history** does. It has no other job yet.
(`agents/orchestrator.py` is an unused, parked draft of a future one.)

### Agents are chats

Each agent card in the workspace is a conversation, like Claude Code's: orders you send
appear as messages, the agent's steps stream in live (reasoning, each tool it uses and
the result), and its reply follows. The composer at the bottom takes the next order
(Enter sends, Shift+Enter adds a line); **■** stops a run in progress; **+** starts a
fresh conversation; the chip shows the agent's model. Follow-up orders continue the same
conversation, and agents above it in the workspace that have already answered are given
to it as context. Finished runs fold their steps into a "Worked for Ns · N steps" line you
can expand.

There is no repository: an agent works in its own scratch folder
(`agent_workdir/<agent id>/`, writable only if it was created with "can edit files") and
can read the knowledge base. Conversations live in memory in the Streamlit process (they
survive page reloads, not a server restart). Pausing mid-run isn't supported by the
underlying SDK, only stopping.

Each agent box can be resized: drag the grip in its bottom-right corner (width of the box, height of its chat area); double-click the grip to reset. Sizes are remembered while the page stays open.

### Agents ask you directly

An agent that needs a decision only you can make doesn't guess: it uses its `ask_user` tool,
stops, and the question appears in its own chat as multiple choice (options, the agent's ✨
recommendation, and **Other…** to answer in your own words). It produces no output until you
click **Send answers**; it then continues from them. The questions and your answers stay in
the chat and in the Orchestrator's history. **■** stops an agent that is waiting. You can also just type: a message sent while an agent is waiting on a question is taken as your reply in your own words.

### Wiring agents together

The workspace is a canvas of nodes. Drag an agent from the catalog and drop it anywhere; move a
node by its header. Every node has an **input dot** (left) and an **output dot** (right): drag from
one node's output dot onto another node to draw a wire. From then on, whenever you give the
downstream agent an order, the **latest output of every agent wired into it** is passed into its
context (a "Context from: …" row on the node shows which). Wires are what define the flow, not
position; an agent that hasn't answered yet contributes nothing, and nothing runs automatically
when an upstream agent finishes. Click the ✕ on a wire to disconnect; loops and duplicate wires are
refused; removing a node removes its wires. **Every agent can read the knowledge base**, wired or not.
The Orchestrator's history records which agents' output each order was given with.

### Jira & Confluence

Every agent can reach Jira and Confluence **when you ask it to**, the way you'd ask Claude. Nothing is
connected or asked up front: an agent drafts your stories first, and only when you say "now create them in Jira"
does it use its `atlassian_task` tool. At that moment:

1. If no account is connected, a small form appears in the agent's chat (a link from your site, your Atlassian
   email and an [API token](https://id.atlassian.com/manage-profile/security/api-tokens)), saying what it needs
   access for. The token is checked against Jira before it is saved (a wrong one is explained and asked again) and
   is never shown in the chat, the trace or the Orchestrator's history.
2. For anything that **changes** Jira or Confluence it asks your permission first, as a multiple-choice question
   naming what it is about to do: *Allow this once* / *Allow for the rest of this conversation* / *Don't allow*
   (or answer in your own words). Reading never needs a further prompt once you are connected.
3. A helper run, started only now with the open-source [`mcp-atlassian`](https://github.com/sooperset/mcp-atlassian)
   server (`uvx mcp-atlassian`, downloaded on first use), does the task and reports what it found or created. For
   reads the server runs in `READ_ONLY_MODE`, which removes every write tool.

In the agent's form, **Jira & Confluence** is *Ask me when needed* (default) or *Never use Jira / Confluence*.
The small **🔌** button is optional: connect ahead of time or change account. Credentials live in
`atlassian_credentials.json` (owner-only, git-ignored).

### Editing an agent

Open any agent's configuration after it has been created: click **✎** on its catalog entry (hover it) or **⚙** in its node's header. The same form as *Add agent* opens, filled in, titled *Edit agent*; change the name, job description, skills, model, Jira & Confluence access or file-editing, and **Save changes**. Changes apply from the agent's **next order**; its conversation, wires and position are kept, and an order that is already running finishes with the old settings.

### It's a chat

The composer is always usable, and a message is never held back behind the agent's current task. Idle: your message starts the agent. **Working**: your message is delivered to the running agent immediately (streaming input), exactly as when you type to Claude mid-task, so it can change course on the spot; it shows as another bubble in the same turn. **Waiting on a question**: your message is taken as the reply, in your own words (instead of picking an option). **Waiting on the Jira connection form**: your message declines the form and the agent carries on from what you said (it is never taken as the token). **■** is a separate button that stops the agent (nothing runs on its own afterwards); Enter never stops it. A message that lands in the instant a run is finishing starts the next order instead of being lost. Agents treat their job description as their role and expertise, not a gag: they answer follow-ups and off-topic questions like a capable colleague.
