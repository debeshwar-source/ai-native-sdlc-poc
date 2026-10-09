"""
CLI entrypoint for the Claude-Native SDLC engine.

Usage:
    python3 cli.py --repo https://github.com/OWNER/REPO.git \\
        --branch main --spec "Add a GET /health endpoint returning {status: ok}"

    python3 cli.py --repo ... --branch main --spec @spec.txt

    # Pause for approval after every agent-driven stage:
    python3 cli.py --repo ... --branch main --spec ... --human-in-the-loop

    # Skip/force the dependency-install prompt (default: ask):
    python3 cli.py --repo ... --branch main --spec ... --install-deps=no
    python3 cli.py --repo ... --branch main --spec ... --install-deps=yes

    # Use a different Claude model for every agent (default: claude-sonnet-5):
    python3 cli.py --repo ... --branch main --spec ... --model claude-opus-5-5

    # Override just one or two agents, keeping the rest at --model/default:
    python3 cli.py --repo ... --branch main --spec ... \\
        --agent-model coding=claude-opus-5-5 --agent-model requirement=claude-haiku-4-5-20251001

Set GITHUB_TOKEN in the environment to allow pushing the working
branch and opening a real pull request; without it, changes are
committed locally only and the run reports exactly what to push
manually.
"""

import argparse
import itertools
import sys
from pathlib import Path

import approval_ledger
import config
import context.indexer as indexer
import environment_setup
import feedback.store as feedback_store
import orchestrator.graph as graph
import run_log
import workspace as workspace_module


def _parse_agent_model_overrides(pairs: list) -> dict:
    overrides = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--agent-model expects AGENT=MODEL, got: {pair!r}")
        agent, model = (part.strip() for part in pair.split("=", 1))
        if agent not in config.AGENT_NAMES:
            raise SystemExit(
                f"--agent-model: unknown agent {agent!r} -- choices: "
                f"{', '.join(config.AGENT_NAMES)}"
            )
        if not model:
            raise SystemExit(f"--agent-model {agent}=... needs a model id.")
        overrides[agent] = model
    return overrides


def _make_cli_ticker():
    """Same whimsical "still working" pulse as the Streamlit UI's
    _status_ticker (see run_log.STATUS_WORDS), rendered as an
    in-place-overwritten console line via graph.py's on_event ping
    instead of a Streamlit placeholder."""
    counter = itertools.count()

    def _tick(stage: str = "") -> None:
        word = run_log.STATUS_WORDS[next(counter) % len(run_log.STATUS_WORDS)]
        label = f"{stage}: " if stage else ""
        sys.stdout.write(f"\r    {label}✻ {word}…                              ")
        sys.stdout.flush()

    return _tick


def _load_spec(spec_arg: str) -> str:
    if spec_arg.startswith("@"):
        return Path(spec_arg[1:]).read_text(encoding="utf-8")
    return spec_arg


def _should_install_deps(mode: str) -> bool:
    """Resolves --install-deps to a yes/no decision. 'ask' interactively
    prompts -- installing pulls the repo's OWN declared dependencies
    (pip/npm/etc.), which is exactly the kind of real, resource-using
    action (network, disk, time -- torch-sized repos can take minutes)
    that shouldn't happen without the operator having said so, rather
    than as an automatic side effect of just cloning a repo."""
    if mode == "yes":
        return True
    if mode == "no":
        return False

    if not sys.stdin.isatty():
        # Nothing to prompt in a non-interactive/scripted invocation --
        # default to installing (the prior, always-automatic behavior)
        # rather than hang waiting on input() that will never come.
        print(
            "  (non-interactive session; installing by default -- pass "
            "--install-deps=no to skip, or --install-deps=yes to silence "
            "this note)"
        )
        return True

    answer = input(
        "Install this repo's dependencies now? Tests may fail on "
        "ModuleNotFoundError otherwise. [Y/n]: "
    ).strip().lower()
    return answer not in ("n", "no")


def _print_env_result(env_result: dict) -> None:
    if not env_result:
        print("  No recognized dependency manifest found; skipping install.")
        return

    any_failures = False
    markers = {"PASS": "OK", "PARTIAL": "PARTIAL", "FAIL": "FAILED"}

    for ecosystem, result in env_result.items():
        status = result.get("status")
        print(f"  {ecosystem}: {status}")

        if ecosystem == "environment_agent":
            # No hardcoded Python/Node/Go/Rust manifest matched -- the
            # Environment Agent explored the repo itself and decided
            # what this is and how to build/test it.
            print(f"    ecosystem detected: {result.get('ecosystem')}")
            if result.get("test_command"):
                print(f"    test command:      {result['test_command']}")
            if result.get("reason"):
                print(f"    reasoning:          {result['reason']}")
            if status == "FAIL" and result.get("error"):
                any_failures = True
                print(f"    {result['error'][-500:]}")
            continue

        for step in result.get("install_steps", []):
            marker = markers.get(step["status"], step["status"])
            print(f"    - {step['target']}: {marker}")
            if step["status"] != "PASS" and step.get("error"):
                any_failures = any_failures or step["status"] == "FAIL"
                print(f"      {step['error'][-500:]}")

        if status == "FAIL" and result.get("error"):
            any_failures = True
            print(f"    {result['error'][-500:]}")

    if any_failures:
        print(
            "\n  ⚠ WARNING: one or more dependencies failed to install. "
            "The pipeline will still run, but tests may fail on "
            "ModuleNotFoundError for reasons that have nothing to do "
            "with the actual code change. Check the errors above."
        )


def _prompt_for_approval(node_name: str) -> tuple[bool, str]:
    """Blocks for a line of input. Returns (approved, reason)."""
    while True:
        print(f"\n⏸  APPROVAL REQUIRED — stage: {node_name}")
        answer = input("   [a]pprove / [r]eject > ").strip().lower()
        if answer in ("a", "approve"):
            return True, ""
        if answer in ("r", "reject"):
            reason = input("   Reason (optional, press enter to skip): ").strip()
            return False, reason
        print("   Please type 'a' to approve or 'r' to reject.")


def _print_node_update(node_name: str, node_result: dict) -> None:
    history = node_result.get("history") or []
    entry = history[-1] if history else {}
    status = entry.get("status", "")
    print(f"\n>>> {node_name:<24} {status}")

    if node_name == "requirement":
        req = node_result.get("requirement", {})
        print(f"    feature: {req.get('feature', '')}")

    elif node_name == "architecture":
        arch = node_result.get("architecture", {})
        print(f"    change_required: {arch.get('change_required')}")
        if arch.get("files_to_modify"):
            print(f"    modify: {arch['files_to_modify']}")
        if arch.get("files_to_create"):
            print(f"    create: {arch['files_to_create']}")
        if arch.get("files_to_delete"):
            print(f"    delete: {arch['files_to_delete']}")

    elif node_name == "coding":
        result = entry.get("result", {})
        print(f"    files written: {list(result.get('files_written', {}).keys())}")
        if result.get("files_deleted"):
            print(f"    files deleted: {result['files_deleted']}")

    elif node_name == "testing":
        testing = node_result.get("testing", {})
        test_result = testing.get("test_result", {})
        print(f"    command: {test_result.get('command')}")
        print(f"    result: {test_result.get('status')}")

    elif node_name == "review":
        review = node_result.get("review", {})
        findings = review.get("findings", [])
        blocking = sum(1 for f in findings if f.get("severity") == "BLOCKING")
        followup = len(findings) - blocking
        print(f"    findings: {blocking} blocking, {followup} follow-up")

    elif node_name == "qa_review":
        qa = node_result.get("qa_review", {})
        findings = qa.get("findings", [])
        blocking = sum(1 for f in findings if f.get("severity") == "BLOCKING")
        coverage = qa.get("criteria_coverage", [])
        covered = sum(1 for c in coverage if c.get("status") == "COVERED")
        print(f"    status: {qa.get('status')} -- {covered}/{len(coverage)} criteria covered, {blocking} blocking")
        for c in coverage:
            if c.get("status") != "COVERED":
                print(f"    {c['criterion_id']} {c['status']}: {c.get('criterion')}")

    elif node_name == "release_apply":
        result = entry.get("result", {})
        print(f"    commit: {result.get('commit_sha')}")
        print(f"    push: {result.get('push')}")
        if result.get("fork"):
            print(f"    fork: {result['fork']}")
        print(f"    pull_request: {result.get('pull_request')}")
        if result.get("diff_stat"):
            print("\n    --- diff ---")
            for line in result["diff_stat"].strip().splitlines():
                print(f"    {line}")
            print(f"    full diff: {result.get('diff_path')}")

    elif node_name == "finalize":
        print(f"    outcome: {node_result.get('run_outcome')}")

    next_agent = node_result.get("next_agent")
    if next_agent:
        print(f"    -> chose {next_agent}: {node_result.get('next_agent_reason', '')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Claude-Native SDLC engine")
    parser.add_argument("--repo", required=True, help="GitHub repository URL")
    parser.add_argument("--branch", required=True, help="Base branch to build from")
    parser.add_argument(
        "--spec", required=True, help="Spec text, or @path/to/file for a file"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete the cloned workspace after the run (kept by default)",
    )
    parser.add_argument(
        "--human-in-the-loop",
        action="store_true",
        help=(
            "Pause after every agent-driven stage and require explicit "
            "approval before continuing."
        ),
    )
    parser.add_argument(
        "--install-deps",
        choices=["ask", "yes", "no"],
        default="ask",
        help=(
            "Whether to install the repo's own declared dependencies "
            "before running (default: ask interactively). 'no' skips "
            "the install -- tests may then fail on ModuleNotFoundError "
            "for reasons that have nothing to do with the actual change."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL_ID",
        help=(
            f"Default Claude model for every agent (default: "
            f"{config.MODEL_ID}, or $CLAUDE_SDLC_MODEL_ID). Available: "
            f"{', '.join(config.AVAILABLE_MODELS)}."
        ),
    )
    parser.add_argument(
        "--agent-model",
        action="append",
        default=[],
        metavar="AGENT=MODEL",
        help=(
            "Override the model for one agent, e.g. "
            "--agent-model coding=claude-opus-5-5. Repeatable. Agent "
            f"names: {', '.join(config.AGENT_NAMES)}."
        ),
    )
    parser.add_argument(
        "--stages",
        metavar="A,B,C",
        help=(
            "Run only these agents, in this order, instead of the standard "
            "pipeline, e.g. --stages requirement,architecture,quality,coding,testing. "
            f"Names: {', '.join(graph.DEFAULT_PIPELINE)}."
        ),
    )
    parser.add_argument(
        "--order",
        action="append",
        default=[],
        metavar="AGENT=TEXT",
        help=(
            "Standing orders for one agent, e.g. --order 'coding=keep the "
            "change under 50 lines'. Repeatable; works with or without --stages."
        ),
    )
    args = parser.parse_args()

    custom_pipeline = [x.strip() for x in args.stages.split(",") if x.strip()] if args.stages else None
    stage_orders = {}
    for item in args.order:
        agent, sep, text = item.partition("=")
        if not sep or agent.strip() not in graph.DEFAULT_PIPELINE:
            parser.error(f"--order must be AGENT=TEXT with AGENT one of: {', '.join(graph.DEFAULT_PIPELINE)}")
        stage_orders[agent.strip()] = text.strip()
    if custom_pipeline:
        problems = graph.validate_custom_pipeline(custom_pipeline)
        if problems:
            parser.error("invalid --stages: " + "; ".join(problems))

    if args.model:
        config.MODEL_ID = args.model
    agent_models = _parse_agent_model_overrides(args.agent_model)

    spec_text = _load_spec(args.spec)
    run_id = graph.new_run_id()

    print("=" * 70)
    print("CLAUDE-NATIVE SDLC")
    print("=" * 70)
    print(f"Repo:   {args.repo}")
    print(f"Branch: {args.branch}")
    print(f"Run ID: {run_id}")
    if args.human_in_the_loop:
        print("Mode:   human-in-the-loop (approval required after each stage)")
    print(f"Model:  {config.MODEL_ID} (default for every agent)")
    if agent_models:
        for agent, model in agent_models.items():
            print(f"        {agent}: {model}")

    print("\nCloning repository ...")
    try:
        ws = workspace_module.clone_and_branch(args.repo, args.branch, run_id)
    except RuntimeError as e:
        print(f"FAILED to clone: {e}", file=sys.stderr)
        return 1

    print(f"Workspace:      {ws.root}")
    print(f"Working branch: {ws.working_branch}")

    print("\nIndexing repository ...")
    index = indexer.build_repo_index(ws.root)
    print(f"Primary language: {index.primary_language or 'unknown'}")
    print(f"Files indexed:     {len(index.file_list)}")

    if _should_install_deps(args.install_deps):
        print("\nInstalling dependencies (this can take a while on first run) ...")
        env_result = environment_setup.ensure_environment(
            ws.root, index, model=agent_models.get("environment")
        )
        _print_env_result(env_result)
    else:
        print(
            "\nSkipped dependency install (--install-deps=no) -- tests may "
            "fail on ModuleNotFoundError for reasons unrelated to the "
            "actual change."
        )

    pipeline = graph.build_custom_graph(custom_pipeline) if custom_pipeline else graph.build_graph()

    initial_state = {
        "run_id": run_id,
        "spec_text": spec_text,
        "workspace": ws,
        "repo_index": index,
        "iteration": 0,
        "failure_history": [],
        "stagnation_count": 0,
        "total_steps": 0,
        "history": [],
        "on_event": _make_cli_ticker(),
        "agent_models": agent_models,
        "stage_orders": stage_orders,
    }
    if custom_pipeline:
        initial_state["custom_pipeline"] = custom_pipeline

    print("\n" + "=" * 70)
    print("RUNNING PIPELINE")
    print("=" * 70)

    final_state: dict = {}
    try:
        for event in pipeline.stream(initial_state, stream_mode="updates"):
            node_name = next(iter(event))
            node_result = event[node_name]
            final_state.update(node_result)
            _print_node_update(node_name, node_result)

            if args.human_in_the_loop and node_name in graph.GATED_NODE_NAMES:
                approved, reason = _prompt_for_approval(node_name)
                approval_ledger.record(
                    run_id=run_id,
                    repo_key=feedback_store.repo_key(ws.owner, ws.name),
                    stage=node_name,
                    decision="APPROVE" if approved else "REJECT",
                    reason=reason,
                )
                if not approved:
                    print(f"\n✗ Rejected at stage '{node_name}'. Stopping.")
                    run_log.record_human_rejection(
                        run_id, ws, spec_text, final_state, node_name, reason
                    )
                    run_dir = run_log.dump_run_state(
                        run_id,
                        final_state,
                        ws,
                        error=f"Rejected by human reviewer at stage: {node_name}",
                    )
                    print(f"Run log: {run_dir / 'history.json'}")
                    return 1
    except Exception as e:
        print(f"\nPIPELINE ERROR: {e}", file=sys.stderr)
        run_dir = run_log.dump_run_state(run_id, final_state, ws, error=str(e))
        print(f"Partial state dumped to: {run_dir}", file=sys.stderr)
        return 1

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(f"Outcome:    {final_state.get('run_outcome')}")
    print(f"Iterations: {final_state.get('iteration', 0)}")
    print(f"Workspace:  {ws.root}")

    print("\n" + "=" * 70)
    print("ROUTING PATH (which agent chose to consult which)")
    print("=" * 70)
    sequence = run_log.stage_sequence_from_history(final_state.get("history", []))
    print(run_log.format_routing_text(sequence))

    diff_text = final_state.get("release_result", {}).get("diff")
    if diff_text:
        print("\n" + "=" * 70)
        print("DIFF (like a PR's 'Files changed' tab)")
        print("=" * 70)
        print(diff_text)

    run_dir = run_log.dump_run_state(run_id, final_state, ws)
    print(f"Run log:    {run_dir / 'history.json'}")

    if args.cleanup:
        workspace_module.cleanup(ws)
        print("Workspace cleaned up.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
