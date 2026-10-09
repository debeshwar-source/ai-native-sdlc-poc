"""
Real git workspace management: clone a GitHub repo, branch off the
requested base branch, and later commit/push/diff against it.

This is the one module that talks to a real, possibly-remote git
repository, so it's kept small and easy to audit.
"""

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import config


def _scrub(text: str) -> str:
    if config.GITHUB_TOKEN:
        text = text.replace(config.GITHUB_TOKEN, "<redacted>")
    return text


def _run_git(args: List[str], cwd=None, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip()
        # Scrub the WHOLE message, not just `error` -- args itself can
        # contain a token-embedded URL (push/clone use one), and that
        # must never reach an exception message, a log file, or the
        # UI in plaintext.
        message = f"git {' '.join(args[:2])} ... failed: {error}"
        raise RuntimeError(_scrub(message))
    return result


def _slugify(text: str, max_length: int = 50) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug or "change")[:max_length]


def parse_github_repo(repo_url: str):
    """Return (owner, name) for a github.com URL, or (None, None)."""
    match = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(\.git)?/?$", repo_url)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def authenticated_url(repo_url: str) -> str:
    """Public: also used to build a token-authenticated fork clone URL."""
    if config.GITHUB_TOKEN and repo_url.startswith("https://github.com/"):
        return repo_url.replace(
            "https://github.com/",
            f"https://x-access-token:{config.GITHUB_TOKEN}@github.com/",
        )
    return repo_url


@dataclass
class Workspace:
    root: Path
    repo_url: str
    base_branch: str
    working_branch: str
    owner: Optional[str]
    name: Optional[str]


def _repo_cache_dir(repo_url: str) -> Path:
    owner, name = parse_github_repo(repo_url)
    key = f"{owner}__{name}" if owner and name else _slugify(repo_url, max_length=80)
    return config.REPO_CACHE_ROOT / key


def _clone_with_fallback(repo_url: str, dest: Path, extra_args: List[str]) -> None:
    """Shared by the cache's first-time clone: try the plain URL
    first (ambient credentials may already grant read access), only
    falling back to embedding GITHUB_TOKEN if that actually fails."""
    plain_result = _run_git(["clone", *extra_args, repo_url, str(dest)], check=False)
    if plain_result.returncode == 0:
        return

    if not config.GITHUB_TOKEN:
        error = _scrub(plain_result.stderr.strip() or plain_result.stdout.strip())
        raise RuntimeError(f"git clone failed: {error}")

    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)

    token_result = _run_git(
        ["clone", *extra_args, authenticated_url(repo_url), str(dest)], check=False
    )
    if token_result.returncode != 0:
        plain_error = _scrub(plain_result.stderr.strip() or plain_result.stdout.strip())
        token_error = _scrub(token_result.stderr.strip() or token_result.stdout.strip())
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        raise RuntimeError(
            f"git clone failed with ambient credentials ({plain_error}); "
            f"retry with GITHUB_TOKEN also failed: {token_error}"
        )


def _ensure_repo_cache(repo_url: str, base_branch: str) -> Path:
    """A persistent local bare clone of repo_url, reused and updated
    incrementally across runs instead of re-downloaded from GitHub
    every time -- the actual fix for "every run treats the repo as
    never seen before". A run's own workspace still clones fresh
    from this (fast, local, no network) so runs stay fully
    independent; only this shared cache persists between them."""
    config.REPO_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache = _repo_cache_dir(repo_url)

    if not cache.exists():
        _clone_with_fallback(repo_url, cache, ["--bare"])
        return cache

    # Already cached: fetch just the branch we need, updating it in
    # place -- far cheaper than a full clone. A corrupt/interrupted
    # cache from an earlier run falls back to rebuilding it once
    # rather than failing this run.
    fetch_args = ["fetch", "origin", f"+{base_branch}:{base_branch}"]
    fetch_result = _run_git(fetch_args, cwd=cache, check=False)
    if fetch_result.returncode != 0 and config.GITHUB_TOKEN:
        fetch_result = _run_git(
            ["fetch", authenticated_url(repo_url), f"+{base_branch}:{base_branch}"],
            cwd=cache,
            check=False,
        )
    if fetch_result.returncode != 0:
        shutil.rmtree(cache, ignore_errors=True)
        _clone_with_fallback(repo_url, cache, ["--bare"])

    return cache


def clone_and_branch(repo_url: str, base_branch: str, run_id: str) -> Workspace:
    """Clone repo_url at base_branch and create a fresh working branch.
    Clones from a local, incrementally-updated cache of the repo
    rather than from GitHub directly, so a repo already used in an
    earlier run sets up fast instead of downloading it all again."""

    config.WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    root = config.WORKSPACE_ROOT / run_id

    if root.exists():
        raise RuntimeError(f"Workspace already exists: {root}")

    owner, name = parse_github_repo(repo_url)

    cache = _ensure_repo_cache(repo_url, base_branch)

    # Cloning between two local paths lets git hardlink (or, if that
    # isn't possible, fast-copy) the object store instead of
    # transferring anything over the network.
    clone_result = _run_git(
        ["clone", "--branch", base_branch, "--single-branch", str(cache), str(root)],
        check=False,
    )
    if clone_result.returncode != 0:
        error = _scrub(clone_result.stderr.strip() or clone_result.stdout.strip())
        raise RuntimeError(f"git clone from local cache failed: {error}")

    # The clone's origin now points at the local cache path, not
    # GitHub -- nothing here currently relies on that remote (push
    # always targets repo_url explicitly), but pointing it at the
    # real repo avoids surprising any future code that assumes origin
    # is meaningful.
    _run_git(["remote", "set-url", "origin", repo_url], cwd=root)

    # Run-scoped identity so commits are attributable without touching
    # the operator's global git config.
    _run_git(["config", "user.name", "Claude-Native SDLC"], cwd=root)
    _run_git(["config", "user.email", "claude-native-sdlc@local"], cwd=root)

    # Local-only exclusions (never written to the repo's own
    # .gitignore, so they never show up as an unwanted diff) for
    # directories this tool creates -- the isolated venv
    # ensure_python_environment() sets up, and the separate venv
    # tools/static_analysis.py installs Semgrep/pip-audit into --
    # neither of which must ever be staged into a commit.
    exclude_path = root / ".git" / "info" / "exclude"
    exclude_path.parent.mkdir(parents=True, exist_ok=True)
    with exclude_path.open("a", encoding="utf-8") as handle:
        handle.write("\n.ai_sdlc_venv/\n.ai_sdlc_analysis_venv/\n")

    working_branch = f"ai-sdlc/{_slugify(run_id)}"
    _run_git(["checkout", "-b", working_branch], cwd=root)

    return Workspace(
        root=root,
        repo_url=repo_url,
        base_branch=base_branch,
        working_branch=working_branch,
        owner=owner,
        name=name,
    )


BUILD_ARTIFACT_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache"}


def _clean_build_artifacts(root: Path) -> None:
    """
    Remove generated artifact directories (e.g. __pycache__ created by
    running tests) before they can be picked up by git status/add --
    they're never a real part of the change and must never leak into
    a commit or PR diff.
    """
    dirs_to_remove = [
        path
        for path in root.rglob("*")
        if path.is_dir() and path.name in BUILD_ARTIFACT_DIR_NAMES
    ]
    for path in dirs_to_remove:
        shutil.rmtree(path, ignore_errors=True)


def status_porcelain(workspace: Workspace) -> str:
    _clean_build_artifacts(workspace.root)
    return _run_git(["status", "--porcelain"], cwd=workspace.root).stdout


def reset_to_base(workspace: Workspace) -> None:
    """
    Discard any uncommitted working-tree changes, back to the base
    branch's state. Used when the Architecture Agent is re-run after
    a Coding Agent attempt already wrote files -- architecture should
    always reason about the real, unmodified repo, not a partial
    attempt left over from a diagnosis that turned out to implicate
    the architecture decision itself.
    """
    _run_git(["checkout", "--", "."], cwd=workspace.root, check=False)
    _run_git(["clean", "-fd"], cwd=workspace.root, check=False)


def changed_files(workspace: Workspace) -> List[str]:
    files = []
    for line in status_porcelain(workspace).splitlines():
        path = line[3:].strip()
        if "->" in path:
            path = path.split("->")[-1].strip()
        if path:
            files.append(path)
    return files


def _mark_new_files_for_diff(workspace: Workspace, path: Optional[str] = None) -> None:
    """`git diff` never shows untracked files, even with a pathspec --
    a brand-new file the Coding Agent just wrote would otherwise
    silently produce an empty diff. `git add --intent-to-add` records
    just the path's presence in the index (no content staged), which
    is enough for `git diff` to show it as a pure addition -- and
    doesn't change what commit_all stages later, since that restages
    everything anyway."""
    args = ["add", "--intent-to-add"]
    args += ["--", path] if path else ["-A"]
    _run_git(args, cwd=workspace.root, check=False)


def diff_against_base(workspace: Workspace, path: Optional[str] = None) -> str:
    _mark_new_files_for_diff(workspace, path)
    args = ["diff", workspace.base_branch, "--"]
    if path:
        args.append(path)
    result = _run_git(args, cwd=workspace.root, check=False)
    return result.stdout


def diff_stat_against_base(workspace: Workspace) -> str:
    _mark_new_files_for_diff(workspace)
    result = _run_git(
        ["diff", "--stat", workspace.base_branch], cwd=workspace.root, check=False
    )
    return result.stdout


def commit_all(workspace: Workspace, message: str) -> Optional[str]:
    """Stage and commit everything. Returns the commit SHA, or None if
    there was nothing to commit."""

    _clean_build_artifacts(workspace.root)
    _run_git(["add", "-A"], cwd=workspace.root)

    nothing_staged = (
        _run_git(["diff", "--cached", "--quiet"], cwd=workspace.root, check=False)
        .returncode
        == 0
    )
    if nothing_staged:
        return None

    _run_git(["commit", "-m", message], cwd=workspace.root)
    return _run_git(["rev-parse", "HEAD"], cwd=workspace.root).stdout.strip()


def push_to_url(workspace: Workspace, remote_url: str) -> dict:
    """Push the working branch (same name) to an arbitrary remote --
    the original repo_url, or a fork's clone URL."""
    try:
        _run_git(
            ["push", remote_url, f"HEAD:refs/heads/{workspace.working_branch}"],
            cwd=workspace.root,
        )
    except RuntimeError as e:
        return {"status": "FAIL", "error": str(e), "branch": workspace.working_branch}

    return {"status": "PASS", "branch": workspace.working_branch}


def push(workspace: Workspace) -> dict:
    if not config.GITHUB_TOKEN:
        return {
            "status": "SKIPPED",
            "reason": (
                "No GITHUB_TOKEN configured; branch and commit(s) exist "
                "locally only."
            ),
            "branch": workspace.working_branch,
        }

    return push_to_url(workspace, authenticated_url(workspace.repo_url))


def cleanup(workspace: Workspace) -> None:
    shutil.rmtree(workspace.root, ignore_errors=True)
