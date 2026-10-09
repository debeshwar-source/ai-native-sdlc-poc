"""
Real GitHub REST API calls: PR creation, plus a fork-based fallback
for when GITHUB_TOKEN only has read access to the target repo.

Requires GITHUB_TOKEN; every caller must handle the case where it's
absent (no `gh` CLI is available in this environment, so this uses
plain HTTPS requests instead of shelling out).
"""

import time
from typing import Optional

import requests

import config

FORK_READY_TIMEOUT_SECONDS = 60
FORK_READY_POLL_INTERVAL_SECONDS = 3


def _auth_headers() -> dict:
    return {
        "Authorization": f"Bearer {config.GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }


def create_pull_request(
    owner: str,
    repo: str,
    head_branch: str,
    base_branch: str,
    title: str,
    body: str,
    head_owner: Optional[str] = None,
) -> dict:
    """
    Opens the PR against owner/repo (the upstream), always. When the
    branch actually lives in a fork, pass head_owner (the fork's
    owner login) and GitHub's "owner:branch" head syntax is used --
    this is what makes a cross-repo (fork -> upstream) PR possible.
    """
    if not config.GITHUB_TOKEN:
        return {
            "status": "SKIPPED",
            "reason": "No GITHUB_TOKEN configured; branch was not pushed either.",
        }

    if not owner or not repo:
        return {
            "status": "SKIPPED",
            "reason": "Could not determine owner/repo from the repository URL.",
        }

    head = (
        f"{head_owner}:{head_branch}"
        if head_owner and head_owner != owner
        else head_branch
    )

    response = requests.post(
        f"{config.GITHUB_API_URL}/repos/{owner}/{repo}/pulls",
        headers=_auth_headers(),
        json={
            "title": title,
            "body": body,
            "head": head,
            "base": base_branch,
        },
        timeout=30,
    )

    if response.status_code >= 300:
        return {
            "status": "FAIL",
            "error": f"{response.status_code}: {response.text[:500]}",
        }

    data = response.json()
    return {
        "status": "PASS",
        "url": data.get("html_url"),
        "number": data.get("number"),
    }


def create_or_get_fork(owner: str, repo: str) -> dict:
    """
    Fork owner/repo under the token's account, or return the existing
    fork if the token's account already forked it -- GitHub's fork
    endpoint is idempotent this way, it never errors or duplicates on
    a repeat call.

    A newly created fork is populated asynchronously by GitHub, so
    this polls until the fork is actually reachable before returning,
    rather than handing back a URL that isn't clone-able yet.
    """
    if not config.GITHUB_TOKEN:
        return {"status": "SKIPPED", "reason": "No GITHUB_TOKEN configured."}

    response = requests.post(
        f"{config.GITHUB_API_URL}/repos/{owner}/{repo}/forks",
        headers=_auth_headers(),
        timeout=30,
    )

    if response.status_code not in (200, 202):
        return {
            "status": "FAIL",
            "error": f"{response.status_code}: {response.text[:500]}",
        }

    data = response.json()
    fork_owner = data["owner"]["login"]
    fork_full_name = data["full_name"]
    clone_url = data["clone_url"]

    if not _wait_for_repo_ready(fork_owner, repo):
        return {
            "status": "FAIL",
            "error": (
                f"Fork {fork_full_name} was created but never became "
                f"reachable within {FORK_READY_TIMEOUT_SECONDS}s."
            ),
        }

    return {
        "status": "PASS",
        "fork_owner": fork_owner,
        "full_name": fork_full_name,
        "clone_url": clone_url,
    }


def _wait_for_repo_ready(owner: str, repo: str) -> bool:
    deadline = time.time() + FORK_READY_TIMEOUT_SECONDS

    while time.time() < deadline:
        response = requests.get(
            f"{config.GITHUB_API_URL}/repos/{owner}/{repo}",
            headers=_auth_headers(),
            timeout=15,
        )
        if response.status_code == 200:
            return True
        time.sleep(FORK_READY_POLL_INTERVAL_SECONDS)

    return False
