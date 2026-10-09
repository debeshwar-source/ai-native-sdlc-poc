"""
Deterministic static/security analysis: Semgrep (pattern-based static
analysis; run separately for its secrets ruleset and its general
security/OWASP rulesets, see below) over the actual diff, and
pip-audit (known-vulnerable-dependency check against the PyPI
Advisory DB) over the repo's own declared Python dependencies.

No LLM calls. Runs in a venv dedicated to these analysis tools --
kept separate from the repo's own dependency venv so a pinned version
one needs can never conflict with a pinned version the other needs.

This exists to give the Review Agent (agents/review.py) a signal to
reconcile against instead of trusting an LLM's read of a diff alone.
Semgrep's general security/OWASP rulesets have a real false-positive
rate, so those findings are handed to the Review Agent as context for
it to judge, not force-blocking. A hardcoded secret (our own bundled
rules file, see SECRETS_RULES_FILE below) and a known CVE matched
against the exact installed version (pip-audit) both have a
near-zero false-positive rate, though, so those ARE force-blocking
regardless of what the LLM concludes -- `blocking_findings` below,
which agents/review.py folds into its own findings list as an
orchestrator-level invariant, the same way agents/testing.py already
treats a failed test run as authoritative over the model's own
summary.
"""

import json
import subprocess
import venv
from pathlib import Path
from typing import Dict, List, Optional

ANALYSIS_VENV_DIR = ".ai_sdlc_analysis_venv"
TOOL_INSTALL_TIMEOUT_SECONDS = 300
SEMGREP_TIMEOUT_SECONDS = 300
PIP_AUDIT_TIMEOUT_SECONDS = 180

# Kept as two separate scans (two invocations, not one --config list)
# so a finding's ruleset -- and therefore its false-positive risk --
# is known from which call produced it, not guessed from its rule id.
#
# Secrets use a small rules file we ship ourselves (tools/
# semgrep_rules/secrets.yml), not Semgrep's own free-tier `p/secrets`
# registry config -- verified by hand that the latter needs `semgrep
# login` for its real (entropy/provider-validated) rules, and
# anonymously does NOT catch something as basic as a hardcoded AWS
# key. Our own rules match distinctive, fixed token formats (AWS,
# GitHub, Slack, Stripe, private-key headers), so they keep the same
# near-zero-false-positive guarantee without that dependency.
SECRETS_RULES_FILE = Path(__file__).resolve().parent / "semgrep_rules" / "secrets.yml"
SECURITY_CONFIGS = ["p/security-audit", "p/owasp-top-ten"]

# Cached per workspace root within this process, mirroring
# test_runner.py's own venv cache -- the tool venv only needs
# creating/installing once per run.
_tools_ready_cache: Dict[str, bool] = {}


def _tool_path(root: Path, name: str) -> Path:
    return root / ANALYSIS_VENV_DIR / "bin" / name


def _ensure_analysis_tools(root: Path) -> dict:
    key = str(root)
    if _tools_ready_cache.get(key):
        return {"status": "CACHED"}

    venv_dir = root / ANALYSIS_VENV_DIR
    if not venv_dir.exists():
        venv.EnvBuilder(with_pip=True).create(venv_dir)

    try:
        install_result = subprocess.run(
            [str(_tool_path(root, "python")), "-m", "pip", "install", "-q", "semgrep", "pip-audit"],
            cwd=root, capture_output=True, text=True, timeout=TOOL_INSTALL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "FAIL",
            "reason": f"Analysis tool install exceeded {TOOL_INSTALL_TIMEOUT_SECONDS}s timeout.",
        }
    if install_result.returncode != 0:
        return {
            "status": "FAIL",
            "reason": f"Analysis tool install failed: {install_result.stderr[-1000:]}",
        }

    _tools_ready_cache[key] = True
    return {"status": "READY"}


def _run_semgrep_scan(root: Path, configs: List[str], ruleset: str, targets: List[str]) -> dict:
    config_args = [arg for cfg in configs for arg in ("--config", cfg)]

    try:
        result = subprocess.run(
            [str(_tool_path(root, "semgrep")), "scan", *config_args, "--json", "--quiet", "--no-git-ignore", *targets],
            cwd=root, capture_output=True, text=True, timeout=SEMGREP_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "SKIPPED",
            "reason": f"Semgrep ({ruleset}) exceeded {SEMGREP_TIMEOUT_SECONDS}s timeout.",
            "findings": [],
        }

    # Semgrep exits 1 for "ran fine, found matches" -- only a
    # different return code means the tool itself failed.
    if result.returncode not in (0, 1):
        return {
            "status": "SKIPPED",
            "reason": f"Semgrep ({ruleset}) exited {result.returncode}: {result.stderr[-1000:]}",
            "findings": [],
        }

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return {"status": "SKIPPED", "reason": f"Semgrep ({ruleset}) produced unparseable output.", "findings": []}

    findings = [
        {
            "ruleset": ruleset,
            "rule_id": item.get("check_id"),
            "path": item.get("path"),
            "line": (item.get("start") or {}).get("line"),
            "severity": (item.get("extra") or {}).get("severity", "INFO"),
            "message": (item.get("extra") or {}).get("message", ""),
        }
        for item in payload.get("results", [])
    ]

    return {"status": "DONE", "findings": findings}


def run_semgrep(root: Path, changed_files: Optional[List[str]] = None) -> dict:
    """
    Runs Semgrep's secrets ruleset and its general security/OWASP
    rulesets as two separate scans, scoped to `changed_files` when
    given (the same diff the Review Agent evaluates) or the whole repo
    otherwise. Deterministic, no LLM call. Never raises -- a tool-setup
    or scan failure is reported as its own status, not fatal to the
    pipeline.
    """
    setup = _ensure_analysis_tools(root)
    if setup["status"] == "FAIL":
        return {"status": "SKIPPED", "reason": setup["reason"], "findings": []}

    targets = [f for f in changed_files if (root / f).is_file()] if changed_files else ["."]
    if not targets:
        return {"status": "SKIPPED", "reason": "No existing changed files to scan.", "findings": []}

    secrets = _run_semgrep_scan(root, [str(SECRETS_RULES_FILE)], "secrets", targets)
    security = _run_semgrep_scan(root, SECURITY_CONFIGS, "security", targets)

    return {
        "status": "DONE" if secrets["status"] == "DONE" or security["status"] == "DONE" else "SKIPPED",
        "findings": secrets.get("findings", []) + security.get("findings", []),
        "secrets_scan": secrets,
        "security_scan": security,
    }


DEPENDENCY_MANIFEST_NAMES = {
    "requirements.txt", "requirements-dev.txt", "requirements_dev.txt",
    "requirements-test.txt", "pyproject.toml",
}


def run_pip_audit(root: Path, changed_files: Optional[List[str]] = None) -> dict:
    """
    Checks the repo's OWN declared Python dependencies against the
    PyPI Advisory DB (via pip-audit) for known vulnerabilities.
    Deterministic, no LLM call. Skipped entirely for non-Python repos.

    When `changed_files` is given (the Review Agent's actual diff
    scope) and it does NOT include a dependency manifest, this is
    skipped entirely -- pip-audit audits the WHOLE manifest, not just
    what changed, so on any real repo with existing pinned
    dependencies it will surface pre-existing vulnerabilities that
    have nothing to do with this change and that the Coding Agent has
    no path-scoped permission to fix anyway (Architecture never listed
    requirements.txt as a file to touch). Force-blocking release on
    those makes review unpassable forever, not safer -- this only
    gates when the change ITSELF introduced or touched a dependency.
    """
    if changed_files is not None and not any(
        Path(f).name in DEPENDENCY_MANIFEST_NAMES for f in changed_files
    ):
        return {
            "status": "SKIPPED",
            "reason": "This change didn't touch a dependency manifest.",
            "findings": [],
        }

    requirements = [
        p for p in (
            root / "requirements.txt",
            root / "requirements-dev.txt",
            root / "requirements_dev.txt",
            root / "requirements-test.txt",
        )
        if p.is_file()
    ]
    if not requirements and not (root / "pyproject.toml").is_file():
        return {"status": "SKIPPED", "reason": "No Python dependency manifest detected.", "findings": []}

    setup = _ensure_analysis_tools(root)
    if setup["status"] == "FAIL":
        return {"status": "SKIPPED", "reason": setup["reason"], "findings": []}

    manifests = requirements or [root / "pyproject.toml"]
    findings: List[dict] = []
    errors: List[str] = []

    for manifest in manifests:
        label = str(manifest.relative_to(root))
        args = ["-r", label] if manifest.suffix == ".txt" else ["."]

        try:
            result = subprocess.run(
                [str(_tool_path(root, "pip-audit")), *args, "--format", "json", "--progress-spinner", "off"],
                cwd=root, capture_output=True, text=True, timeout=PIP_AUDIT_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            errors.append(f"{label}: pip-audit exceeded {PIP_AUDIT_TIMEOUT_SECONDS}s timeout.")
            continue

        # pip-audit exits 1 for "ran fine, found vulnerabilities" --
        # only a different return code means the tool itself failed.
        if result.returncode not in (0, 1):
            errors.append(f"{label}: pip-audit exited {result.returncode}: {result.stderr[-500:]}")
            continue

        try:
            payload = json.loads(result.stdout or "[]")
        except json.JSONDecodeError:
            errors.append(f"{label}: pip-audit produced unparseable output.")
            continue

        # pip-audit's JSON is either a bare list of {name, version,
        # vulns: [...]} entries, or (newer versions) a
        # {"dependencies": [...]} envelope around the same shape.
        dependencies = payload if isinstance(payload, list) else payload.get("dependencies", [])
        for dep in dependencies:
            for vuln in dep.get("vulns", []) or []:
                findings.append(
                    {
                        "source": label,
                        "package": dep.get("name"),
                        "installed_version": dep.get("version"),
                        "vulnerability_id": vuln.get("id"),
                        "fix_versions": vuln.get("fix_versions", []),
                        "description": (vuln.get("description") or "")[:500],
                    }
                )

    return {"status": "DONE", "findings": findings, "errors": errors}


def run(root: Path, changed_files: Optional[List[str]] = None) -> dict:
    """
    Both scans, combined into one deterministic report. `blocking_findings`
    is the near-zero-false-positive subset (hardcoded secrets + known
    CVEs at the exact installed version) that agents/review.py treats
    as an orchestrator-level invariant, not a suggestion. Everything
    else in `security_findings` is handed to the Review Agent as
    context for its own judgment, since general security-audit/OWASP
    findings do have a real false-positive rate.
    """
    semgrep_result = run_semgrep(root, changed_files)
    pip_audit_result = run_pip_audit(root, changed_files)

    secrets_findings = [f for f in semgrep_result.get("findings", []) if f.get("ruleset") == "secrets"]
    security_findings = [f for f in semgrep_result.get("findings", []) if f.get("ruleset") == "security"]
    blocking_findings = secrets_findings + pip_audit_result.get("findings", [])

    return {
        "semgrep": semgrep_result,
        "pip_audit": pip_audit_result,
        "security_findings": security_findings,
        "blocking_findings": blocking_findings,
        "has_blocking_findings": bool(blocking_findings),
    }
