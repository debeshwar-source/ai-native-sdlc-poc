"""
Language-aware test execution against a real workspace checkout.

Deterministic, no LLM calls. Detects a reasonable test command from
the repo's own manifests rather than assuming Python/pytest, and --
critically -- installs the repo's OWN declared dependencies into an
isolated environment before running anything. Without this, any real
repo with third-party dependencies fails every test on
ModuleNotFoundError regardless of what the Coding Agent does, which
looks identical to a genuine code defect but isn't fixable by retrying.
"""

import json
import subprocess
import sys
import venv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_TIMEOUT_SECONDS = 180
DEPENDENCY_INSTALL_TIMEOUT_SECONDS = 600

# Cached per workspace root within this process, so dependency
# installation happens once per run, not on every retry iteration.
_venv_python_cache: Dict[str, Optional[Path]] = {}

# Set by environment_setup.py (via set_fallback_test_command) when
# none of the hardcoded ecosystems below matched and the Environment
# Agent determined a test command for whatever this repo actually is.
# Checked by detect_test_command as a last resort -- this module
# itself never calls that agent or decides what an unrecognized repo
# needs; it only executes an already-decided, already-sanitized
# command, same as everything else here.
_fallback_test_command_cache: Dict[str, List[str]] = {}


def set_fallback_test_command(root: Path, command: List[str]) -> None:
    _fallback_test_command_cache[str(root)] = command


def run_command(root: Path, command: List[str], timeout: int = DEPENDENCY_INSTALL_TIMEOUT_SECONDS) -> dict:
    """Runs an already-decided command (e.g. environment_setup.py's
    Environment Agent fallback install_command) and reports it in the
    same PASS/FAIL shape as ensure_environment's per-ecosystem
    results. Still just subprocess.run -- no LLM call, no decision
    made here about what's safe to run; that's the caller's job."""
    try:
        result = subprocess.run(
            command, cwd=root, capture_output=True, text=True, timeout=timeout
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return {"status": "FAIL", "error": str(e)}

    return {
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "error": result.stderr[-2000:] if result.returncode != 0 else "",
    }


# ============================================================
# ENVIRONMENT SETUP
# ============================================================

def _requirements_files(root: Path) -> List[Path]:
    candidates = [
        root / "requirements.txt",
        root / "requirements-dev.txt",
        root / "requirements-test.txt",
    ]
    return [p for p in candidates if p.is_file()]


def _venv_python_path(root: Path) -> Path:
    venv_dir = root / ".ai_sdlc_venv"
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _looks_like_python_project(root: Path) -> bool:
    return bool(
        (root / "pyproject.toml").is_file()
        or (root / "setup.py").is_file()
        or _requirements_files(root)
        or any(root.glob("**/test_*.py"))
        or any(root.glob("**/*_test.py"))
    )


def ensure_python_environment(root: Path) -> dict:
    """
    Create an isolated venv for this workspace and pip-install its
    OWN declared dependencies (requirements*.txt, or `pip install .`
    for a pyproject.toml-only project), plus pytest/httpx as a
    baseline. Safe to call multiple times -- cached per root.
    """
    key = str(root)
    if key in _venv_python_cache:
        cached = _venv_python_cache[key]
        return {
            "status": "CACHED",
            "python": str(cached) if cached else None,
        }

    if not _looks_like_python_project(root):
        _venv_python_cache[key] = None
        return {"status": "SKIPPED", "reason": "No Python project detected."}

    venv_dir = root / ".ai_sdlc_venv"
    python_path = _venv_python_path(root)

    if not python_path.exists():
        venv.EnvBuilder(with_pip=True).create(venv_dir)

    install_steps = []

    def _run_pip(args: List[str], timeout: int) -> Optional[subprocess.CompletedProcess]:
        try:
            return subprocess.run(
                [str(python_path), "-m", "pip", "install", "-q", *args],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None

    def _pip_install(args: List[str], target_label: str) -> None:
        result = _run_pip(args, DEPENDENCY_INSTALL_TIMEOUT_SECONDS)
        if result is None:
            install_steps.append(
                {
                    "target": target_label,
                    "status": "FAIL",
                    "error": (
                        f"pip install exceeded "
                        f"{DEPENDENCY_INSTALL_TIMEOUT_SECONDS}s timeout."
                    ),
                }
            )
            return
        install_steps.append(
            {
                "target": target_label,
                "status": "PASS" if result.returncode == 0 else "FAIL",
                "error": result.stderr[-2000:] if result.returncode != 0 else "",
            }
        )

    def _pip_install_requirements_file(req_file: Path) -> None:
        label = str(req_file.relative_to(root))
        extra_index_urls, requirement_specs = _parse_requirements_file(req_file)
        extra_index_args = [
            arg for url in extra_index_urls for arg in ("--extra-index-url", url)
        ]

        bulk_result = _run_pip(
            [*extra_index_args, "-r", str(req_file)],
            DEPENDENCY_INSTALL_TIMEOUT_SECONDS,
        )
        if bulk_result is not None and bulk_result.returncode == 0:
            install_steps.append({"target": label, "status": "PASS", "error": ""})
            return

        # pip aborts installing an ENTIRE requirements file if even one
        # line is unresolvable (e.g. a pinned version with no wheel for
        # this Python) -- so a single bad line silently blocks every
        # other package in the file, including ones the tests actually
        # need. Fall back to installing each requirement independently
        # so the rest still get in.
        bulk_error = (
            f"exceeded {DEPENDENCY_INSTALL_TIMEOUT_SECONDS}s timeout"
            if bulk_result is None
            else bulk_result.stderr[-500:]
        )
        install_steps.append(
            {
                "target": label,
                "status": "PARTIAL",
                "error": (
                    f"Bulk install failed, retrying "
                    f"{len(requirement_specs)} package(s) individually so "
                    f"one incompatible package doesn't block the rest: "
                    f"{bulk_error}"
                ),
            }
        )

        for spec in requirement_specs:
            result = _run_pip([*extra_index_args, spec], 180)
            if result is None:
                install_steps.append(
                    {"target": spec, "status": "FAIL", "error": "install timed out"}
                )
            else:
                install_steps.append(
                    {
                        "target": spec,
                        "status": "PASS" if result.returncode == 0 else "FAIL",
                        "error": (
                            result.stderr[-500:] if result.returncode != 0 else ""
                        ),
                    }
                )

    # Baseline test tooling, in case the repo's own manifest treats
    # it as a dev-only dependency it doesn't declare.
    _pip_install(["pytest", "httpx"], "baseline test packages")

    requirements = _requirements_files(root)
    for req_file in requirements:
        _pip_install_requirements_file(req_file)

    if not requirements and (root / "pyproject.toml").is_file():
        _pip_install([str(root)], "pyproject.toml")

    _venv_python_cache[key] = python_path

    overall_status = "READY"
    if any(step["status"] == "FAIL" for step in install_steps):
        overall_status = "READY_WITH_FAILURES"

    return {
        "status": overall_status,
        "python": str(python_path),
        "install_steps": install_steps,
    }


def _parse_requirements_file(req_file: Path):
    """
    Minimal requirements.txt parser: separates global pip options
    (currently just --extra-index-url, applied to every subsequent
    install) from individual requirement specs. Nested -r includes
    and other pip flags are skipped rather than fully resolved --
    good enough for the common case, not a full pip parser.
    """
    extra_index_urls: List[str] = []
    requirement_specs: List[str] = []

    for raw_line in req_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue
        if line.startswith("--extra-index-url"):
            parts = line.split(None, 1)
            if len(parts) == 2:
                extra_index_urls.append(parts[1].strip())
            continue
        if line.startswith("-"):
            continue

        requirement_specs.append(line)

    return extra_index_urls, requirement_specs


def ensure_node_environment(root: Path) -> dict:
    if not (root / "package.json").is_file():
        return {"status": "SKIPPED", "reason": "No package.json."}

    try:
        result = subprocess.run(
            ["npm", "install"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=DEPENDENCY_INSTALL_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return {"status": "FAIL", "error": str(e)}

    return {
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "error": result.stderr[-2000:] if result.returncode != 0 else "",
    }


def ensure_environment(root: Path) -> dict:
    """
    Best-effort dependency setup for whatever ecosystem(s) this repo
    uses. Never raises -- a failed install is reported, not fatal,
    since tests may still partially run (and the failure will show
    up clearly in test output either way).
    """
    results = {}

    python_result = ensure_python_environment(root)
    if python_result.get("status") != "SKIPPED":
        results["python"] = python_result

    if (root / "package.json").is_file():
        results["node"] = ensure_node_environment(root)

    if (root / "go.mod").is_file():
        try:
            result = subprocess.run(
                ["go", "mod", "download"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=DEPENDENCY_INSTALL_TIMEOUT_SECONDS,
            )
            results["go"] = {
                "status": "PASS" if result.returncode == 0 else "FAIL",
                "error": result.stderr[-2000:] if result.returncode != 0 else "",
            }
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            results["go"] = {"status": "FAIL", "error": str(e)}

    return results


# ============================================================
# TEST EXECUTION
# ============================================================

def detect_test_command(root: Path) -> Optional[List[str]]:
    has_pytest_files = any(root.glob("**/test_*.py")) or any(
        root.glob("**/*_test.py")
    )
    has_python_project = (
        (root / "pyproject.toml").exists()
        or (root / "setup.cfg").exists()
        or (root / "pytest.ini").exists()
    )
    if has_pytest_files or has_python_project:
        # Use the isolated venv set up by ensure_python_environment,
        # if one was; otherwise fall back to this process's own
        # interpreter (e.g. a toy repo with no real dependencies).
        python_executable = _venv_python_cache.get(str(root)) or sys.executable
        return [str(python_executable), "-m", "pytest", "-q"]

    package_json = root / "package.json"
    if package_json.exists():
        try:
            data = json.loads(package_json.read_text(encoding="utf-8"))
            if "test" in data.get("scripts", {}):
                return ["npm", "test", "--silent"]
        except (json.JSONDecodeError, OSError):
            pass

    if (root / "go.mod").exists():
        return ["go", "test", "./..."]

    if (root / "Cargo.toml").exists():
        return ["cargo", "test"]

    # `or None` folds an empty list (the Environment Agent couldn't
    # determine a safe command) into the same "nothing detected"
    # signal run_tests already treats as SKIPPED.
    return _fallback_test_command_cache.get(str(root)) or None


def run_tests(root: Path, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> dict:
    command = detect_test_command(root)

    if command is None:
        return {
            "status": "SKIPPED",
            "command": None,
            "output": "",
            "error": "No recognized test command detected in this repository.",
            "return_code": None,
        }

    try:
        result = subprocess.run(
            command, cwd=root, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        return {
            "status": "FAIL",
            "command": " ".join(command),
            "output": stdout,
            "error": f"Tests exceeded the {timeout}-second timeout.",
            "return_code": None,
        }
    except FileNotFoundError as e:
        return {
            "status": "SKIPPED",
            "command": " ".join(command),
            "output": "",
            "error": f"Test tool not available in this environment: {e}",
            "return_code": None,
        }

    return {
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "command": " ".join(command),
        "output": result.stdout[-8000:],
        "error": result.stderr[-4000:],
        "return_code": result.returncode,
    }
