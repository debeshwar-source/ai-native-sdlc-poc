"""Knowledge base: folders, local locations and git repos the user points
the agents at as reference material.

Sources are persisted to knowledge_base.json. Folders are used in place
(read-only); a repo is shallow-cloned once into .knowledge_cache/ when a
run starts (resolve_paths). Browsing for a folder opens the operating
system's own folder picker on the machine running the app.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

KB_PATH = Path(__file__).parent / "knowledge_base.json"
CACHE_ROOT = Path(__file__).parent / ".knowledge_cache"

Source = Dict[str, str]  # {"kind": "folder" | "repo", "location": str}


def load() -> List[Source]:
    try:
        data = json.loads(KB_PATH.read_text(encoding="utf-8"))
        return [s for s in data if s.get("kind") in ("folder", "repo") and s.get("location")]
    except (OSError, ValueError):
        return []


def save(sources: List[Source]) -> None:
    KB_PATH.write_text(json.dumps(sources, indent=2), encoding="utf-8")


def classify(location: str) -> Optional[Source]:
    """A source for what the user typed/picked, or None if it is neither
    a git URL nor an existing folder."""
    location = location.strip()
    if not location:
        return None
    if location.startswith(("http://", "https://", "git@", "ssh://")):
        return {"kind": "repo", "location": location}
    path = Path(location).expanduser()
    if path.is_dir():
        return {"kind": "folder", "location": str(path.resolve())}
    return None


def browse_for_folder() -> Optional[str]:
    """Open the OS folder picker; the chosen path, or None if cancelled
    or no picker is available. Blocks until the dialog closes."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["osascript", "-e", 'POSIX path of (choose folder with prompt "Add to knowledge base")'],
                capture_output=True, text=True, timeout=300,
            )
        else:
            script = (
                "import tkinter, tkinter.filedialog as f;"
                "r = tkinter.Tk(); r.withdraw(); r.attributes('-topmost', True);"
                "print(f.askdirectory(title='Add to knowledge base'))"
            )
            out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return None
    path = out.stdout.strip()
    return path.rstrip("/") or None if out.returncode == 0 and path else None


def _clone_dir(url: str) -> Path:
    return CACHE_ROOT / hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def resolve_paths(sources: List[Source]) -> List[str]:
    """Local directories for every source that can be read right now:
    folders as-is, repos shallow-cloned (or refreshed) into the cache.
    Sources that vanished or fail to clone are skipped."""
    paths = []
    for src in sources:
        if src["kind"] == "folder":
            if Path(src["location"]).is_dir():
                paths.append(src["location"])
            continue
        target = _clone_dir(src["location"])
        if target.exists():
            subprocess.run(["git", "-C", str(target), "pull", "--ff-only", "-q"], capture_output=True)
        else:
            CACHE_ROOT.mkdir(parents=True, exist_ok=True)
            done = subprocess.run(
                ["git", "clone", "--depth", "1", "-q", src["location"], str(target)], capture_output=True
            )
            if done.returncode != 0:
                continue
        paths.append(str(target))
    return paths
