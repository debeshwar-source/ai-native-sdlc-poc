"""
Issue tracker / wiki abstraction for the Release & Operations stage.

No Jira/Confluence connector is authorized in this environment, so
only a local-file mock is implemented here -- but a real integration
is a drop-in: implement IssueTracker/Wiki and swap it in at the call
site in agents/release.py.
"""

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List


class IssueTracker(ABC):
    @abstractmethod
    def create_issue(
        self,
        title: str,
        description: str,
        acceptance_criteria: List[str],
        status: str,
    ) -> dict:
        raise NotImplementedError


class Wiki(ABC):
    @abstractmethod
    def create_page(self, title: str, sections: List[Dict[str, str]]) -> dict:
        raise NotImplementedError


def _slugify(text: str, max_length: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug or "item")[:max_length]


class LocalMockIssueTracker(IssueTracker):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def create_issue(
        self,
        title: str,
        description: str,
        acceptance_criteria: List[str],
        status: str = "DONE",
    ) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        issue_id = f"SDLC-{_slugify(title, 24)}"

        record = {
            "issue_id": issue_id,
            "title": title,
            "description": description,
            "acceptance_criteria": acceptance_criteria,
            "status": status,
        }
        (self.output_dir / f"{issue_id}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )

        lines = [
            f"# {issue_id}: {title}",
            "",
            f"**Status:** {status}",
            "",
            description,
            "",
            "## Acceptance Criteria",
            "",
        ] + [f"- {item}" for item in acceptance_criteria]

        md_path = self.output_dir / f"{issue_id}.md"
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        return {"issue_id": issue_id, "path": str(md_path)}


class LocalMockWiki(Wiki):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def create_page(self, title: str, sections: List[Dict[str, str]]) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        lines = [f"# {title}", ""]
        for section in sections:
            lines.append(f"## {section.get('heading', '')}")
            lines.append("")
            lines.append(section.get("content", ""))
            lines.append("")

        path = self.output_dir / f"{_slugify(title)}.md"
        path.write_text("\n".join(lines), encoding="utf-8")

        return {"title": title, "path": str(path)}
