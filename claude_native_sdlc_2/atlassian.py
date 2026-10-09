"""Jira & Confluence access for agents, on request.

An agent created with "Jira & Confluence access" gets the open-source
mcp-atlassian MCP server (run with `uvx`) attached for its runs. In "read"
mode the server is started with READ_ONLY_MODE, which removes every tool that
could change anything (58 read tools, none that write); "write" mode leaves
the full set (create/update/transition issues, comments, pages, ...).

Credentials are an Atlassian API token (https://id.atlassian.com/manage-profile/security/api-tokens)
plus the account email and the site URL. A link alone can't grant access --
Jira and Confluence need to know who is asking -- but any Jira/Confluence link
is enough to tell us the site and project. They live in
atlassian_credentials.json (owner-only permissions, git-ignored) and are used
only to start the MCP server and to test the connection.
"""

import base64
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

CREDENTIALS_PATH = Path(__file__).parent / "atlassian_credentials.json"
TOKEN_URL = "https://id.atlassian.com/manage-profile/security/api-tokens"
MODES = ("", "read", "write")   # "" = no access


def parse_link(link: str) -> Dict[str, str]:
    """{"site": "https://x.atlassian.net", "project_key": "KAN"} from any pasted
    Jira/Confluence link or bare site URL; empty strings for what isn't there."""
    link = (link or "").strip()
    if link and "://" not in link:
        link = "https://" + link
    parts = urllib.parse.urlparse(link)
    if "." not in parts.netloc:
        return {"site": "", "project_key": ""}
    site = f"{parts.scheme or 'https'}://{parts.netloc}"
    text = urllib.parse.unquote(parts.path + "?" + parts.query)
    key = (re.search(r"/projects/([A-Z][A-Z0-9_]+)", text)
           or re.search(r"project\s*=\s*\"?([A-Z][A-Z0-9_]+)", text)
           or re.search(r"/browse/([A-Z][A-Z0-9_]+)-\d+", text))
    return {"site": site, "project_key": key.group(1) if key else ""}


def load() -> Dict[str, str]:
    try:
        data = json.loads(CREDENTIALS_PATH.read_text(encoding="utf-8"))
        return {k: str(data.get(k) or "") for k in ("site", "email", "token", "project_key")}
    except (OSError, ValueError, AttributeError):
        return {"site": "", "email": "", "token": "", "project_key": ""}


def save(site: str, email: str, token: str, project_key: str = "") -> None:
    data = {"site": site.rstrip("/"), "email": email.strip(), "token": token.strip(), "project_key": project_key.strip()}
    CREDENTIALS_PATH.write_text(json.dumps(data, indent=1), encoding="utf-8")
    try:
        os.chmod(CREDENTIALS_PATH, 0o600)
    except OSError:
        pass


def clear() -> None:
    try:
        CREDENTIALS_PATH.unlink()
    except OSError:
        pass


def is_configured(creds: Optional[Dict[str, str]] = None) -> bool:
    creds = creds or load()
    return bool(creds["site"] and creds["email"] and creds["token"])


def test_connection(creds: Optional[Dict[str, str]] = None, timeout: int = 15) -> Dict[str, Any]:
    """Check the saved credentials against Jira and Confluence over their REST
    APIs: {"jira": (ok, message), "confluence": (ok, message)}."""
    creds = creds or load()
    auth = base64.b64encode(f"{creds['email']}:{creds['token']}".encode()).decode()

    def call(path: str, describe) -> Any:
        req = urllib.request.Request(creds["site"] + path, headers={"Authorization": f"Basic {auth}", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return True, describe(json.loads(resp.read().decode() or "{}"))
        except urllib.error.HTTPError as exc:
            return False, {401: "Rejected: wrong email or API token.", 403: "Signed in, but this account isn't allowed here.",
                           404: "Not found: is the site URL right (and is this product on the site)?"}.get(exc.code, f"HTTP {exc.code}")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return False, f"Could not reach {creds['site']}: {getattr(exc, 'reason', exc)}"

    return {
        "jira": call("/rest/api/3/myself", lambda d: f"Connected as {d.get('displayName', 'unknown')}"),
        "confluence": call("/wiki/rest/api/space?limit=1", lambda d: "Connected"),
    }


def mcp_server_config(mode: str, creds: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """The stdio MCP server definition for `mode` ("read" or "write")."""
    creds = creds or load()
    env = {
        "JIRA_URL": creds["site"], "JIRA_USERNAME": creds["email"], "JIRA_API_TOKEN": creds["token"],
        "CONFLUENCE_URL": creds["site"] + "/wiki", "CONFLUENCE_USERNAME": creds["email"],
        "CONFLUENCE_API_TOKEN": creds["token"],
        "READ_ONLY_MODE": "false" if mode == "write" else "true",
        "PATH": os.environ.get("PATH", ""),
    }
    return {"type": "stdio", "command": "uvx", "args": ["mcp-atlassian"], "env": env}


def operator_prompt(mode: str, creds: Optional[Dict[str, str]] = None) -> str:
    """System prompt of the short-lived helper that actually talks to Jira and
    Confluence for an agent's `atlassian_task` call."""
    creds = creds or load()
    project = f" The user's usual Jira project key is {creds['project_key']}." if creds.get("project_key") else ""
    can = ("You may create and update issues, comments and pages -- but only exactly what the task asks for. "
           "Never delete anything unless the task explicitly says to."
           if mode == "write" else
           "You have READ-ONLY access: search and read, never change anything. If the task needs a change, say so.")
    return f"""
You carry out one task in Jira and/or Confluence at {creds['site']}, using the `mcp__atlassian__*` tools.{project}
{can}
Do exactly the task, nothing more. Quote issue keys, page titles and links exactly. When you create or change
something, report precisely what (key/title/link) so the requester can check it. If something fails, say what
failed and why instead of guessing. Return a concise result.
"""


def prompt_section() -> str:
    """What every agent is told about Jira & Confluence."""
    return """
============================================================
JIRA & CONFLUENCE (on request)
============================================================
You can reach Jira and Confluence through the `atlassian_task` tool: describe what to do in plain language and set
`write` to true only if it needs to create or change something. Use it only when the user's request actually involves
Jira or Confluence -- e.g. if they ask you to draft stories, draft them in your answer first; call the tool to put them
into Jira only once they ask you to. Do not look anything up in Jira unprompted.
You do NOT need to ask the user for a connection, token or permission yourself: when the tool needs them it asks the user
directly, and tells you if they declined (then accept that and do not retry). Include everything the task needs in `task`
(project key, issue type, titles, descriptions), because the helper that runs it cannot see this conversation.
"""
