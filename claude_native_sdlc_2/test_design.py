"""Design layer checks. Run: python3 -m pytest -q test_design.py"""

import design as d

FILES = ["app/routes.py", "app/db.py", "main.py", "tests/test_x.py"]
ARCH = {"files_to_modify": ["app/routes.py"], "files_to_create": ["app/health.py"], "files_to_delete": []}


def _raw():
    return {
        "existing_overview": "An API over a database.",
        "proposed_overview": "Adds a health module.",
        "components": [
            {"id": "api", "name": "API", "description": "routes", "files": ["app/routes.py", "ghost.py"], "is_new": False},
            {"id": "db", "name": "DB", "description": "storage", "files": ["app/db.py"], "is_new": False},
            {"id": "ghost", "name": "Invented", "description": "x", "files": ["nope.py"], "is_new": False},
            {"id": "health", "name": "Health", "description": "probe", "files": [], "is_new": True},
        ],
        "relationships": [
            {"from": "api", "to": "db", "label": "queries", "is_new": False},
            {"from": "api", "to": "health", "label": "calls", "is_new": False},
            {"from": "api", "to": "nowhere", "label": "x", "is_new": False},
            {"from": "api", "to": "api", "label": "self", "is_new": False},
        ],
    }


def test_status_is_computed_from_the_plan_not_the_model():
    r = d.normalize(_raw(), ARCH, FILES)
    status = {c["id"]: c["status"] for c in r["components"]}
    assert status == {"api": "MODIFIED", "db": "UNCHANGED", "health": "NEW"}


def test_invented_components_and_files_are_dropped():
    r = d.normalize(_raw(), ARCH, FILES)
    assert "ghost" not in {c["id"] for c in r["components"]}
    api = next(c for c in r["components"] if c["id"] == "api")
    assert api["files"] == ["app/routes.py"]
    assert any("dropped" in n for n in r["notes"])


def test_bad_edges_dropped_and_edges_to_new_components_are_new():
    r = d.normalize(_raw(), ARCH, FILES)
    edges = {(e["from"], e["to"]): e["is_new"] for e in r["relationships"]}
    assert edges == {("api", "db"): False, ("api", "health"): True}


def test_single_new_component_adopts_created_files():
    r = d.normalize(_raw(), ARCH, FILES)
    health = next(c for c in r["components"] if c["id"] == "health")
    assert health["files"] == ["app/health.py"]
    assert not any(c["id"] == "new-files" for c in r["components"])


def test_every_planned_change_is_visible():
    arch = {"files_to_modify": ["main.py"], "files_to_create": ["app/a.py", "app/b.py"], "files_to_delete": []}
    raw = _raw()
    raw["components"] = raw["components"][:2]  # no component claims main.py or the new files
    r = d.normalize(raw, arch, FILES)
    shown = {f for c in r["components"] for f in c["files"]}
    assert {"main.py", "app/a.py", "app/b.py"} <= shown


def test_removed_status():
    arch = {"files_to_modify": [], "files_to_create": [], "files_to_delete": ["app/db.py"]}
    r = d.normalize(_raw(), arch, FILES)
    assert next(c for c in r["components"] if c["id"] == "db")["status"] == "REMOVED"


def test_fallback_when_model_gives_nothing():
    for junk in (None, "junk", {}, {"components": "x"}):
        r = d.normalize(junk, ARCH, FILES)
        assert r["source"] == "derived" and r["components"]
    assert d.normalize(None, {}, []) is None


def test_never_raises():
    assert d.normalize({"components": [None, 5, {"name": None}]}, None, None) is not None or True


def test_no_change_has_no_highlights():
    r = d.normalize(_raw(), {"files_to_modify": [], "files_to_create": [], "files_to_delete": []}, FILES)
    raw = _raw()
    raw["components"] = [c for c in raw["components"] if not c["is_new"]]
    r = d.normalize(raw, {"files_to_modify": [], "files_to_create": [], "files_to_delete": []}, FILES)
    assert r["has_change"] is False


def test_dot_output_is_wellformed_and_distinct():
    r = d.normalize(_raw(), ARCH, FILES)
    existing, proposed = d.existing_dot(r), d.proposed_dot(r)
    for dot in (existing, proposed):
        assert dot.startswith("digraph") and dot.rstrip().endswith("}")
        assert dot.count("{") == dot.count("}")
    assert '"health"' not in existing                      # new things aren't in today's picture
    assert '"health"' in proposed and "#16A34A" in proposed  # ...but are in the proposal, in green
    assert "#DC2626" in proposed                            # new relationship highlighted
    assert "->" in existing and '"api" -> "health"' not in existing


def test_dot_escapes_quotes():
    raw = _raw()
    raw["components"][0]["name"] = 'A "quoted" name'
    dot = d.proposed_dot(d.normalize(raw, ARCH, FILES))
    assert 'A \\"quoted\\" name' in dot
