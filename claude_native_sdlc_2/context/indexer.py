"""
Engineering Knowledge & Context Layer.

Deterministic, no LLM calls. Builds a language-agnostic index of a
real cloned repository: file tree, detected primary language,
ecosystem/manifest hints, small manifest file contents, and a
tree-sitter-based repo map (see below). Agents get this as a cheap
starting overview; deep exploration beyond it happens through the
real read_file/search_code tools, not by stuffing the whole repo into
a prompt.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

EXCLUDED_DIR_NAMES = {
    ".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv",
    ".ai_sdlc_venv", ".ai_sdlc_analysis_venv", "dist", "build",
    ".next", ".turbo", "target", "vendor",
    ".mypy_cache", ".idea", ".vscode", "coverage", ".tox", "egg-info",
}

EXCLUDED_FILE_SUFFIXES = {".pyc", ".lock", ".map"}

LANGUAGE_BY_EXTENSION = {
    ".py": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".go": "Go",
    ".java": "Java",
    ".rb": "Ruby",
    ".rs": "Rust",
    ".php": "PHP",
    ".cs": "C#",
    ".cpp": "C++",
    ".c": "C",
    ".kt": "Kotlin",
}

MANIFEST_HINTS = {
    "requirements.txt": ("Python", "pip"),
    "pyproject.toml": ("Python", "pip/poetry"),
    "Pipfile": ("Python", "pipenv"),
    "package.json": ("JavaScript/TypeScript", "npm"),
    "go.mod": ("Go", "go modules"),
    "Cargo.toml": ("Rust", "cargo"),
    "pom.xml": ("Java", "maven"),
    "build.gradle": ("Java/Kotlin", "gradle"),
    "Gemfile": ("Ruby", "bundler"),
    "composer.json": ("PHP", "composer"),
}

MAX_MANIFEST_BYTES = 20_000
MAX_TREE_LINES = 400


@dataclass
class RepoIndex:
    root: Path
    file_list: List[str]
    excluded: List[str]
    primary_language: Optional[str]
    ecosystem_hints: List[str]
    manifest_files: Dict[str, str]
    tree_text: str
    repo_map_text: str


# ============================================================
# TREE-SITTER REPO MAP
# ============================================================
#
# Aider's repo-map technique, adapted: tree-sitter-parse every source
# file this project has a grammar for, extract definition/reference
# tags via each language package's own bundled TAGS_QUERY (the same
# tag-query convention GitHub code navigation and Aider itself use --
# no hand-written queries here), build a weighted reference graph
# between files (an edge from a file that references a name to every
# file that defines it, split across ambiguous multi-definition
# names), and rank both files and the definitions within them by a
# plain, unpersonalized PageRank.
#
# This turns "file tree + manifests" into "here's what actually
# matters in this repo," handed to agents up front instead of them
# rediscovering structure via live Read/Grep/Glob calls on every run.
#
# Deterministic, no LLM calls, and never fatal: a repo with no
# supported language, or this process missing the optional tree-sitter
# packages, just gets an empty map -- the file tree/manifest overview
# alone, exactly as before this existed.

try:
    from tree_sitter import Language, Parser, Query, QueryCursor
    import tree_sitter_go as _ts_go
    import tree_sitter_javascript as _ts_javascript
    import tree_sitter_python as _ts_python
    import tree_sitter_rust as _ts_rust
    import tree_sitter_typescript as _ts_typescript

    _TREE_SITTER_AVAILABLE = True
except ImportError:
    _TREE_SITTER_AVAILABLE = False

MAX_REPO_MAP_FILES = 3000          # bound cost on a pathologically large repo
MAX_REPO_MAP_FILE_BYTES = 512_000  # skip tagging individual huge generated files
MAX_REPO_MAP_CHARS = 8_000
TOP_DEFS_PER_FILE = 6
PAGERANK_DAMPING = 0.85
PAGERANK_ITERATIONS = 50

_LANGUAGE_SPECS: Optional[Dict[str, Tuple["Parser", "Query"]]] = None


@dataclass
class _Tag:
    kind: str      # "def" or "ref"
    name: str
    line: int       # 0-indexed
    signature: str   # first line of the definition's span, for display


def _build_language_specs() -> Dict[str, Tuple["Parser", "Query"]]:
    if not _TREE_SITTER_AVAILABLE:
        return {}

    def spec(raw_language, query_text: str) -> Tuple["Parser", "Query"]:
        language = Language(raw_language)
        # Parser/Query are built once per language, not per file --
        # Query compiles the pattern text, which isn't free, and
        # there's no reason to pay that thousands of times over.
        return Parser(language), Query(language, query_text)

    return {
        "Python": spec(_ts_python.language(), _ts_python.TAGS_QUERY),
        "JavaScript": spec(_ts_javascript.language(), _ts_javascript.TAGS_QUERY),
        # TypeScript's own TAGS_QUERY only covers TS-specific
        # constructs (interfaces, type annotations, abstract
        # classes) -- its grammar is a superset of JS's for
        # functions/classes/calls, so JS's query still matches those
        # node types unchanged. Combining both is what gets full
        # coverage for a .ts/.tsx file.
        "TypeScript": spec(
            _ts_typescript.language_typescript(),
            _ts_javascript.TAGS_QUERY + "\n" + _ts_typescript.TAGS_QUERY,
        ),
        "Go": spec(_ts_go.language(), _ts_go.TAGS_QUERY),
        "Rust": spec(_ts_rust.language(), _ts_rust.TAGS_QUERY),
    }


def _get_language_specs() -> Dict[str, Tuple["Parser", "Query"]]:
    global _LANGUAGE_SPECS
    if _LANGUAGE_SPECS is None:
        _LANGUAGE_SPECS = _build_language_specs()
    return _LANGUAGE_SPECS


def _extract_tags(path: Path, language_name: str) -> List[_Tag]:
    spec = _get_language_specs().get(language_name)
    if spec is None:
        return []

    try:
        source = path.read_bytes()
    except OSError:
        return []
    if len(source) > MAX_REPO_MAP_FILE_BYTES:
        return []

    parser, query = spec
    try:
        tree = parser.parse(source)
        matches = QueryCursor(query).matches(tree.root_node)
    except Exception:
        # A grammar/query mismatch or malformed source degrades this
        # ONE file to "no tags" -- never the whole repo map.
        return []

    tags: List[_Tag] = []
    seen: set = set()

    for _pattern_index, captures in matches:
        name_nodes = captures.get("name")
        if not name_nodes:
            continue
        name = name_nodes[0].text.decode("utf-8", errors="replace")

        for capture_name, nodes in captures.items():
            if capture_name in ("name", "doc"):
                continue
            kind, _, _category = capture_name.partition(".")
            if kind not in ("definition", "reference"):
                continue

            node = nodes[0]
            key = (kind, name, node.start_point[0], node.end_point[0])
            if key in seen:
                continue
            seen.add(key)

            signature = node.text.split(b"\n", 1)[0].decode("utf-8", errors="replace").strip()
            tags.append(_Tag(
                kind="def" if kind == "definition" else "ref",
                name=name,
                line=node.start_point[0],
                signature=signature,
            ))

    return tags


def _pagerank(nodes: List[str], edges: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Plain weighted PageRank via power iteration -- no personalization
    (no notion of "files already in context" at index-build time), and
    no networkx dependency: these graphs are small enough (bounded by
    file count) that a ~30-line power iteration is simpler than adding
    a dependency for it."""
    if not nodes:
        return {}

    n = len(nodes)
    rank = {node: 1.0 / n for node in nodes}
    out_weight = {node: sum(edges.get(node, {}).values()) for node in nodes}

    for _ in range(PAGERANK_ITERATIONS):
        new_rank = {node: (1.0 - PAGERANK_DAMPING) / n for node in nodes}
        for node in nodes:
            total_out = out_weight[node]
            if total_out <= 0:
                continue
            share = PAGERANK_DAMPING * rank[node] / total_out
            for target, weight in edges[node].items():
                new_rank[target] = new_rank.get(target, 0.0) + share * weight
        rank = new_rank

    return rank


def _build_repo_map(root: Path, file_list: List[str]) -> str:
    if not _TREE_SITTER_AVAILABLE:
        return ""

    specs = _get_language_specs()
    candidates = [
        relative for relative in file_list
        if LANGUAGE_BY_EXTENSION.get(Path(relative).suffix) in specs
    ][:MAX_REPO_MAP_FILES]
    if not candidates:
        return ""

    defs_by_file: Dict[str, List[_Tag]] = {}
    refs_by_file: Dict[str, List[_Tag]] = {}

    for relative in candidates:
        language_name = LANGUAGE_BY_EXTENSION[Path(relative).suffix]
        tags = _extract_tags(root / relative, language_name)
        if not tags:
            continue
        defs_by_file[relative] = [t for t in tags if t.kind == "def"]
        refs_by_file[relative] = [t for t in tags if t.kind == "ref"]

    if not defs_by_file:
        return ""

    definers_by_name: Dict[str, List[str]] = {}
    for file_path, defs in defs_by_file.items():
        for tag in defs:
            definers_by_name.setdefault(tag.name, []).append(file_path)

    edges: Dict[str, Dict[str, float]] = {f: {} for f in defs_by_file}
    ref_counts_by_def: Dict[Tuple[str, str], int] = {}

    for referencing_file, refs in refs_by_file.items():
        edges.setdefault(referencing_file, {})
        for tag in refs:
            definer_files = definers_by_name.get(tag.name)
            if not definer_files:
                continue
            # A name defined in many files is generic/ambiguous (e.g.
            # "get", "id") and says less about any ONE of them than a
            # name defined exactly once -- spread its weight across
            # all its definers instead of over-crediting every one.
            weight = 1.0 / len(definer_files)
            for definer_file in definer_files:
                if definer_file == referencing_file:
                    continue
                edges[referencing_file][definer_file] = (
                    edges[referencing_file].get(definer_file, 0.0) + weight
                )
                ref_counts_by_def[(definer_file, tag.name)] = (
                    ref_counts_by_def.get((definer_file, tag.name), 0) + 1
                )

    ranks = _pagerank(list(defs_by_file.keys()), edges)
    ranked_files = sorted(defs_by_file.keys(), key=lambda f: ranks.get(f, 0.0), reverse=True)

    blocks: List[str] = []
    total_chars = 0

    for file_path in ranked_files:
        top_defs = sorted(
            defs_by_file[file_path],
            key=lambda t: ref_counts_by_def.get((file_path, t.name), 0),
            reverse=True,
        )[:TOP_DEFS_PER_FILE]
        if not top_defs:
            continue
        top_defs.sort(key=lambda t: t.line)  # display in source order, not rank order

        block_lines = [f"{file_path}:"] + [
            f"    {tag.line + 1}: {tag.signature}" for tag in top_defs
        ]
        block_text = "\n".join(block_lines)

        if blocks and total_chars + len(block_text) > MAX_REPO_MAP_CHARS:
            break
        blocks.append(block_text)
        total_chars += len(block_text)

    return "\n\n".join(blocks)


def build_repo_index(root: Path) -> RepoIndex:
    root = Path(root).resolve()

    file_list: List[str] = []
    excluded: List[str] = []
    language_counts: Dict[str, int] = {}
    ecosystem_hints: List[str] = []
    manifest_files: Dict[str, str] = {}

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        relative = path.relative_to(root)
        relative_str = relative.as_posix()

        if any(part in EXCLUDED_DIR_NAMES for part in relative.parts[:-1]):
            excluded.append(relative_str)
            continue

        if relative.name.startswith(".") and relative.name != ".gitignore":
            excluded.append(relative_str)
            continue

        if path.suffix in EXCLUDED_FILE_SUFFIXES:
            excluded.append(relative_str)
            continue

        file_list.append(relative_str)

        language = LANGUAGE_BY_EXTENSION.get(path.suffix)
        if language:
            language_counts[language] = language_counts.get(language, 0) + 1

        if relative.name in MANIFEST_HINTS and len(relative.parts) <= 2:
            language, tool = MANIFEST_HINTS[relative.name]
            ecosystem_hints.append(f"{relative_str} -> {language} ({tool})")

            try:
                if path.stat().st_size <= MAX_MANIFEST_BYTES:
                    manifest_files[relative_str] = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                pass

    primary_language = (
        max(language_counts, key=language_counts.get) if language_counts else None
    )

    return RepoIndex(
        root=root,
        file_list=file_list,
        excluded=excluded,
        primary_language=primary_language,
        ecosystem_hints=sorted(set(ecosystem_hints)),
        manifest_files=manifest_files,
        tree_text=_render_tree(file_list),
        repo_map_text=_build_repo_map(root, file_list),
    )


def _render_tree(file_list: List[str]) -> str:
    lines = file_list[:MAX_TREE_LINES]
    text = "\n".join(lines)
    if len(file_list) > MAX_TREE_LINES:
        text += f"\n... ({len(file_list) - MAX_TREE_LINES} more files not shown)"
    return text or "(empty repository)"


def repo_overview_text(index: RepoIndex) -> str:
    manifest_section = "\n\n".join(
        f"### {path}\n```\n{content}\n```" for path, content in index.manifest_files.items()
    )

    sections = [
        f"Primary language: {index.primary_language or 'unknown'}",
        f"Ecosystem hints: {', '.join(index.ecosystem_hints) or 'none detected'}",
        f"Total files: {len(index.file_list)}",
        "",
        "File tree:",
        index.tree_text,
        "",
        "Key manifest files:" if manifest_section else "",
        manifest_section,
    ]

    if index.repo_map_text:
        sections += [
            "",
            "Repo map (most-referenced files and their most-referenced "
            "definitions, ranked by how much of the codebase actually "
            "calls/uses them -- not exhaustive, a starting point):",
            index.repo_map_text,
        ]

    return "\n".join(sections).strip()
