"""
Hash-chained, append-only ledger of every human-in-the-loop gate
decision (Approve / Suggest changes / Override / Reject), across every run.

Deterministic, no LLM call. Each record's own hash covers the
canonical JSON of that record's other fields, including its
`prev_hash` -- the previous record's hash -- so the file forms a
chain: editing a record's content without recomputing its hash is
caught at that record (the stored hash no longer matches); editing it
AND recomputing its own hash to hide the edit is instead caught at the
NEXT record (its prev_hash no longer matches the edited record's new
hash). Either way, tampering with any record is detectable without
also forging every record after it. verify() below recomputes the
whole chain and reports exactly where it first breaks. This is a
durable audit trail on top of what run_log.py/history.json already
capture (those record what a run did; this records whether that
record has since been tampered with).

Based on the *idea* of bashebr/ai-native-sdlc's gate_ledger.py, not its
implementation (theirs has no verify(), and nothing reads it back).
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

import config

GENESIS_HASH = "0" * 64


def _canonical(entry: Dict[str, Any]) -> str:
    return json.dumps(entry, sort_keys=True, default=str)


def _hash_entry(entry: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(entry).encode("utf-8")).hexdigest()


def load_all() -> List[Dict[str, Any]]:
    if not config.APPROVAL_LEDGER_PATH.is_file():
        return []

    entries = []
    for line in config.APPROVAL_LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _last_hash() -> str:
    entries = load_all()
    return entries[-1]["hash"] if entries else GENESIS_HASH


def record(
    *,
    run_id: str,
    repo_key: str,
    stage: str,
    decision: str,
    note: str = "",
    reason: str = "",
) -> Dict[str, Any]:
    """Appends one gate decision. decision is one of APPROVE,
    SUGGEST_CHANGES, OVERRIDE, REJECT. Returns the written entry, hash included."""
    entry = {
        "run_id": run_id,
        "repo_key": repo_key,
        "stage": stage,
        "decision": decision,
        "note": note,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prev_hash": _last_hash(),
    }
    entry["hash"] = _hash_entry(entry)

    config.APPROVAL_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.APPROVAL_LEDGER_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")

    return entry


def verify() -> Dict[str, Any]:
    """Re-derives every record's hash and its link to the previous
    record's hash. Returns {"valid", "record_count", "broken_at"} --
    0-based indices where either the stored hash doesn't match what's
    actually in the record, or prev_hash doesn't match the preceding
    record's real hash. An edited record surfaces either at its own
    index (if its stored hash was left stale) or at the next index (if
    its hash was recomputed to match the edit)."""
    entries = load_all()
    broken: List[int] = []
    expected_prev = GENESIS_HASH

    for i, entry in enumerate(entries):
        stored_hash = entry.get("hash")
        recomputed = _hash_entry({k: v for k, v in entry.items() if k != "hash"})
        if entry.get("prev_hash") != expected_prev or stored_hash != recomputed:
            broken.append(i)
        expected_prev = stored_hash if stored_hash is not None else expected_prev

    return {"valid": not broken, "record_count": len(entries), "broken_at": broken}
