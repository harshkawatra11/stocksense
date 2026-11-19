from __future__ import annotations

import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GATE = REPO / "config" / "gate.toml"

# SHA-256 of config/gate.toml with CRLF normalized to LF. The gate is frozen: if this test fails, someone
# edited the pass/fail bars. Thresholds are never adjusted after seeing a result (PLAN rule); revert the edit.
GATE_SHA256 = "e26a13769b707cba98b409f19880f905d012abaa5d0205d2ab84f062492f270e"


def gate_hash(path: Path) -> str:
    """SHA-256 hex of the file's bytes after normalizing \\r\\n to \\n."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_gate_toml_is_frozen():
    assert gate_hash(GATE) == GATE_SHA256


def test_hash_checker_detects_edit_and_ignores_line_endings(tmp_path):
    f = tmp_path / "gate.toml"
    f.write_bytes(GATE.read_bytes().replace(b"\r\n", b"\n"))
    assert gate_hash(f) == GATE_SHA256
    f.write_bytes(GATE.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert gate_hash(f) == GATE_SHA256  # CRLF checkout hashes the same
    f.write_bytes(
        f.read_bytes().replace(b"min_dsr                 = 0.95", b"min_dsr                 = 0.90")
    )
    assert gate_hash(f) != GATE_SHA256  # a loosened threshold is caught


# Gate v2 (G2-B1, PLAN-G2 4.3). Same thresholds as v1, plus the two scope keys. Frozen the day it was created,
# after the G1 results were seen (recorded in DECISIONS); never edited afterwards.
GATE_V2 = REPO / "config" / "gate_v2.toml"
GATE_V2_SHA256 = "eb01912dd5a9b1b90dc9b960aa5a5f7a0bc9b07784a15d6b0bc9f1417efa2cac"


def test_gate_v2_toml_is_frozen():
    assert gate_hash(GATE_V2) == GATE_V2_SHA256


def test_gate_v1_hash_unchanged_by_v2():
    assert gate_hash(GATE) == GATE_SHA256
