"""Phase J1.2: the safety-critical test. This is layer 2 of 3 enforcing
that no order-placement code path can exist in this codebase -- an AST
walk over every .py file under src/, not a substring search.

AST, not substring, deliberately: foreman/adversary.py's own history
recorded exactly this bug once already (checking for the literal
substring "pytest.raises" inside ast.dump() output, which renders
attribute access as nested Attribute(...) nodes, not dotted text, so it
never matched real usages). The same failure mode here would mean this
test could pass while a real order-placement call sits in the codebase
undetected -- so every identifier is inspected as a real ast.Name or
ast.Attribute node, matched via ast.unparse, never string-matched
against source text or ast.dump() output."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from stocksense.brokers.angel_readonly import ALLOWED_METHODS, OrderPlacementForbidden, ReadOnlyBrokerClient

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "stocksense"

FORBIDDEN_IDENTIFIERS = frozenset({
    "placeOrder", "placeOrderFullResponse", "modifyOrder", "cancelOrder",
    "convertPosition", "gttCreateRule", "gttModifyRule", "gttCancelRule",
    "generateTPIN", "verifyDis",
})


def _referenced_identifiers(tree: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def test_no_order_placement_identifier_anywhere_in_src() -> None:
    hits = {}
    for path in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found = _referenced_identifiers(tree) & FORBIDDEN_IDENTIFIERS
        if found:
            hits[str(path.relative_to(SRC_ROOT))] = found
    assert not hits, f"order-placement identifiers found: {hits}"


def test_allowed_methods_excludes_every_forbidden_identifier() -> None:
    assert ALLOWED_METHODS.isdisjoint(FORBIDDEN_IDENTIFIERS)


def test_readonly_client_passes_through_allowed_method() -> None:
    class _Fake:
        def holding(self):
            return {"status": True, "data": []}

    client = ReadOnlyBrokerClient(_Fake())
    assert client.holding() == {"status": True, "data": []}


@pytest.mark.parametrize("method", sorted(FORBIDDEN_IDENTIFIERS))
def test_readonly_client_rejects_every_forbidden_method(method) -> None:
    class _Fake:
        def __getattr__(self, name):
            return lambda *a, **k: "SHOULD NEVER BE CALLED"

    client = ReadOnlyBrokerClient(_Fake())
    with pytest.raises(OrderPlacementForbidden):
        getattr(client, method)


def test_readonly_client_has_no_bypass_parameter() -> None:
    import inspect

    sig = inspect.signature(ReadOnlyBrokerClient.__init__)
    assert list(sig.parameters) == ["self", "client"]
