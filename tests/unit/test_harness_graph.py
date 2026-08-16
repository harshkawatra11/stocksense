from __future__ import annotations

import pytest

from stocksense.harness.graph import CycleError, Graph, Node


def test_topological_order_respects_dependencies() -> None:
    g = Graph(
        [
            Node("a", fn=lambda ctx: {}),
            Node("b", fn=lambda ctx: {}, depends_on=("a",)),
            Node("c", fn=lambda ctx: {}, depends_on=("a", "b")),
        ]
    )
    order = g.topological_order()
    assert order.index("a") < order.index("b") < order.index("c")


def test_independent_nodes_included() -> None:
    g = Graph([Node("x", fn=lambda ctx: {}), Node("y", fn=lambda ctx: {})])
    order = g.topological_order()
    assert set(order) == {"x", "y"}


def test_cycle_detected() -> None:
    g = Graph(
        [
            Node("a", fn=lambda ctx: {}, depends_on=("b",)),
            Node("b", fn=lambda ctx: {}, depends_on=("a",)),
        ]
    )
    with pytest.raises(CycleError):
        g.topological_order()


def test_unknown_dependency_raises() -> None:
    with pytest.raises(ValueError):
        Graph([Node("a", fn=lambda ctx: {}, depends_on=("ghost",))])


def test_duplicate_node_names_raises() -> None:
    with pytest.raises(ValueError):
        Graph([Node("a", fn=lambda ctx: {}), Node("a", fn=lambda ctx: {})])
