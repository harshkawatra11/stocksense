"""
Job graph (docs/11-agent-harness.md). A minimal DAG, not a workflow
framework: nodes are plain callables with declared dependencies, and the
only structural guarantee this module provides is a valid topological
order plus cycle detection. Execution, resumption, and idempotency live
in runner.py — this module is deliberately just the graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Node:
    name: str
    fn: Callable[..., dict]
    depends_on: tuple[str, ...] = field(default_factory=tuple)
    idempotency_key: str | None = None  # if set, runner skips re-execution once succeeded for this key


class CycleError(ValueError):
    pass


class Graph:
    def __init__(self, nodes: list[Node]):
        names = [n.name for n in nodes]
        if len(names) != len(set(names)):
            dupes = {n for n in names if names.count(n) > 1}
            raise ValueError(f"duplicate node names: {dupes}")
        self._nodes: dict[str, Node] = {n.name: n for n in nodes}
        for n in nodes:
            for dep in n.depends_on:
                if dep not in self._nodes:
                    raise ValueError(f"node '{n.name}' depends on unknown node '{dep}'")

    def __getitem__(self, name: str) -> Node:
        return self._nodes[name]

    def __contains__(self, name: str) -> bool:
        return name in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def nodes(self) -> list[Node]:
        return list(self._nodes.values())

    def topological_order(self) -> list[str]:
        """Kahn's algorithm. Raises CycleError if the graph is not a DAG,
        rather than silently truncating — a cycle is a build-time bug in
        the graph definition, not a runtime condition to route around."""
        in_degree = {name: 0 for name in self._nodes}
        for n in self._nodes.values():
            for dep in n.depends_on:
                in_degree[n.name] += 1

        # build reverse adjacency: dep -> [dependents]
        dependents: dict[str, list[str]] = {name: [] for name in self._nodes}
        for n in self._nodes.values():
            for dep in n.depends_on:
                dependents[dep].append(n.name)

        queue = sorted([name for name, deg in in_degree.items() if deg == 0])
        order: list[str] = []
        while queue:
            queue.sort()
            current = queue.pop(0)
            order.append(current)
            for dependent in dependents[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(order) != len(self._nodes):
            remaining = set(self._nodes) - set(order)
            raise CycleError(f"cycle detected among nodes: {remaining}")
        return order
