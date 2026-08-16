"""
Tool registry (docs/19-foreman.md). Every capability the Foreman can
invoke is registered here as a typed callable with a declared side-effect
class, so the planner can reason about risk before choosing a tool and
the ledger records what CLASS of action happened even if the specific
tool is new.

The registry is deliberately a closed set: the planner selects from
`registry.names()`, and an emitted tool name that isn't registered fails
fast in the executor rather than being invented at call time. This is
what stops "hallucinated tool" from becoming "arbitrary code execution."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class ActionClass(str, Enum):
    READ = "read"        # no side effects
    WRITE = "write"       # modifies files in the working tree
    EXEC = "exec"          # runs a subprocess (tests, lint, sweeps)
    NETWORK = "network"     # touches an external service (git push, web fetch, data API)


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str
    data: dict = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class Tool:
    name: str
    action_class: ActionClass
    fn: Callable[..., ToolResult]
    description: str
    # paths this specific invocation would touch, if known ahead of the
    # call -- WRITE tools must be able to report this so the executor can
    # check it against foreman.policy BEFORE calling fn.
    touches_paths: Callable[..., list[str]] | None = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: '{name}' (not in registry — planner may have hallucinated it)")
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def describe_all(self) -> str:
        return "\n".join(f"- {t.name} [{t.action_class.value}]: {t.description}" for t in sorted(self._tools.values(), key=lambda t: t.name))


registry = ToolRegistry()
