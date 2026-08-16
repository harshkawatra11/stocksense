"""
Tool package. Importing this package is what populates `registry` --
each tool module registers its tools as an import-time side effect, so
this __init__ imports every tool module explicitly rather than relying
on whoever imports the package to also import the right submodules.

A caller that does `from stocksense.foreman.tools import registry` and
gets an EMPTY registry (missing read_file, write_patch, etc.) means a
new tool module was added without being imported here — that failure
mode is exactly why this file exists instead of leaving registration to
whatever happens to import first.
"""

from __future__ import annotations

from stocksense.foreman.tools.base import ActionClass, Tool, ToolResult, registry

# Import for registration side effects — order doesn't matter, but
# completeness does.
from stocksense.foreman.tools import code as _code  # noqa: F401
from stocksense.foreman.tools import git_tools as _git_tools  # noqa: F401

__all__ = ["ActionClass", "Tool", "ToolResult", "registry"]
