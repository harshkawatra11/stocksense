"""
Phase J1.2: the enforcement layer. This wrapper is the reason it is safe
to hold a live, authenticated Angel One SmartAPI session in this
process at all -- confirmed live via the ALLOWED_METHODS list below,
enumerated directly against `dir(SmartApi.SmartConnect)` before writing
it, not guessed.

Three layers of enforcement exist for this, deliberately redundant --
one is not enough for something that can move real money:
1. THIS wrapper (runtime): any attribute not in ALLOWED_METHODS raises
   OrderPlacementForbidden, with no bypass parameter anywhere in this
   module.
2. tests/unit/test_angel_readonly.py's AST walk over all of src/,
   asserting no order-placement identifier (placeOrder, modifyOrder,
   cancelOrder, gttCreateRule, gttModifyRule, gttCancelRule, ...) is
   ever referenced anywhere in this codebase -- an AST walk, not a
   substring search, per the exact lesson foreman/adversary.py's own
   history recorded (a substring check against ast.dump() output
   silently never matched real attribute-access nodes).
3. foreman/policy.py's PROTECTED_PATTERNS -- this file and its test are
   both protected paths, so widening the allowlist is a human-reviewed
   decision, never a silent one.
"""

from __future__ import annotations

ALLOWED_METHODS = frozenset({
    "holding", "allholding", "position", "orderBook", "tradeBook",
    "rmsLimit", "getProfile",
})
"""Enumerated directly against dir(SmartApi.SmartConnect) before this
file was written. Every OTHER real method on that class -- placeOrder,
placeOrderFullResponse, modifyOrder, cancelOrder, convertPosition,
gttCreateRule, gttModifyRule, gttCancelRule, generateTPIN, verifyDis,
and every session/token-management method (generateSession,
terminateSession, setAccessToken, renewAccessToken, ...) -- is
deliberately absent. Session management stays in angel_session.py,
which owns the SmartConnect instance directly and is the only place
credentials are handled; this wrapper is what every OTHER module in
this codebase is handed instead of the raw client."""


class OrderPlacementForbidden(Exception):
    """Raised for any attribute access not in ALLOWED_METHODS. There is
    no bypass parameter on ReadOnlyBrokerClient -- if this ever fires in
    production, that is a bug in the calling code, not a permission to
    work around."""


class ReadOnlyBrokerClient:
    """Wraps a live SmartConnect instance. Attribute access for
    anything not in ALLOWED_METHODS raises immediately, before the
    underlying call is ever attempted -- this is a static allowlist
    check, not a runtime inspection of what the call WOULD have done."""

    def __init__(self, client) -> None:
        self._client = client

    def __getattr__(self, name: str):
        if name not in ALLOWED_METHODS:
            raise OrderPlacementForbidden(
                f"{name!r} is not on the read-only allowlist -- this client can never place, "
                f"modify, or cancel an order, or touch a GTT rule. Allowed: {sorted(ALLOWED_METHODS)}"
            )
        return getattr(self._client, name)

    def __repr__(self) -> str:
        return "ReadOnlyBrokerClient(<redacted>)"
