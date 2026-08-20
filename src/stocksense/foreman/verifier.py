"""
The verifier (docs/19-foreman.md): a sequential chain of gates a goal's
result must clear before it is eligible for auto-merge. Any single gate
failing stops the chain -- later, more expensive gates (remote CI) never
run on a result that already failed something cheap and local.

Order matters and is deliberate: protected-path check first (free,
instant, and the one that must never be skipped), then local tests
(cheap), then leakage/determinism (still local but slower), then remote
CI (the independent check -- see foreman/tools/git_tools.check_ci), then
-- for research goals only -- the pre-registered gate itself, evaluated
without any threshold adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from stocksense.foreman.policy import check_patch
from stocksense.foreman.tools.code import run_tests
from stocksense.foreman.tools.git_tools import check_ci


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class VerificationResult:
    gates: list[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.gates)

    @property
    def first_failure(self) -> GateResult | None:
        for g in self.gates:
            if not g.passed:
                return g
        return None


def verify(
    changed_paths: list[str],
    branch: str | None = None,
    require_remote_ci: bool = True,
    is_research_goal: bool = False,
) -> VerificationResult:
    """Runs the gate chain, stopping at the first failure. `branch` is
    required (and require_remote_ci must be True) for anything that will
    actually be merged; verify() can be called with require_remote_ci=False
    for a quick local-only check during iteration, but such a result must
    never be treated as merge-eligible -- see executor.py's usage."""
    gates: list[GateResult] = []

    protected = check_patch(changed_paths)
    gates.append(GateResult("protected_paths", passed=len(protected) == 0, detail=f"blocked: {protected}" if protected else "clear"))
    if protected:
        return VerificationResult(gates=gates)

    test_result = run_tests("tests/unit")
    gates.append(GateResult("local_tests", passed=test_result.ok, detail=test_result.output[-500:]))
    if not test_result.ok:
        return VerificationResult(gates=gates)

    leakage_result = run_tests("tests/unit/test_leakage.py")
    gates.append(GateResult("leakage_suite", passed=leakage_result.ok, detail=leakage_result.output[-500:]))
    if not leakage_result.ok:
        return VerificationResult(gates=gates)

    determinism_result = run_tests("tests/unit/test_determinism.py")
    gates.append(GateResult("determinism_suite", passed=determinism_result.ok, detail=determinism_result.output[-500:]))
    if not determinism_result.ok:
        return VerificationResult(gates=gates)

    if require_remote_ci:
        if not branch:
            gates.append(GateResult("remote_ci", passed=False, detail="no branch provided; cannot check remote CI"))
            return VerificationResult(gates=gates)
        ci_result = check_ci(branch)
        gates.append(GateResult("remote_ci", passed=ci_result.ok, detail=ci_result.output))
        if not ci_result.ok:
            return VerificationResult(gates=gates)

    if is_research_goal:
        # AUDIT FIX: this gate was hardcoded passed=True, delegating to
        # foreman/tools/research.py -- which does not exist. A gate that
        # always passes for a check nothing implements is not a gate, it
        # is a rubber stamp; is_research_goal defaults to False and no
        # caller currently sets it True, so this was latent rather than
        # actively firing, but it must fail closed rather than open the
        # moment something does set it. Once foreman/tools/research.py
        # lands (Phase D3/E4's pre-registered-gate wiring) this becomes a
        # real check against that module's own evaluate_gate() result,
        # not a permanent False.
        gates.append(GateResult(
            "pre_registered_gate", passed=False,
            detail="research-goal gate check has no implementation yet (foreman/tools/research.py "
                   "does not exist) -- failing closed rather than rubber-stamping a check that isn't real",
        ))

    return VerificationResult(gates=gates)
