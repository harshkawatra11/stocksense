---
name: claude-report-writing
description: How to narrate from a computed fact sheet without inventing numbers. Use whenever composing prose (a report, a summary, an explanation) that draws on financial figures StockSense has already computed.
allowed-tools: []
---

# Claude Report Writing

This is the safety-critical skill in this codebase. It exists because the
whole system rests on one rule: **Python computes every number, Claude
writes every narrative.** A Claude call is handed a fact sheet (JSON) and
asked to explain, prioritize, and prescribe — never to calculate a
statistic, and its output is never parsed back into a number that feeds a
decision.

Why this matters here specifically: StockSense reports on a user's real
money — their P&L, their tax liability, their behavioral patterns. A
hallucinated figure in that context isn't a stylistic slip, it's a false
financial statement.

## The rule, operationally

1. **Every number in your response must trace to the fact sheet you were
   given.** Not "close to" a number in the fact sheet — the literal value,
   or a value trivially derivable by arithmetic already shown to you (a
   sum, a percentage of a total that's present). If you don't have the
   number, say the finding qualitatively or say the data isn't available.
   Do not estimate, round in a way that misrepresents precision, or fill a
   gap with a plausible-sounding figure.

2. **You will be checked.** `agent/claude_cli.py` runs a numeric tripwire
   against every response: it extracts numeric tokens and flags any that
   don't trace back to the facts block. This isn't a courtesy — treat it
   as adversarial. Assume your output will be audited.

3. **Distinguish computed fact from your own framing.** "Your cost drag
   was 34.5%" (a fact) is different from "that's unusually high for an
   active trader" (your framing, a judgment call you're allowed to make —
   but say it as framing, not as if 34.5% being "unusually high" is itself
   a number in the fact sheet).

4. **Counterfactuals and predictions are explicitly not facts about what
   happened.** When narrating a counterfactual ("what if you'd capped
   position size"), say plainly that it is a replay of historical fills
   under a changed rule, not a prediction and not certain to reproduce if
   repeated (changed behavior changes market impact and subsequent
   decisions too).

5. **Silence is a valid output.** If a fact sheet has `insufficient_sample:
   true` or an empty findings list, say that plainly rather than
   manufacturing a finding to seem useful. A thin tradebook producing "no
   significant finding" is the correct, honest report.

## Structure that tends to work

- **Verdict first.** One sentence, using an exact number, naming the
  single most important thing.
- **Ranked by severity**, not by the order data happens to appear.
- **Each claim paired with its number**, inline, not in a separate table
  the reader has to cross-reference.
- **Remedies are specific and quantified** where the fact sheet supports
  it ("cap position size at ₹X" beats "consider smaller positions" if X is
  actually in the facts).
- **What's working, not just what's broken** — genuinely, using real
  `strengths`/`ok`-severity data if present, not as a softening ritual.

## Anti-patterns, explicitly

- Inventing a percentage that "sounds about right" for the finding you're
  describing.
- Restating a number with different precision than given (a fact sheet
  value of `0.345` becoming "about a third" is fine as a paraphrase if the
  exact figure is *also* stated; it is not fine as a replacement for it).
- Treating a `null`/`None` field as zero.
- Comparing the user's numbers to an external benchmark or "typical
  trader" statistic that wasn't given in the fact sheet — you likely don't
  have a reliable one, and inventing one is exactly what this rule exists
  to prevent.
