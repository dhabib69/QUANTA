---
name: QUANTA Working Feedback
description: How Habib wants Claude to behave when working on QUANTA
type: feedback
---

Keep responses concise and direct — no preamble, no trailing summaries of what was just done unless asked.

**Why:** Habib explicitly said "use tokens as efficiently as possible". He can read the diffs.

**How to apply:** Lead with action. Tables over prose for status updates. Skip "I'll now..." or "Let me..." intros.

---

When he says "continue", resume exactly where left off without re-summarizing what was already done.

**Why:** Context is preserved in the conversation. Repetition wastes tokens.

**How to apply:** Just start the next task immediately.

---

When he says "done?" or "confirm", give a tight summary table (fix | file | status checkmark).

**Why:** He wants confirmation, not re-explanation.

**How to apply:** One table, done.

---

Always read the file before proposing changes.

**Why:** QUANTA has complex interdependencies. Line numbers matter. Assumptions about code structure are frequently wrong.

**How to apply:** Read → understand → edit. Never propose changes to code not yet read.

---

Don't add emoji unless they're already in the surrounding code context.

**Why:** Habib's own code uses emoji heavily but Claude's explanatory text should be clean.

**How to apply:** Match the style of the file being edited, but keep explanatory text emoji-free.
