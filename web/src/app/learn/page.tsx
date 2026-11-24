import { readFileSync } from "node:fs";
import path from "node:path";
import { Suspense } from "react";
import { LearnView, type LearnNoteBody } from "@/components/LearnView";
import type { LearnNote } from "@/lib/types";

// Reads content/learn/ (written from docs/learn/ by scripts/copy-learn.mjs) at build time.
function loadNotes(): LearnNoteBody[] {
  const dir = path.join(process.cwd(), "content", "learn");
  const index = JSON.parse(readFileSync(path.join(dir, "index.json"), "utf8")) as LearnNote[];
  return index.map((n) => ({ ...n, body: readFileSync(path.join(dir, `${n.slug}.md`), "utf8") }));
}

export default function LearnPage() {
  return (
    <Suspense fallback={<main className="page-plain">Loading…</main>}>
      <LearnView notes={loadNotes()} />
    </Suspense>
  );
}
