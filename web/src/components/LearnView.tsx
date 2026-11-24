"use client";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Markdown } from "@/components/Markdown";
import type { LearnNote } from "@/lib/types";

export interface LearnNoteBody extends LearnNote {
  body: string;
}

/** `/learn/` lists the notes; `/learn/?topic=<slug>` shows one. */
export function LearnView({ notes }: { notes: LearnNoteBody[] }) {
  const topic = useSearchParams().get("topic");
  const note = topic ? notes.find((n) => n.slug === topic) : undefined;
  if (topic && !note) {
    return (
      <main className="page-plain">
        <p>No learning note called “{topic}”.</p>
        <p>
          <Link href="/learn/">All notes</Link>
        </p>
      </main>
    );
  }
  if (note) {
    return (
      <main className="page-plain">
        <p>
          <Link href="/learn/">← All notes</Link>
        </p>
        <Markdown source={note.body} />
      </main>
    );
  }
  return (
    <main className="page-plain">
      <h1>Learn</h1>
      <p>What each tool shows and what NSE data allows.</p>
      <ol className="learn-list">
        {notes.map((n) => (
          <li key={n.slug}>
            <Link href={`/learn/?topic=${n.slug}`}>{n.title}</Link>
          </li>
        ))}
      </ol>
    </main>
  );
}
