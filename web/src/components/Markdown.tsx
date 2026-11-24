"use client";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// Links between notes ("06-vwap.md" or "./06-vwap.md#x") become /learn/?topic=<slug>; other links open as written.
export function noteHref(href: string | undefined): string | undefined {
  if (!href) return href;
  const m = /^(?:\.\/)?(?:\.\.\/learn\/)?([\w-]+)\.md(#.*)?$/.exec(href);
  return m ? `/learn/?topic=${m[1]}${m[2] ?? ""}` : href;
}

export function Markdown({ source }: { source: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a({ href, children }) {
          const to = noteHref(href);
          return to && to.startsWith("/learn/") ? <Link href={to}>{children}</Link> : <a href={to}>{children}</a>;
        },
      }}
    >
      {source}
    </ReactMarkdown>
  );
}
