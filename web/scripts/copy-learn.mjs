// Copies ../docs/learn/*.md to content/learn/ and writes content/learn/index.json ([{slug, title}]).
import { cpSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const src = path.resolve(here, "..", "..", "docs", "learn");
const dst = path.resolve(here, "..", "content", "learn");

rmSync(dst, { recursive: true, force: true });
mkdirSync(dst, { recursive: true });
const index = [];
for (const f of readdirSync(src).filter((n) => n.endsWith(".md")).sort()) {
  cpSync(path.join(src, f), path.join(dst, f));
  const text = readFileSync(path.join(src, f), "utf8");
  const h = text.split(/\r?\n/).find((l) => l.startsWith("# "));
  const slug = f.replace(/\.md$/, "");
  index.push({ slug, title: h ? h.slice(2).trim() : slug });
}
writeFileSync(path.join(dst, "index.json"), JSON.stringify(index, null, 1), "utf8");
console.log(`copy-learn: ${index.length} notes`);
