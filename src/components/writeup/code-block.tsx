import type { ReactNode } from "react";
import { CopyCodeButton } from "@/components/writeup/copy-code-button";

type Language = "python" | "bash";

const KEYWORDS: Record<Language, Set<string>> = {
  python: new Set(["def", "return", "for", "in", "if", "else", "elif", "and", "or", "not", "import", "from", "as", "lambda", "with", "True", "False", "None", "class", "while"]),
  bash: new Set(["export", "uv", "run", "python", "sync"]),
};

// One pass over the source: comments, strings, numbers, identifiers. Small on purpose;
// the excerpts are short and server-rendered, so no highlighting library is needed.
const TOKEN = /(#[^\n]*)|("""[\s\S]*?"""|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')|(\b\d+(?:\.\d+)?(?:e-?\d+)?\b|\.\d+\b)|(--?[a-zA-Z][\w-]*)|([A-Za-z_]\w*)/g;

function highlight(code: string, language: Language): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  let previousWord = "";
  for (const match of code.matchAll(TOKEN)) {
    const [text, comment, string, number, flag, word] = match;
    const index = match.index ?? 0;
    if (index > last) out.push(code.slice(last, index));
    let className: string | undefined;
    if (comment) className = "text-muted-foreground italic";
    else if (string) className = "text-[var(--code-string)]";
    else if (number) className = "text-[var(--code-number)]";
    else if (flag) className = language === "bash" ? "text-[var(--code-keyword)]" : undefined;
    else if (word && KEYWORDS[language].has(word)) className = "text-[var(--code-keyword)] font-medium";
    else if (word && previousWord === "def") className = "text-[var(--code-function)] font-medium";
    out.push(className ? <span key={index} className={className}>{text}</span> : text);
    if (word) previousWord = word;
    last = index + text.length;
  }
  if (last < code.length) out.push(code.slice(last));
  return out;
}

export function CodeBlock({ code, language = "python", file, caption }: { code: string; language?: Language; file: string; caption?: ReactNode }) {
  const source = code.replace(/^\n+|\s+$/g, "");
  return (
    <figure className="not-prose my-6 overflow-hidden rounded-xl border border-border bg-card shadow-sm [--code-function:#1f6f9a] [--code-keyword:var(--primary)] [--code-number:#a64f00] [--code-string:#8a4b00] dark:[--code-function:#7cc4e8] dark:[--code-number:#ffb66d] dark:[--code-string:#f0a35a]">
      <div className="flex items-center justify-between gap-3 border-b border-border bg-muted/60 px-4 py-2">
        <span className="truncate font-mono text-xs text-muted-foreground">{file}</span>
        <div className="flex items-center gap-2">
          <span className="rounded-md bg-background/70 px-1.5 py-0.5 font-mono text-[0.65rem] uppercase tracking-wide text-muted-foreground">{language}</span>
          <CopyCodeButton code={source} />
        </div>
      </div>
      <pre className="overflow-x-auto p-4 font-mono text-[0.8rem] leading-relaxed text-card-foreground"><code>{highlight(source, language)}</code></pre>
      {caption && <figcaption className="border-t border-border px-4 py-2.5 text-xs leading-relaxed text-muted-foreground">{caption}</figcaption>}
    </figure>
  );
}
