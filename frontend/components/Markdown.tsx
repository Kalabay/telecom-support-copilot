"use client";

import type { ReactNode } from "react";

function inline(text: string, keyBase: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).map((p, i) =>
    p.startsWith("**") && p.endsWith("**") ? (
      <strong key={`${keyBase}-${i}`} className="font-semibold text-foreground">
        {p.slice(2, -2)}
      </strong>
    ) : (
      <span key={`${keyBase}-${i}`}>{p}</span>
    ),
  );
}

export function Markdown({ text }: { text: string }) {
  const lines = text.split("\n");
  const blocks: ReactNode[] = [];
  let list: string[] = [];

  const flush = (k: number) => {
    if (list.length) {
      const items = list;
      blocks.push(
        <ul key={`ul-${k}`} className="list-disc pl-5 space-y-1 text-[14px] text-muted-foreground">
          {items.map((li, j) => (
            <li key={j}>{inline(li, `li-${k}-${j}`)}</li>
          ))}
        </ul>,
      );
      list = [];
    }
  };

  lines.forEach((ln, i) => {
    if (/^#\s/.test(ln)) {
      flush(i);
      blocks.push(
        <h1 key={i} className="text-lg font-bold text-foreground mt-1">
          {ln.replace(/^#\s+/, "")}
        </h1>,
      );
    } else if (/^##\s/.test(ln)) {
      flush(i);
      blocks.push(
        <h2 key={i} className="text-[15px] font-semibold text-primary mt-4">
          {ln.replace(/^##\s+/, "")}
        </h2>,
      );
    } else if (/^###\s/.test(ln)) {
      flush(i);
      blocks.push(
        <h3 key={i} className="text-sm font-semibold text-foreground mt-3">
          {ln.replace(/^###\s+/, "")}
        </h3>,
      );
    } else if (/^\s*[-*]\s/.test(ln)) {
      list.push(ln.replace(/^\s*[-*]\s+/, ""));
    } else if (ln.trim() === "") {
      flush(i);
    } else {
      flush(i);
      blocks.push(
        <p key={i} className="text-[14px] leading-relaxed text-foreground/90">
          {inline(ln, `p-${i}`)}
        </p>,
      );
    }
  });
  flush(lines.length);

  return <div className="space-y-2">{blocks}</div>;
}
