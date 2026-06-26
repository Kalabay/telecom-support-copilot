"use client";

import { useState } from "react";
import {
  BookOpen,
  Check,
  ChevronLeft,
  ChevronRight,
  Copy,
  CornerDownLeft,
  Sparkles,
  X,
} from "lucide-react";

import type { KBSource, Suggestion } from "@/lib/types";
import type { SuggestionSet } from "@/lib/useChats";
import { cn } from "@/lib/utils";

interface Props {
  sets: SuggestionSet[];
  onSend: (text: string, sources: KBSource[], setId: number) => void;
  onOpenSource: (docId: string) => void;
  onDismiss: (setId: number) => void;
  demo?: boolean;
}

function SourceChips({
  s,
  onOpenSource,
}: {
  s: Suggestion;
  onOpenSource: (docId: string) => void;
}) {
  if (!s.sources.length) return null;
  return (
    <div className="flex flex-wrap gap-1.5 mt-2">
      {s.sources.map((src) => (
        <button
          key={src.doc_id}
          type="button"
          onClick={() => onOpenSource(src.doc_id)}
          className="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-secondary hover:bg-primary/20 text-[11px] text-muted-foreground hover:text-foreground transition max-w-full"
          title="Открыть статью базы знаний"
        >
          <BookOpen className="w-3 h-3 shrink-0 text-primary/80" />
          <span className="truncate max-w-[150px]">{src.title}</span>
          <span className="tabular-nums opacity-60">{(src.score * 100).toFixed(0)}%</span>
        </button>
      ))}
    </div>
  );
}

function SetCard({
  set,
  isNewest,
  demo,
  onSend,
  onOpenSource,
  onDismiss,
}: {
  set: SuggestionSet;
  isNewest: boolean;
  demo?: boolean;
  onSend: (text: string, sources: KBSource[], setId: number) => void;
  onOpenSource: (docId: string) => void;
  onDismiss: (setId: number) => void;
}) {
  const [idx, setIdx] = useState(0);
  const [copied, setCopied] = useState(false);
  const total = set.suggestions.length;
  const cur = set.suggestions[Math.min(idx, total - 1)];
  if (!cur) return null;

  const copy = () => {
    if (typeof navigator !== "undefined" && navigator.clipboard) {
      navigator.clipboard.writeText(cur.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    }
  };

  return (
    <div
      className={cn(
        "rounded-2xl border p-3 transition",
        isNewest ? "border-primary/40 bg-primary/5" : "border-border bg-card/60",
        demo && isNewest && "border-emerald-500 bg-emerald-500/5",
      )}
    >
      <div className="flex items-center justify-between gap-2 mb-1.5">
        <span className="text-[11px] text-muted-foreground truncate min-w-0">
          {isNewest ? (
            <span className="font-semibold uppercase tracking-wider text-primary">
              Рекомендованный ответ
            </span>
          ) : (
            <>ответ на: <span className="opacity-70">«{set.replyTo}»</span></>
          )}
        </span>
        <button
          type="button"
          onClick={() => onDismiss(set.id)}
          className="w-5 h-5 rounded-full flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-secondary transition shrink-0"
          title="Убрать подсказку"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      <p className={cn("leading-relaxed text-foreground", isNewest ? "text-[15px]" : "text-sm")}>
        {cur.text}
      </p>
      <SourceChips s={cur} onOpenSource={onOpenSource} />

      <div className="flex items-center gap-2 mt-3">
        <button
          type="button"
          onClick={() => onSend(cur.text, cur.sources, set.id)}
          className={cn(
            "inline-flex items-center gap-2 rounded-xl font-semibold transition",
            demo
              ? "bg-emerald-500 text-white"
              : "bg-primary text-primary-foreground hover:brightness-105",
            isNewest ? "px-4 py-2 text-sm" : "px-3 py-1.5 text-xs",
          )}
        >
          {demo ? <Check className="w-4 h-4" /> : <CornerDownLeft className="w-4 h-4" />}
          {demo ? "Выбрано" : "Выбрать подсказку"}
        </button>

        {total > 1 && (
          <div className="flex items-center gap-0.5 rounded-lg bg-secondary px-1 py-1">
            <button
              type="button"
              onClick={() => setIdx((i) => Math.max(0, i - 1))}
              disabled={idx === 0}
              className="w-6 h-6 rounded-md flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-card disabled:opacity-30 disabled:hover:bg-transparent transition"
              title="Предыдущий вариант"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="text-[11px] tabular-nums text-muted-foreground px-1 select-none">
              {idx + 1}/{total}
            </span>
            <button
              type="button"
              onClick={() => setIdx((i) => Math.min(total - 1, i + 1))}
              disabled={idx >= total - 1}
              className="w-6 h-6 rounded-md flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-card disabled:opacity-30 disabled:hover:bg-transparent transition"
              title="Следующий вариант"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        )}

        <button
          type="button"
          onClick={copy}
          className="ml-auto inline-flex items-center gap-1.5 px-2.5 py-2 rounded-xl text-xs text-muted-foreground hover:text-foreground hover:bg-secondary transition"
          title="Скопировать"
        >
          {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
        </button>
      </div>
    </div>
  );
}

export function SuggestionsPanel({ sets, onSend, onOpenSource, onDismiss, demo }: Props) {
  if (sets.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-border bg-card/50 px-4 py-3 flex items-center gap-2 text-sm text-muted-foreground">
        <Sparkles className="w-4 h-4 text-primary/70 shrink-0" />
        Подсказка появится автоматически после реплики клиента
      </div>
    );
  }

  return (
    <div className="max-h-[44vh] overflow-y-auto scrollbar-thin pr-1 space-y-2">
      {sets.map((set, i) => (
        <SetCard
          key={set.id}
          set={set}
          isNewest={i === 0}
          demo={demo}
          onSend={onSend}
          onOpenSource={onOpenSource}
          onDismiss={onDismiss}
        />
      ))}
    </div>
  );
}
