"use client";

import { useEffect, useState } from "react";
import { AlertTriangle, Play, Square } from "lucide-react";

import { cn, EMOTION_LABEL_RU, STAGE_LABEL_RU } from "@/lib/utils";
import type { EmotionState, PipelineStage } from "@/lib/types";

interface Props {
  connected: boolean;
  stage: PipelineStage;
  emotion: EmotionState | null;
  title: string;
  onRename: (title: string) => void;
  onStart: () => void;
  onStop: () => void;
  isActive: boolean;
}

const AVATAR_BG: Record<string, string> = {
  neutral: "bg-slate-400",
  angry: "bg-red-500",
  positive: "bg-emerald-500",
  sad: "bg-blue-500",
  other: "bg-violet-500",
};

export function HeaderBar({
  connected,
  stage,
  emotion,
  title,
  onRename,
  onStart,
  onStop,
  isActive,
}: Props) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(title);

  useEffect(() => {
    if (!editing) setDraft(title);
  }, [title, editing]);

  const commit = () => {
    onRename(draft);
    setEditing(false);
  };

  const avatarColor = emotion ? AVATAR_BG[emotion.label] : "bg-primary";

  const status = isActive
    ? STAGE_LABEL_RU[stage]
    : emotion
      ? `${EMOTION_LABEL_RU[emotion.label]} · уверенность ${(emotion.confidence * 100).toFixed(0)}%`
      : connected
        ? "в сети"
        : "нет соединения";

  return (
    <header className="h-14 shrink-0 flex items-center justify-between gap-3 px-4 bg-card border-b border-border">
      <div className="flex items-center gap-3 min-w-0">
        <div className={cn("tg-avatar w-10 h-10 text-sm transition-colors", avatarColor)}>
          {(title.replace(/[^А-Яа-яA-Za-z]/g, "").charAt(0) || "К").toUpperCase()}
        </div>
        <div className="min-w-0">
          {editing ? (
            <input
              autoFocus
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onBlur={commit}
              onKeyDown={(e) => {
                if (e.key === "Enter") commit();
                if (e.key === "Escape") {
                  setDraft(title);
                  setEditing(false);
                }
              }}
              className="bg-secondary text-foreground text-[15px] font-semibold rounded px-1.5 py-0.5 outline-none ring-1 ring-primary/50 w-[220px] max-w-full"
            />
          ) : (
            <div
              onDoubleClick={() => {
                setDraft(title);
                setEditing(true);
              }}
              className="text-[15px] font-semibold text-foreground leading-tight truncate cursor-text"
              title="Двойной клик — переименовать чат"
            >
              {title}
            </div>
          )}
          <div
            className={cn(
              "text-xs leading-tight truncate",
              isActive ? "text-primary" : "text-muted-foreground",
            )}
          >
            {status}
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2 shrink-0">
        {emotion?.escalation_risk && (
          <span className="hidden sm:flex items-center gap-1 px-2 py-1 rounded-full bg-red-500/15 text-red-300 text-xs font-medium">
            <AlertTriangle className="w-3.5 h-3.5" />
            эскалация
          </span>
        )}
        <button
          type="button"
          onClick={onStart}
          disabled={!connected || isActive}
          className={cn(
            "inline-flex items-center gap-2 px-3.5 py-2 rounded-full text-sm font-medium transition",
            "bg-primary text-primary-foreground hover:brightness-105",
            "disabled:opacity-40 disabled:cursor-not-allowed",
          )}
        >
          <Play className="w-4 h-4" />
          <span className="hidden sm:inline">Демо-звонок</span>
        </button>
        <button
          type="button"
          onClick={onStop}
          disabled={!connected}
          className="inline-flex items-center justify-center w-9 h-9 rounded-full text-muted-foreground hover:bg-secondary hover:text-foreground transition disabled:opacity-40"
          title="Сбросить сессию"
        >
          <Square className="w-4 h-4" />
        </button>
      </div>
    </header>
  );
}
