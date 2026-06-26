"use client";

import { AlertTriangle, Activity } from "lucide-react";

import type { EmotionState } from "@/lib/types";
import { cn, EMOTION_LABEL_RU } from "@/lib/utils";

interface Props {
  emotion: EmotionState | null;
}

const STYLE_BY_LABEL: Record<string, string> = {
  neutral: "border-slate-500/30 bg-slate-500/15 text-slate-200",
  angry: "border-red-500/30 bg-red-500/15 text-red-300",
  positive: "border-emerald-500/30 bg-emerald-500/15 text-emerald-300",
  sad: "border-blue-500/30 bg-blue-500/15 text-blue-300",
  other: "border-violet-500/30 bg-violet-500/15 text-violet-300",
};

const BAR_BY_LABEL: Record<string, string> = {
  neutral: "bg-slate-400",
  angry: "bg-red-500",
  positive: "bg-emerald-500",
  sad: "bg-blue-500",
  other: "bg-violet-500",
};

function Bar({
  label,
  value,
  color,
  bipolar = false,
}: {
  label: string;
  value: number;
  color: string;
  bipolar?: boolean;
}) {
  const pct = bipolar ? ((value + 1) / 2) * 100 : value * 100;
  return (
    <div>
      <div className="flex items-center justify-between text-xs text-muted-foreground mb-1.5">
        <span>{label}</span>
        <span className="tabular-nums font-medium text-foreground/70">
          {bipolar ? value.toFixed(2) : `${pct.toFixed(0)}%`}
        </span>
      </div>
      <div className="h-2 rounded-full bg-secondary overflow-hidden">
        <div
          className={cn("h-full rounded-full transition-all duration-500", color)}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export function EmotionPanel({ emotion }: Props) {
  return (
    <section className="glass p-5">
      <header className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-primary" />
          <h2 className="eyebrow">Эмоция клиента</h2>
        </div>
      </header>

      {emotion ? (
        <>
          <div
            className={cn(
              "rounded-xl border px-4 py-3 flex items-center justify-between gap-3",
              STYLE_BY_LABEL[emotion.label],
            )}
          >
            <div>
              <div className="text-2xl font-semibold tracking-tight leading-none">
                {EMOTION_LABEL_RU[emotion.label]}
              </div>
              <div className="text-xs opacity-80 mt-1">
                уверенность {(emotion.confidence * 100).toFixed(0)}%
              </div>
            </div>
            {emotion.escalation_risk && (
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-red-500/20 text-red-300 text-xs font-medium animate-pulse_glow">
                <AlertTriangle className="w-3.5 h-3.5" />
                Риск эскалации
              </div>
            )}
          </div>

          <div className="mt-4 grid sm:grid-cols-2 gap-4">
            <Bar
              label="Возбуждение (arousal)"
              value={emotion.arousal}
              color={BAR_BY_LABEL[emotion.label]}
            />
            <Bar
              label="Позитив / негатив (valence)"
              value={emotion.valence}
              color={BAR_BY_LABEL[emotion.label]}
              bipolar
            />
          </div>
        </>
      ) : (
        <div className="h-24 flex flex-col items-center justify-center text-center text-muted-foreground gap-1.5">
          <Activity className="w-7 h-7 opacity-30" />
          <p className="text-xs">Эмоция появится после первой реплики клиента</p>
        </div>
      )}
    </section>
  );
}
