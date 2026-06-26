"use client";

import { Gauge } from "lucide-react";

import type { LatencyBreakdown } from "@/lib/types";
import { cn, formatLatency } from "@/lib/utils";

interface Props {
  latency: LatencyBreakdown;
}

const COMPONENTS: {
  key: keyof Omit<LatencyBreakdown, "total_ms">;
  label: string;
  color: string;
}[] = [
  { key: "asr_ms", label: "ASR", color: "bg-sky-500" },
  { key: "ser_ms", label: "SER", color: "bg-violet-500" },
  { key: "retrieval_ms", label: "Поиск", color: "bg-amber-500" },
  { key: "llm_ms", label: "LLM", color: "bg-emerald-500" },
];

export function LatencyPanel({ latency }: Props) {
  const total = Math.max(latency.total_ms, 1);
  const slaOk = latency.total_ms > 0 && latency.total_ms < 3000;
  return (
    <section className="glass p-5">
      <header className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Gauge className="w-4 h-4 text-primary" />
          <h2 className="eyebrow">Задержки пайплайна</h2>
        </div>
        <span
          className={cn(
            "text-xs font-medium px-2.5 py-1 rounded-full",
            latency.total_ms === 0
              ? "text-muted-foreground bg-secondary"
              : slaOk
                ? "text-emerald-300 bg-emerald-500/15"
                : "text-amber-300 bg-amber-500/15",
          )}
        >
          итого {formatLatency(latency.total_ms)}
          {slaOk ? " · ≤ 3 с" : ""}
        </span>
      </header>

      <div className="h-2 rounded-full bg-secondary overflow-hidden flex">
        {COMPONENTS.map((c) => {
          const v = latency[c.key];
          const pct = (v / total) * 100;
          return (
            <div
              key={c.key}
              className={cn(c.color, "transition-all duration-500")}
              style={{ width: `${pct}%` }}
              title={`${c.label}: ${formatLatency(v)}`}
            />
          );
        })}
      </div>

      <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-x-4 gap-y-2 text-xs">
        {COMPONENTS.map((c) => (
          <div key={c.key} className="flex items-center gap-1.5">
            <span className={cn("w-2 h-2 rounded-sm", c.color)} />
            <span className="text-muted-foreground">{c.label}</span>
            <span className="ml-auto tabular-nums font-medium text-foreground/70">
              {latency[c.key] === 0 ? "—" : formatLatency(latency[c.key])}
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}
