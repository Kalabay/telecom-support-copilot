"use client";

import { useCallback, useRef, useState } from "react";
import { Loader2, Mic, Upload } from "lucide-react";

import type { EmotionState } from "@/lib/types";
import { cn, EMOTION_LABEL_RU } from "@/lib/utils";

interface SerResponse {
  emotion: EmotionState;
  probs: Record<string, number>;
  inference_ms: number;
  duration_ms: number;
  filename?: string;
}

const STYLE_BY_LABEL: Record<string, string> = {
  neutral: "border-slate-500/30 bg-slate-500/15 text-slate-200",
  angry: "border-red-500/30 bg-red-500/15 text-red-300",
  positive: "border-emerald-500/30 bg-emerald-500/15 text-emerald-300",
  sad: "border-blue-500/30 bg-blue-500/15 text-blue-300",
  other: "border-violet-500/30 bg-violet-500/15 text-violet-300",
};

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export function SerUploadPanel() {
  const [result, setResult] = useState<SerResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const upload = useCallback(async (file: File) => {
    setError(null);
    setLoading(true);
    setResult(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const r = await fetch(`${API}/api/ser`, { method: "POST", body: fd });
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
      setResult(await r.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) upload(file);
  };

  const onFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) upload(file);
  };

  return (
    <section className="glass p-5">
      <header className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Mic className="w-4 h-4 text-primary" />
          <h2 className="eyebrow">Распознавание эмоции из файла</h2>
        </div>
        <span className="text-xs text-muted-foreground">5 классов</span>
      </header>

      <div className="space-y-4">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          className={cn(
            "rounded-xl border-2 border-dashed cursor-pointer transition-colors",
            "p-6 flex flex-col items-center justify-center gap-2 text-center min-h-[140px]",
            dragOver
              ? "border-primary bg-accent"
              : "border-border hover:border-primary/50 hover:bg-secondary/50",
          )}
        >
          {loading ? (
            <Loader2 className="w-6 h-6 animate-spin text-primary" />
          ) : (
            <Upload className="w-6 h-6 text-muted-foreground" />
          )}
          <p className="text-sm font-medium text-foreground/80">
            {loading ? "Распознаю эмоцию…" : "Перетащите WAV/MP3 или нажмите для выбора"}
          </p>
          <p className="text-xs text-muted-foreground">≤ 25 МБ · до 10 сек</p>
          <input
            ref={inputRef}
            type="file"
            accept="audio/*"
            className="hidden"
            onChange={onFile}
          />
        </div>

        <div>
          {error && (
            <div className="rounded-lg border border-red-500/30 bg-red-500/15 text-red-300 px-3 py-2 text-xs">
              {error}
            </div>
          )}

          {!result && !error && (
            <div className="h-full min-h-[140px] flex items-center justify-center text-center text-xs text-muted-foreground">
              Результат распознавания появится здесь
            </div>
          )}

          {result && (
            <div className="space-y-3 animate-slide_up">
              <div
                className={cn(
                  "rounded-xl border px-4 py-3 flex items-center justify-between gap-3",
                  STYLE_BY_LABEL[result.emotion.label],
                )}
              >
                <div>
                  <div className="text-xl font-semibold tracking-tight leading-none">
                    {EMOTION_LABEL_RU[result.emotion.label]}
                  </div>
                  <div className="text-xs opacity-80 mt-1">
                    уверенность {(result.emotion.confidence * 100).toFixed(0)}% · arousal{" "}
                    {result.emotion.arousal.toFixed(2)} · valence{" "}
                    {result.emotion.valence.toFixed(2)}
                  </div>
                </div>
                {result.emotion.escalation_risk && (
                  <div className="px-2.5 py-1 rounded-full bg-red-500/20 text-red-300 text-xs font-medium">
                    Риск эскалации
                  </div>
                )}
              </div>

              <div className="space-y-1.5">
                {Object.entries(result.probs)
                  .sort(([, a], [, b]) => b - a)
                  .map(([label, p]) => (
                    <div key={label} className="flex items-center gap-2 text-xs">
                      <span className="w-20 text-muted-foreground">
                        {EMOTION_LABEL_RU[label] ?? label}
                      </span>
                      <div className="flex-1 h-1.5 rounded-full bg-secondary overflow-hidden">
                        <div
                          className="h-full bg-primary rounded-full"
                          style={{ width: `${p * 100}%` }}
                        />
                      </div>
                      <span className="w-12 tabular-nums text-right text-foreground/70">
                        {(p * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
              </div>

              <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground pt-1">
                {result.filename && <span>{result.filename}</span>}
                <span>длительность {result.duration_ms} мс</span>
                <span className="ml-auto">инференс {result.inference_ms} мс</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
