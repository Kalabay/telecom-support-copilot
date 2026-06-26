"use client";

import { useEffect, useRef } from "react";
import { BookOpen } from "lucide-react";

import type { TranscriptSegment } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  segments: TranscriptSegment[];
  onOpenSource?: (docId: string) => void;
}

export function TranscriptPanel({ segments, onOpenSource }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [segments.length]);

  return (
    <div className="flex flex-col gap-1.5 py-4 px-3 sm:px-6 min-h-full justify-end">
      {segments.length === 0 && (
        <div className="m-auto text-center">
          <div className="inline-block rounded-2xl bg-black/15 text-white/95 text-xs px-4 py-2 backdrop-blur-sm">
            Звонок ещё не начался. Нажмите «Демо-звонок» или говорите в микрофон.
          </div>
        </div>
      )}

      {segments.map((seg, i) => {
        const isCustomer = seg.speaker === "customer";
        const sources = seg.sources ?? [];
        return (
          <div
            key={i}
            className={cn("flex w-full animate-slide_up", isCustomer ? "justify-start" : "justify-end")}
          >
            <div className={cn("tg-bubble", isCustomer ? "tg-in" : "tg-out")}>
              <span className="whitespace-pre-wrap">{seg.text}</span>
              <span className="float-right ml-2 mt-1 text-[10px] text-white/45 tabular-nums select-none">
                {(seg.start_ms / 1000).toFixed(0)}s
              </span>

              {!isCustomer && sources.length > 0 && (
                <div className="clear-both flex flex-wrap gap-1 mt-1.5 pt-1.5 border-t border-white/15">
                  {sources.map((src) => (
                    <button
                      key={src.doc_id}
                      type="button"
                      onClick={() => onOpenSource?.(src.doc_id)}
                      className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-white/10 hover:bg-white/20 text-[10.5px] text-white/80 transition max-w-full"
                      title="Открыть статью базы знаний"
                    >
                      <BookOpen className="w-2.5 h-2.5 shrink-0" />
                      <span className="truncate max-w-[140px]">{src.title}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        );
      })}

      <div ref={bottomRef} />
    </div>
  );
}
