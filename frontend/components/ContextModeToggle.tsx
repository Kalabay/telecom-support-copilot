"use client";

import { MessageSquare, Users } from "lucide-react";

import { cn } from "@/lib/utils";

export type ContextMode = "customer" | "dialog";

interface Props {
  mode: ContextMode;
  onChange: (m: ContextMode) => void;
}

export function ContextModeToggle({ mode, onChange }: Props) {
  return (
    <div className="glass px-4 py-3">
      <p className="eyebrow mb-2">Контекст для копилота</p>
      <div className="grid grid-cols-2 gap-1 p-1 rounded-xl bg-secondary">
        <button
          onClick={() => onChange("customer")}
          className={cn(
            "inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition",
            mode === "customer"
              ? "bg-primary text-primary-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          <MessageSquare className="w-3.5 h-3.5" />
          Только клиент
        </button>
        <button
          onClick={() => onChange("dialog")}
          className={cn(
            "inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition",
            mode === "dialog"
              ? "bg-primary text-primary-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          <Users className="w-3.5 h-3.5" />
          Клиент + оператор
        </button>
      </div>
      <p className="text-xs text-muted-foreground mt-2">
        {mode === "customer"
          ? "Учитываются только реплики клиента"
          : "Учитывается весь диалог — клик по подсказке фиксирует ответ оператора"}
      </p>
    </div>
  );
}
