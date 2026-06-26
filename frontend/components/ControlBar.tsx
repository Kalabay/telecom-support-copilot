"use client";

import { Play, Square } from "lucide-react";

import { cn } from "@/lib/utils";

interface Props {
  onStart: () => void;
  onStop: () => void;
  disabled: boolean;
  isActive: boolean;
}

export function ControlBar({ onStart, onStop, disabled, isActive }: Props) {
  return (
    <div className="glass rounded-2xl px-5 py-4 flex items-center gap-4">
      <div className="flex-1">
        <p className="text-sm font-medium">Демо-сценарий: «Не работает интернет»</p>
        <p className="text-xs text-muted-foreground">
          Имитация звонка клиента МТС с нарастающей эскалацией · 4 реплики · ~10 сек
        </p>
      </div>
      <button
        type="button"
        onClick={onStart}
        disabled={disabled || isActive}
        className={cn(
          "inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition",
          "bg-primary text-primary-foreground hover:brightness-110",
          "disabled:opacity-40 disabled:cursor-not-allowed",
        )}
      >
        <Play className="w-4 h-4" />
        Запустить демо
      </button>
      <button
        type="button"
        onClick={onStop}
        disabled={disabled}
        className={cn(
          "inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition",
          "bg-secondary text-secondary-foreground hover:bg-secondary/80",
          "disabled:opacity-40 disabled:cursor-not-allowed",
        )}
      >
        <Square className="w-4 h-4" />
        Сбросить
      </button>
    </div>
  );
}
