"use client";

import { Moon, Sun, X } from "lucide-react";

import { type Settings } from "@/lib/useSettings";
import { cn } from "@/lib/utils";

interface Props {
  open: boolean;
  onClose: () => void;
  settings: Settings;
  update: (patch: Partial<Settings>) => void;
}

export function SettingsModal({ open, onClose, settings, update }: Props) {
  if (!open) return null;
  const options: { id: Settings["theme"]; label: string; icon: typeof Sun }[] = [
    { id: "dark", label: "Тёмная", icon: Moon },
    { id: "light", label: "Светлая", icon: Sun },
  ];
  return (
    <>
      <div className="fixed inset-0 bg-black/50 z-40" onClick={onClose} />
      <div className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-[400px] max-w-[92vw] bg-card border border-border rounded-2xl shadow-2xl overflow-hidden">
        <div className="h-14 flex items-center justify-between px-4 border-b border-border">
          <span className="font-semibold text-[15px]">Настройки</span>
          <button
            type="button"
            onClick={onClose}
            className="w-8 h-8 rounded-full flex items-center justify-center text-muted-foreground hover:bg-secondary hover:text-foreground transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-4 space-y-5">
          <div>
            <p className="eyebrow mb-2">Тема оформления</p>
            <div className="grid grid-cols-2 gap-2.5">
              {options.map((o) => (
                <button
                  key={o.id}
                  type="button"
                  onClick={() => update({ theme: o.id })}
                  className={cn(
                    "flex items-center justify-center gap-2 py-2.5 rounded-xl border text-sm font-medium transition",
                    settings.theme === o.id
                      ? "border-primary bg-primary/10 text-foreground"
                      : "border-border bg-secondary text-muted-foreground hover:text-foreground",
                  )}
                >
                  <o.icon className="w-4 h-4" />
                  {o.label}
                </button>
              ))}
            </div>
          </div>

          <p className="text-xs text-muted-foreground">
            Настройки сохраняются в браузере и применяются при следующем открытии
          </p>
        </div>
      </div>
    </>
  );
}
