"use client";

import { useEffect, useState } from "react";

export interface Settings {
  theme: "dark" | "light";
}

const DEFAULT: Settings = { theme: "dark" };
const KEY = "copilot.settings";

function apply(s: Settings) {
  const root = document.documentElement;
  root.classList.toggle("light", s.theme === "light");
}

export function useSettings() {
  const [settings, setSettings] = useState<Settings>(DEFAULT);

  useEffect(() => {
    let s = DEFAULT;
    try {
      const raw = localStorage.getItem(KEY);
      if (raw) s = { ...DEFAULT, ...JSON.parse(raw) };
    } catch {
    }
    setSettings(s);
    apply(s);
  }, []);

  const update = (patch: Partial<Settings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch };
      try {
        localStorage.setItem(KEY, JSON.stringify(next));
      } catch {
      }
      apply(next);
      return next;
    });
  };

  return { settings, update };
}
