"use client";

import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface Engine {
  id: string;
  label: string;
}

export function AsrSelector() {
  const [engines, setEngines] = useState<Engine[]>([]);
  const [current, setCurrent] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/api/asr/engines`)
      .then((r) => r.json())
      .then((d) => {
        setEngines(d.engines || []);
        setCurrent(d.current || "");
      })
      .catch(() => {});
  }, []);

  const change = async (id: string) => {
    setBusy(true);
    setCurrent(id);
    try {
      await fetch(`${API_BASE}/api/asr/engine`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ engine: id }),
      });
    } catch {
      /* ignore */
    }
    setBusy(false);
  };

  if (!engines.length) return null;

  return (
    <div className="glass p-4">
      <p className="eyebrow mb-2">Распознавание речи (ASR)</p>
      <select
        value={current}
        disabled={busy}
        onChange={(e) => change(e.target.value)}
        className="w-full bg-secondary text-foreground text-sm rounded-lg px-3 py-2 outline-none border border-border focus:border-primary/60 disabled:opacity-60"
      >
        {engines.map((e) => (
          <option key={e.id} value={e.id}>
            {e.label}
          </option>
        ))}
      </select>
      <p className="text-xs text-muted-foreground mt-2">
        Модель распознавания речи (голос&rarr;текст)
      </p>
    </div>
  );
}
