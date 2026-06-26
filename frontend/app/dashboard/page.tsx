"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  AlertTriangle,
  ArrowLeft,
  BookOpen,
  Clock,
  MessageSquare,
  Phone,
  RefreshCw,
  X,
} from "lucide-react";

import { EMOTION_LABEL_RU } from "@/lib/utils";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const EMO_COLOR: Record<string, string> = {
  neutral: "bg-slate-500",
  angry: "bg-red-500",
  positive: "bg-emerald-500",
  sad: "bg-blue-500",
  other: "bg-violet-500",
};

interface RecentSession {
  session_id: string;
  started: string;
  company: string | null;
  client_turns: number;
  operator_turns: number;
  last_emotion: string | null;
  escalated: boolean;
}

interface Analytics {
  total_sessions: number;
  total_client_turns: number;
  total_operator_turns: number;
  escalation_count: number;
  avg_latency_ms: number;
  emotion_counts: Record<string, number>;
  company_counts: Record<string, number>;
  recent_sessions: RecentSession[];
}

interface SessionEvent {
  ts: string;
  type: string;
  text: string;
  emotion?: { label: string; escalation_risk: boolean } | null;
  suggestions?: string[];
  sources?: string[];
  company?: string | null;
}

function StatCard({
  icon,
  label,
  value,
  accent,
}: {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  accent?: string;
}) {
  return (
    <div className="glass p-4">
      <div className="flex items-center gap-2 text-muted-foreground mb-1">
        {icon}
        <span className="text-xs">{label}</span>
      </div>
      <div className={`text-2xl font-bold ${accent ?? "text-foreground"}`}>{value}</div>
    </div>
  );
}

export default function Dashboard() {
  const [data, setData] = useState<Analytics | null>(null);
  const [loading, setLoading] = useState(true);
  const [detail, setDetail] = useState<{ session_id: string; events: SessionEvent[] } | null>(
    null,
  );

  const load = async () => {
    setLoading(true);
    try {
      const r = await fetch(`${API_BASE}/api/analytics`);
      if (r.ok) setData(await r.json());
    } catch {
    }
    setLoading(false);
  };

  useEffect(() => {
    load();
  }, []);

  const openSession = async (sid: string) => {
    try {
      const r = await fetch(`${API_BASE}/api/analytics/session/${sid}`);
      if (r.ok) setDetail(await r.json());
    } catch {
    }
  };

  const emoTotal = data
    ? Object.values(data.emotion_counts).reduce((a, b) => a + b, 0)
    : 0;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="h-14 flex items-center justify-between px-4 sm:px-6 border-b border-border bg-card">
        <div className="flex items-center gap-3">
          <Link
            href="/"
            className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition"
          >
            <ArrowLeft className="w-4 h-4" />
            Копилот
          </Link>
          <span className="font-semibold text-[15px]">Аналитика звонков</span>
        </div>
        <button
          type="button"
          onClick={load}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm text-muted-foreground hover:text-foreground hover:bg-secondary transition"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
          Обновить
        </button>
      </header>

      <main className="max-w-5xl mx-auto p-4 sm:p-6 space-y-5">
        {!data || data.total_sessions === 0 ? (
          <div className="glass p-8 text-center text-muted-foreground">
            Пока нет данных. Проведите звонок в копилоте (демо или голос) — события
            появятся здесь.
          </div>
        ) : (
          <>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <StatCard
                icon={<Phone className="w-4 h-4" />}
                label="Сессий"
                value={data.total_sessions}
              />
              <StatCard
                icon={<MessageSquare className="w-4 h-4" />}
                label="Реплик клиента"
                value={data.total_client_turns}
              />
              <StatCard
                icon={<AlertTriangle className="w-4 h-4" />}
                label="Эскалаций"
                value={data.escalation_count}
                accent={data.escalation_count > 0 ? "text-red-400" : undefined}
              />
              <StatCard
                icon={<Clock className="w-4 h-4" />}
                label="Средняя задержка"
                value={`${(data.avg_latency_ms / 1000).toFixed(1)} с`}
              />
            </div>

            <div className="grid md:grid-cols-2 gap-3">
              <div className="glass p-4">
                <p className="eyebrow mb-3">Распределение эмоций клиентов</p>
                <div className="space-y-2">
                  {Object.entries(data.emotion_counts)
                    .sort((a, b) => b[1] - a[1])
                    .map(([label, count]) => {
                      const pct = emoTotal ? Math.round((count / emoTotal) * 100) : 0;
                      return (
                        <div key={label} className="flex items-center gap-2">
                          <span className="w-20 text-xs text-muted-foreground shrink-0">
                            {EMOTION_LABEL_RU[label as keyof typeof EMOTION_LABEL_RU] ?? label}
                          </span>
                          <div className="flex-1 h-5 rounded bg-secondary overflow-hidden">
                            <div
                              className={`h-full ${EMO_COLOR[label] ?? "bg-primary"}`}
                              style={{ width: `${pct}%` }}
                            />
                          </div>
                          <span className="w-14 text-right text-xs tabular-nums text-muted-foreground">
                            {count} · {pct}%
                          </span>
                        </div>
                      );
                    })}
                </div>
              </div>

              <div className="glass p-4">
                <p className="eyebrow mb-3">Компании (база знаний)</p>
                <div className="space-y-2">
                  {Object.entries(data.company_counts)
                    .sort((a, b) => b[1] - a[1])
                    .map(([company, count]) => (
                      <div
                        key={company}
                        className="flex items-center justify-between text-sm"
                      >
                        <span className="text-muted-foreground">{company}</span>
                        <span className="tabular-nums">{count}</span>
                      </div>
                    ))}
                </div>
              </div>
            </div>

            <div className="glass p-4">
              <p className="eyebrow mb-3">Недавние звонки</p>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs text-muted-foreground border-b border-border">
                      <th className="py-2 pr-3 font-medium">Сессия</th>
                      <th className="py-2 pr-3 font-medium">Время</th>
                      <th className="py-2 pr-3 font-medium">Компания</th>
                      <th className="py-2 pr-3 font-medium text-right">Реплик</th>
                      <th className="py-2 pr-3 font-medium">Эмоция</th>
                      <th className="py-2 font-medium"></th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.recent_sessions.map((s) => (
                      <tr
                        key={s.session_id}
                        onClick={() => openSession(s.session_id)}
                        className="border-b border-border/50 hover:bg-secondary/50 cursor-pointer transition"
                      >
                        <td className="py-2 pr-3 font-mono text-xs">{s.session_id}</td>
                        <td className="py-2 pr-3 text-muted-foreground text-xs">
                          {s.started.replace("T", " ")}
                        </td>
                        <td className="py-2 pr-3">{s.company ?? "—"}</td>
                        <td className="py-2 pr-3 text-right tabular-nums">
                          {s.client_turns}/{s.operator_turns}
                        </td>
                        <td className="py-2 pr-3">
                          {s.last_emotion && (
                            <span className="inline-flex items-center gap-1.5">
                              <span
                                className={`w-2 h-2 rounded-full ${EMO_COLOR[s.last_emotion] ?? "bg-slate-500"}`}
                              />
                              {EMOTION_LABEL_RU[
                                s.last_emotion as keyof typeof EMOTION_LABEL_RU
                              ] ?? s.last_emotion}
                            </span>
                          )}
                        </td>
                        <td className="py-2">
                          {s.escalated && (
                            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-red-500/15 text-red-300 text-xs">
                              <AlertTriangle className="w-3 h-3" />
                              эскалация
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-[11px] text-muted-foreground mt-2">
                Клик по строке — полный диалог: реплики клиента с эмоцией, предложенные
                варианты и что оператор отправил.
              </p>
            </div>
          </>
        )}
      </main>

      {detail && (
        <>
          <div className="fixed inset-0 bg-black/50 z-40" onClick={() => setDetail(null)} />
          <aside className="fixed right-0 top-0 h-full w-full sm:w-[480px] bg-card border-l border-border z-50 flex flex-col shadow-2xl">
            <div className="h-14 shrink-0 flex items-center justify-between px-4 border-b border-border">
              <span className="font-semibold text-sm font-mono">{detail.session_id}</span>
              <button
                type="button"
                onClick={() => setDetail(null)}
                className="w-8 h-8 rounded-full flex items-center justify-center text-muted-foreground hover:bg-secondary hover:text-foreground transition"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto scrollbar-thin p-4 space-y-3">
              {detail.events.map((e, i) =>
                e.type === "client_turn" ? (
                  <div key={i} className="rounded-xl border border-border bg-secondary/40 p-3">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs font-semibold text-muted-foreground">Клиент</span>
                      {e.emotion && (
                        <span className="inline-flex items-center gap-1 text-xs">
                          <span
                            className={`w-2 h-2 rounded-full ${EMO_COLOR[e.emotion.label] ?? "bg-slate-500"}`}
                          />
                          {EMOTION_LABEL_RU[
                            e.emotion.label as keyof typeof EMOTION_LABEL_RU
                          ] ?? e.emotion.label}
                          {e.emotion.escalation_risk && (
                            <AlertTriangle className="w-3 h-3 text-red-400" />
                          )}
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-foreground">{e.text}</p>
                    {e.suggestions && e.suggestions.length > 0 && (
                      <div className="mt-2 pt-2 border-t border-border/50">
                        <p className="text-[11px] text-muted-foreground mb-1">
                          Предложено вариантов: {e.suggestions.length}
                        </p>
                        <ul className="space-y-1">
                          {e.suggestions.map((s, j) => (
                            <li key={j} className="text-xs text-muted-foreground">
                              {j + 1}. {s}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                ) : (
                  <div
                    key={i}
                    className="rounded-xl border border-primary/30 bg-primary/5 p-3 ml-6"
                  >
                    <span className="text-xs font-semibold text-primary">Оператор отправил</span>
                    <p className="text-sm text-foreground mt-1">{e.text}</p>
                    {e.sources && e.sources.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-2">
                        {e.sources.map((doc) => (
                          <span
                            key={doc}
                            className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-secondary text-[10.5px] text-muted-foreground"
                          >
                            <BookOpen className="w-2.5 h-2.5" />
                            {doc}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ),
              )}
            </div>
          </aside>
        </>
      )}
    </div>
  );
}
