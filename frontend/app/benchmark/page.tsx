"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { ArrowLeft, AudioLines, Check, Play, Square, X, Sparkles } from "lucide-react";

import { cn } from "@/lib/utils";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const COMPANY: Record<string, string> = {
  mts: "МТС", beeline: "Билайн", megafon: "МегаФон",
  tele2: "Tele2", rostelecom: "Ростелеком", vektor: "Вектор",
};
const EMO: Record<string, { ru: string; c: string }> = {
  angry: { ru: "Раздражён", c: "bg-red-500/20 text-red-300" },
  sad: { ru: "Расстроен", c: "bg-blue-500/20 text-blue-300" },
  neutral: { ru: "Нейтрально", c: "bg-slate-500/25 text-slate-200" },
  positive: { ru: "Позитив", c: "bg-emerald-500/20 text-emerald-300" },
};

interface Turn {
  role: string;
  idx: number;
  text: string;
  emotion?: string;
  audio_url?: string | null;
  voice_name?: string;
  stability?: number;
}
interface Dialogue {
  dialogue_id: string;
  company: string;
  scenario: string;
  turns: Turn[];
}
interface Sample {
  id: string;
  batch: number;
  emotion: string;
  voice_name?: string;
  stability?: number;
  clean_text: string;
  ser_pred?: string | null;
  ser_ok?: boolean | null;
  audio_url: string;
}
interface SimResult {
  asr_text: string;
  emotion: { label: string; confidence: number; arousal: number; escalation: number };
  suggestions: string[];
  sources: { doc_id: string; title: string; snippet: string }[];
  timings_ms: { asr: number; ser: number; rag: number; llm: number };
}
interface RecogRes {
  asr_text: string;
  emotion: { label: string; confidence: number; escalation: boolean };
  timings_ms: { asr: number; ser: number };
}

const EMO_RANK: Record<string, number> = { angry: 0, positive: 1, neutral: 2, sad: 3 };

function dialogueRank(d: Dialogue): number {
  const counts: Record<string, number> = {};
  for (const t of d.turns)
    if (t.role === "client" && t.emotion) counts[t.emotion] = (counts[t.emotion] || 0) + 1;
  let dom = "neutral";
  let best = -1;
  for (const e in counts)
    if (counts[e] > best) {
      best = counts[e];
      dom = e;
    }
  return EMO_RANK[dom] ?? 2;
}

function VoicePlayer({ url }: { url: string }) {
  return <audio controls preload="none" src={`${API_BASE}${url}`} className="w-full h-9" />;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export default function BenchmarkPage() {
  const [tab, setTab] = useState<"dialogues" | "samples">("dialogues");
  const [dialogues, setDialogues] = useState<Dialogue[]>([]);
  const [selId, setSelId] = useState<string | null>(null);
  const [dlgMeta, setDlgMeta] = useState({ count: 0, client_audio: 0 });
  const [samples, setSamples] = useState<Sample[]>([]);
  const [emoFilter, setEmoFilter] = useState<string>("all");
  const [q, setQ] = useState("");
  const [recog, setRecog] = useState<Record<string, RecogRes | "loading">>({});

  const [simActive, setSimActive] = useState(false);
  const [simRunning, setSimRunning] = useState(false);
  const [simStep, setSimStep] = useState(-1);
  const [simStatus, setSimStatus] = useState("");
  const [simResults, setSimResults] = useState<Record<number, SimResult>>({});
  const simAbort = useRef(false);

  useEffect(() => {
    fetch(`${API_BASE}/api/benchmark/dialogues`)
      .then((r) => r.json())
      .then((d) => {
        const ranked = ((d.dialogues || []) as Dialogue[])
          .map((dl, i) => ({ dl, i }))
          .sort((a, b) => dialogueRank(a.dl) - dialogueRank(b.dl) || a.i - b.i)
          .map((x) => x.dl);
        setDialogues(ranked);
        setDlgMeta({ count: d.count || 0, client_audio: d.client_audio || 0 });
        if (ranked.length) setSelId(ranked[0].dialogue_id);
      })
      .catch(() => {});
    fetch(`${API_BASE}/api/benchmark/samples`)
      .then((r) => r.json())
      .then((d) => setSamples(d.samples || []))
      .catch(() => {});
  }, []);

  const sel = dialogues.find((d) => d.dialogue_id === selId);

  function selectDialogue(id: string) {
    simAbort.current = true;
    setSimActive(false);
    setSimRunning(false);
    setSimStep(-1);
    setSimResults({});
    setSimStatus("");
    setSelId(id);
  }

  function playAudio(url: string): Promise<void> {
    return new Promise<void>((resolve) => {
      const a = new Audio(`${API_BASE}${url}`);
      let done = false;
      const finish = () => { if (!done) { done = true; resolve(); } };
      a.onended = finish;
      a.onerror = finish;
      a.play().catch(finish);
      setTimeout(finish, 20000);
    });
  }

  async function runSimulation(d: Dialogue) {
    simAbort.current = false;
    setSimActive(true);
    setSimRunning(true);
    setSimResults({});
    setSimStep(-1);
    for (let i = 0; i < d.turns.length; i++) {
      if (simAbort.current) break;
      const t = d.turns[i];
      setSimStep(i);
      if (t.role === "client") {
        if (t.audio_url) {
          setSimStatus("📢 клиент говорит…");
          await playAudio(t.audio_url);
        }
        if (simAbort.current) break;
        setSimStatus("🎙 распознаю речь, определяю эмоцию, ищу в базе, формирую ответ…");
        try {
          const r: SimResult = await fetch(
            `${API_BASE}/api/benchmark/simulate-turn?dialogue_id=${d.dialogue_id}&turn_idx=${t.idx}`,
          ).then((x) => x.json());
          if (simAbort.current) break;
          setSimResults((prev) => ({ ...prev, [t.idx]: r }));
        } catch {
        }
        setSimStatus("");
        await sleep(2200);
      } else {
        await sleep(2000);
      }
    }
    setSimStatus("");
    setSimRunning(false);
  }

  function stopSimulation() {
    simAbort.current = true;
    setSimRunning(false);
    setSimStatus("");
  }

  async function runRecognize(s: Sample) {
    const file = s.audio_url.split("/").pop();
    if (!file) return;
    setRecog((p) => ({ ...p, [s.id]: "loading" }));
    try {
      const r: RecogRes = await fetch(
        `${API_BASE}/api/benchmark/recognize?file=${encodeURIComponent(file)}`,
      ).then((x) => x.json());
      setRecog((p) => ({ ...p, [s.id]: r }));
    } catch {
      setRecog((p) => {
        const n = { ...p };
        delete n[s.id];
        return n;
      });
    }
  }

  const emoCounts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const s of samples) c[s.emotion] = (c[s.emotion] || 0) + 1;
    return c;
  }, [samples]);
  const filtered = samples.filter(
    (s) =>
      (emoFilter === "all" || s.emotion === emoFilter) &&
      (!q ||
        (s.id ?? "").toLowerCase().includes(q.toLowerCase()) ||
        (s.clean_text ?? "").toLowerCase().includes(q.toLowerCase())),
  );

  const allTurns = sel?.turns ?? [];
  const visibleTurns = simActive
    ? allTurns.filter((_, i) => i <= simStep)
    : allTurns;
  const simKeys = Object.keys(simResults).map(Number).sort((a, b) => a - b);
  const curRes = simKeys.length ? simResults[simKeys[simKeys.length - 1]] : undefined;

  return (
    <div className="h-screen flex flex-col bg-background text-foreground overflow-hidden">
      <header className="h-14 shrink-0 flex items-center gap-4 px-4 border-b border-border bg-card">
        <Link href="/" className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition">
          <ArrowLeft className="w-4 h-4" />
          Копилот
        </Link>
        <span className="font-semibold text-[15px]">Голосовые данные</span>
        <div className="ml-2 flex items-center gap-1 p-0.5 rounded-lg bg-secondary text-sm">
          {(["dialogues", "samples"] as const).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setTab(t)}
              className={cn(
                "px-3 py-1 rounded-md transition",
                tab === t ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground",
              )}
            >
              {t === "dialogues" ? `Диалоги (${dlgMeta.count})` : `Одиночные (${samples.length})`}
            </button>
          ))}
        </div>
        <span className="ml-auto text-xs text-muted-foreground">
          {tab === "dialogues" ? `${dlgMeta.client_audio} аудио` : "SER-датасет"}
        </span>
      </header>

      {tab === "dialogues" ? (
        <div className="flex-1 flex min-h-0">
          <aside className="w-[300px] shrink-0 flex flex-col border-r border-border bg-card">
            <div className="flex-1 overflow-y-auto scrollbar-thin px-2 py-2 space-y-0.5">
              {dialogues.map((d) => {
                const isSel = d.dialogue_id === selId;
                return (
                  <button
                    key={d.dialogue_id}
                    type="button"
                    onClick={() => selectDialogue(d.dialogue_id)}
                    className={cn(
                      "w-full text-left px-2.5 py-2 rounded-xl transition",
                      isSel ? "bg-primary text-primary-foreground" : "hover:bg-secondary",
                    )}
                  >
                    <div className="flex items-center gap-2">
                      <AudioLines className="w-3.5 h-3.5 shrink-0 opacity-70" />
                      <span className="font-medium text-sm">{d.dialogue_id}</span>
                      <span className={cn("ml-auto text-[11px]", isSel ? "opacity-85" : "text-muted-foreground")}>
                        {COMPANY[d.company] ?? d.company}
                      </span>
                    </div>
                    <div className={cn("text-[12px] truncate mt-0.5", isSel ? "opacity-85" : "text-muted-foreground")}>
                      {d.scenario}
                    </div>
                  </button>
                );
              })}
            </div>
          </aside>

          <main className="flex-1 flex flex-col min-w-0">
            {sel && (
              <div className="h-12 shrink-0 flex items-center gap-3 px-4 border-b border-border bg-card">
                <div className="min-w-0">
                  <div className="font-semibold text-sm leading-tight">
                    {sel.dialogue_id} · {COMPANY[sel.company] ?? sel.company}
                  </div>
                  <div className="text-xs text-muted-foreground truncate">{sel.scenario}</div>
                </div>
                <div className="ml-auto flex items-center gap-2">
                  {simStatus && (
                    <span className="text-[11px] text-muted-foreground animate-pulse hidden lg:inline">
                      {simStatus}
                    </span>
                  )}
                  {!simRunning ? (
                    <button
                      type="button"
                      onClick={() => sel && runSimulation(sel)}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:opacity-90 transition"
                      title="Прогнать диалог через весь конвейер: аудио → распознавание → эмоция → поиск → подсказки"
                    >
                      <Play className="w-3.5 h-3.5" />
                      Симуляция
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={stopSimulation}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-500/85 text-white text-sm font-medium hover:opacity-90 transition"
                    >
                      <Square className="w-3.5 h-3.5" />
                      Стоп
                    </button>
                  )}
                </div>
              </div>
            )}
            <div className="flex-1 overflow-y-auto scrollbar-thin tg-wallpaper">
              <div className="max-w-2xl mx-auto px-4 py-5 flex flex-col gap-2">
                {visibleTurns.map((t, i) => {
                  const res = simActive ? simResults[t.idx] : undefined;
                  const shownText = res ? res.asr_text : t.text;
                  const shownEmo = res ? res.emotion.label : t.emotion;
                  return t.role === "client" ? (
                    <div key={i} className="flex flex-col items-start gap-2">
                      <div className="tg-bubble tg-in w-[78%]">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-[11px] font-semibold text-white/70">Клиент</span>
                          {shownEmo && (
                            <span className={cn("px-1.5 py-0.5 rounded-full text-[10.5px] font-medium", EMO[shownEmo]?.c ?? "bg-white/15 text-white/80")}>
                              {EMO[shownEmo]?.ru ?? shownEmo}
                              {res ? ` · ${Math.round(res.emotion.confidence * 100)}%` : ""}
                            </span>
                          )}
                          {res && res.emotion.escalation && (
                            <span className="px-1.5 py-0.5 rounded-full text-[10.5px] font-medium bg-amber-500/25 text-amber-200">
                              риск эскалации
                            </span>
                          )}
                        </div>
                        {t.audio_url ? (
                          <div className="mb-1.5"><VoicePlayer url={t.audio_url} /></div>
                        ) : (
                          <div className="text-[11px] text-white/50 mb-1.5">нет аудио</div>
                        )}
                        <div className="text-[14px] leading-snug">{shownText}</div>
                        {res && (
                          <div className="text-[10px] text-white/45 mt-1">
                            ASR {res.timings_ms.asr}мс · SER {res.timings_ms.ser}мс · поиск {res.timings_ms.rag}мс · LLM {res.timings_ms.llm}мс
                          </div>
                        )}
                      </div>

                      {res && res.suggestions?.length > 0 && (
                        <div className="w-[88%] rounded-2xl border border-primary/30 bg-primary/[0.06] p-3">
                          <div className="flex items-center gap-1.5 mb-2 text-[11px] font-semibold text-primary/90">
                            <Sparkles className="w-3.5 h-3.5" />
                            Подсказки копилота — оператор выбирает одну
                          </div>
                          <div className="flex flex-col gap-1.5">
                            {res.suggestions.map((s, k) => (
                              <div
                                key={k}
                                className={cn(
                                  "rounded-xl px-3 py-2 text-[13px] leading-snug border transition",
                                  k === 0
                                    ? "border-primary/50 bg-primary/15 text-foreground"
                                    : "border-border bg-card/60 text-muted-foreground",
                                )}
                              >
                                <span className="text-[10px] font-semibold mr-1.5 opacity-70">
                                  {k === 0 ? "★ вариант 1" : `вариант ${k + 1}`}
                                </span>
                                {s}
                              </div>
                            ))}
                          </div>
                          {res.sources?.length > 0 && (
                            <div className="mt-2 flex flex-wrap gap-1.5">
                              {res.sources.map((src) => (
                                <span key={src.doc_id} className="text-[10.5px] px-1.5 py-0.5 rounded-md bg-secondary text-muted-foreground" title={src.snippet}>
                                  📄 {src.title}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div key={i} className="flex justify-end">
                      <div className="tg-bubble tg-out w-[78%]">
                        <div className="text-[11px] font-semibold text-white/70 mb-1">
                          Оператор {simActive ? "· выбранный ответ" : "· эталон"}
                        </div>
                        <div className="text-[14px] leading-snug">{t.text}</div>
                      </div>
                    </div>
                  );
                })}
                {simStatus && (
                  <div className="flex justify-center py-2">
                    <span className="text-[12px] text-muted-foreground animate-pulse">{simStatus}</span>
                  </div>
                )}
              </div>
            </div>
          </main>

          {sel && simActive && (
            <aside className="hidden lg:flex w-[300px] shrink-0 flex-col border-l border-border bg-card overflow-y-auto scrollbar-thin">
              <div className="h-12 shrink-0 flex items-center px-4 border-b border-border font-semibold text-sm">
                Сведения о звонке
              </div>
              <div className="p-4 space-y-4">
                <div className="glass p-3">
                  <p className="eyebrow mb-1.5">Оператор связи</p>
                  <p className="text-sm font-medium">{COMPANY[sel.company] ?? sel.company}</p>
                </div>
                {curRes ? (
                  <>
                    <div className="glass p-3">
                      <p className="eyebrow mb-2">Эмоция клиента</p>
                      <div className="flex items-center gap-2">
                        <span className={cn("px-2 py-1 rounded-full text-xs font-medium", EMO[curRes.emotion.label]?.c ?? "bg-secondary")}>
                          {EMO[curRes.emotion.label]?.ru ?? curRes.emotion.label}
                        </span>
                        <span className="text-sm text-muted-foreground">
                          {Math.round(curRes.emotion.confidence * 100)}%
                        </span>
                      </div>
                      <div className="mt-2 text-xs text-muted-foreground space-y-1">
                        <div>Возбуждение: {curRes.emotion.arousal}</div>
                        {curRes.emotion.escalation && (
                          <div className="text-amber-400 font-medium">⚠ риск эскалации</div>
                        )}
                      </div>
                    </div>
                    <div className="glass p-3">
                      <p className="eyebrow mb-2">Задержки конвейера</p>
                      <div className="text-xs space-y-1 text-muted-foreground">
                        <div className="flex justify-between"><span>Распознавание речи</span><span>{curRes.timings_ms.asr} мс</span></div>
                        <div className="flex justify-between"><span>Эмоция</span><span>{curRes.timings_ms.ser} мс</span></div>
                        <div className="flex justify-between"><span>Поиск</span><span>{curRes.timings_ms.rag} мс</span></div>
                        <div className="flex justify-between"><span>Языковая модель</span><span>{curRes.timings_ms.llm} мс</span></div>
                        <div className="flex justify-between font-medium text-foreground pt-1 mt-1 border-t border-border">
                          <span>Итого</span>
                          <span>{curRes.timings_ms.asr + curRes.timings_ms.ser + curRes.timings_ms.rag + curRes.timings_ms.llm} мс</span>
                        </div>
                      </div>
                    </div>
                    {curRes.sources?.length > 0 && (
                      <div className="glass p-3">
                        <p className="eyebrow mb-2">Источники</p>
                        <div className="space-y-1">
                          {curRes.sources.map((s) => (
                            <div key={s.doc_id} className="text-xs text-muted-foreground truncate" title={s.title}>
                              📄 {s.title}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <p className="text-xs text-muted-foreground">
                    Идёт симуляция — здесь появятся эмоция, задержки и источники по текущей реплике.
                  </p>
                )}
              </div>
            </aside>
          )}
        </div>
      ) : (
        <div className="flex-1 flex flex-col min-h-0">
          <div className="shrink-0 flex flex-wrap items-center gap-1.5 px-4 py-2.5 border-b border-border bg-card">
            {["all", "angry", "sad", "neutral", "positive"].map((e) => (
              <button
                key={e}
                type="button"
                onClick={() => setEmoFilter(e)}
                className={cn(
                  "px-2.5 py-1 rounded-full text-xs font-medium transition",
                  emoFilter === e ? "bg-primary text-primary-foreground" : "bg-secondary text-muted-foreground hover:text-foreground",
                )}
              >
                {e === "all" ? `Все (${samples.length})` : `${EMO[e]?.ru ?? e} (${emoCounts[e] || 0})`}
              </button>
            ))}
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Поиск по id или тексту"
              className="ml-auto w-60 px-2.5 py-1 rounded-lg bg-secondary text-sm outline-none placeholder:text-muted-foreground"
            />
          </div>
          <div className="flex-1 overflow-y-auto scrollbar-thin p-4">
            <div className="max-w-5xl mx-auto grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {filtered.map((s) => (
                <div key={s.id} className="glass p-3">
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <span className={cn("px-1.5 py-0.5 rounded-full text-[10.5px] font-medium", EMO[s.emotion]?.c ?? "bg-secondary")}>
                      {EMO[s.emotion]?.ru ?? s.emotion}
                    </span>
                    {s.ser_ok != null && (
                      <span
                        className={cn(
                          "inline-flex items-center gap-1 text-[10.5px] px-1.5 py-0.5 rounded-full",
                          s.ser_ok ? "bg-emerald-500/15 text-emerald-300" : "bg-red-500/15 text-red-300",
                        )}
                        title="Угадал ли SER эмоцию"
                      >
                        {s.ser_ok ? <Check className="w-3 h-3" /> : <X className="w-3 h-3" />}
                        SER{!s.ser_ok && s.ser_pred ? `: ${s.ser_pred}` : ""}
                      </span>
                    )}
                  </div>
                  <VoicePlayer url={s.audio_url} />
                  <p className="text-[13px] leading-snug text-foreground mt-2">{s.clean_text}</p>
                  <p className="text-[10.5px] text-muted-foreground mt-1.5">
                    {s.id} · {s.voice_name}
                    {s.stability != null ? ` · stab ${s.stability}` : ""}
                  </p>
                  <button
                    type="button"
                    onClick={() => runRecognize(s)}
                    disabled={recog[s.id] === "loading"}
                    className="mt-2 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-primary text-primary-foreground text-xs font-medium hover:opacity-90 transition disabled:opacity-60"
                    title="Распознать речь и эмоцию по этому клипу (без поиска и генерации)"
                  >
                    <Play className="w-3 h-3" />
                    {recog[s.id] === "loading" ? "Распознаю…" : "Симуляция"}
                  </button>
                  {recog[s.id] && recog[s.id] !== "loading"
                    ? (() => {
                        const r = recog[s.id] as RecogRes;
                        const ok = r.emotion.label === s.emotion;
                        return (
                          <div className="mt-2 rounded-lg border border-border bg-card/60 p-2 text-[12px] space-y-1">
                            <div className="flex items-center gap-1.5 flex-wrap">
                              <span className="text-muted-foreground">распознано:</span>
                              <span className={cn("px-1.5 py-0.5 rounded-full text-[10.5px] font-medium", EMO[r.emotion.label]?.c ?? "bg-secondary")}>
                                {EMO[r.emotion.label]?.ru ?? r.emotion.label} · {Math.round(r.emotion.confidence * 100)}%
                              </span>
                              <span className={cn("inline-flex items-center gap-1 text-[10.5px] px-1.5 py-0.5 rounded-full", ok ? "bg-emerald-500/15 text-emerald-300" : "bg-red-500/15 text-red-300")}>
                                {ok ? <Check className="w-3 h-3" /> : <X className="w-3 h-3" />}
                                {ok ? "верно" : "мимо"}
                              </span>
                            </div>
                            <p className="text-foreground leading-snug">{r.asr_text}</p>
                            <p className="text-[10px] text-muted-foreground">ASR {r.timings_ms.asr}мс · эмоция {r.timings_ms.ser}мс</p>
                          </div>
                        );
                      })()
                    : null}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
