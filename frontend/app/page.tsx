"use client";

import { useState } from "react";
import Link from "next/link";
import {
  AudioLines,
  BarChart3,
  BookOpen,
  FileText,
  Headphones,
  Plus,
  Search,
  Settings as SettingsIcon,
  X,
} from "lucide-react";

import { EmotionPanel } from "@/components/EmotionPanel";
import { HeaderBar } from "@/components/HeaderBar";
import { AsrSelector } from "@/components/AsrSelector";
import { LatencyPanel } from "@/components/LatencyPanel";
import { MicRecorder } from "@/components/MicRecorder";
import { SerUploadPanel } from "@/components/SerUploadPanel";
import { SettingsModal } from "@/components/SettingsModal";
import { SuggestionsPanel } from "@/components/SuggestionsPanel";
import { TranscriptPanel } from "@/components/TranscriptPanel";
import { useChats } from "@/lib/useChats";
import { useSettings } from "@/lib/useSettings";
import { cn } from "@/lib/utils";
import type { CopilotUpdate } from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const EMPTY: CopilotUpdate = {
  timestamp: "",
  transcript: [],
  emotion: null,
  suggestions: [],
  latency: { asr_ms: 0, ser_ms: 0, retrieval_ms: 0, llm_ms: 0, total_ms: 0 },
  pipeline_stage: "idle",
};

const DOT: Record<string, string> = {
  neutral: "bg-slate-400",
  angry: "bg-red-500",
  positive: "bg-emerald-500",
  sad: "bg-blue-500",
  other: "bg-violet-500",
};

interface Article {
  doc_id: string;
  title: string;
  text: string;
  company: string;
  instruction?: string;
  original?: string;
  source_url?: string;
}

export default function Page() {
  const {
    chats,
    activeId,
    active,
    setActiveId,
    createChat,
    closeChat,
    renameChat,
    setCompany,
    dismissSet,
    startDemo,
    stop,
    sendVoice,
    commitOperator,
    setVoiceSpeaker,
  } = useChats();

  const { settings, update: updateSettings } = useSettings();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [voiceSpeaker, setVoiceSpeakerState] =
    useState<"customer" | "operator">("customer");
  const [article, setArticle] = useState<Article | null>(null);
  const [articleLoading, setArticleLoading] = useState(false);
  const [articleView, setArticleView] = useState<"original" | "doc" | "instruction">("original");

  const st = active?.state ?? EMPTY;
  const connected = active?.connected ?? false;
  const isActive = st.pipeline_stage !== "idle" && st.pipeline_stage !== "ready";

  const changeSpeaker = (s: "customer" | "operator") => {
    setVoiceSpeakerState(s);
    setVoiceSpeaker(s);
  };

  const openArticle = async (docId: string) => {
    setArticleLoading(true);
    setArticleView("original");
    setArticle({ doc_id: docId, title: "", text: "", company: "" });
    try {
      const r = await fetch(`${API_BASE}/api/kb/doc?doc_id=${encodeURIComponent(docId)}`);
      if (r.ok) setArticle(await r.json());
      else setArticle({ doc_id: docId, title: "Статья не найдена", text: "", company: "" });
    } catch {
      setArticle({ doc_id: docId, title: "Ошибка загрузки", text: "", company: "" });
    } finally {
      setArticleLoading(false);
    }
  };

  return (
    <div className="h-screen flex bg-background overflow-hidden text-foreground">
      <aside className="hidden md:flex w-[324px] shrink-0 flex-col border-r border-border bg-card">
        <div className="h-14 shrink-0 flex items-center justify-between gap-2 pl-3 pr-2 border-b border-border">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
              <Headphones className="w-4 h-4 text-primary-foreground" />
            </div>
            <span className="font-semibold text-[15px]">Копилот</span>
          </div>
          <div className="flex items-center gap-0.5">
            <button
              type="button"
              onClick={() => setSettingsOpen(true)}
              className="w-9 h-9 rounded-full flex items-center justify-center text-muted-foreground hover:bg-secondary hover:text-foreground transition"
              title="Настройки"
            >
              <SettingsIcon className="w-5 h-5" />
            </button>
            <Link
              href="/kb"
              className="w-9 h-9 rounded-full flex items-center justify-center text-muted-foreground hover:bg-secondary hover:text-foreground transition"
              title="База знаний (документы)"
            >
              <FileText className="w-5 h-5" />
            </Link>
            <Link
              href="/benchmark"
              className="w-9 h-9 rounded-full flex items-center justify-center text-muted-foreground hover:bg-secondary hover:text-foreground transition"
              title="Голосовые данные бенчмарка"
            >
              <AudioLines className="w-5 h-5" />
            </Link>
            <Link
              href="/dashboard"
              className="w-9 h-9 rounded-full flex items-center justify-center text-muted-foreground hover:bg-secondary hover:text-foreground transition"
              title="Аналитика звонков"
            >
              <BarChart3 className="w-5 h-5" />
            </Link>
            <button
              type="button"
              onClick={() => createChat()}
              className="w-9 h-9 rounded-full flex items-center justify-center text-muted-foreground hover:bg-secondary hover:text-foreground transition"
              title="Новый чат"
            >
              <Plus className="w-5 h-5" />
            </button>
          </div>
        </div>

        <div className="p-2">
          <div className="flex items-center gap-2 px-3 py-2 rounded-full bg-secondary text-muted-foreground text-sm">
            <Search className="w-4 h-4" />
            <span>Поиск</span>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto scrollbar-thin px-2 space-y-0.5">
          <button
            type="button"
            onClick={() => createChat()}
            className="w-full flex items-center gap-2 px-2.5 py-2 mb-1 rounded-lg text-sm font-medium text-primary hover:bg-secondary transition"
          >
            <Plus className="w-4 h-4" />
            Новый чат
          </button>
          {chats.map((c) => {
            const isSel = c.id === activeId;
            const emo = c.state.emotion;
            const preview =
              c.state.transcript.length > 0
                ? c.state.transcript[c.state.transcript.length - 1].text
                : "Нет сообщений";
            return (
              <div
                key={c.id}
                onClick={() => setActiveId(c.id)}
                className={cn(
                  "group flex items-center gap-3 px-2.5 py-2 rounded-xl cursor-pointer transition",
                  isSel ? "bg-primary text-primary-foreground" : "hover:bg-secondary",
                )}
              >
                <div
                  className={cn(
                    "tg-avatar w-12 h-12 text-base shrink-0",
                    emo ? DOT[emo.label] : isSel ? "bg-white/25" : "bg-primary/70",
                  )}
                >
                  {(c.title.replace(/[^А-Яа-яA-Za-z]/g, "").charAt(0) || "К").toUpperCase()}
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-medium text-sm truncate">{c.title}</span>
                    {emo?.escalation_risk && (
                      <span className="w-2 h-2 rounded-full bg-red-500 shrink-0" />
                    )}
                  </div>
                  <div
                    className={cn(
                      "text-[13px] truncate",
                      isSel ? "opacity-90" : "text-muted-foreground",
                    )}
                  >
                    {preview}
                  </div>
                </div>
                {chats.length > 1 && (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      closeChat(c.id);
                    }}
                    className={cn(
                      "opacity-0 group-hover:opacity-100 transition w-6 h-6 rounded-full flex items-center justify-center shrink-0",
                      isSel ? "hover:bg-white/20" : "hover:bg-background",
                    )}
                    title="Закрыть чат"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            );
          })}
        </div>

        <div className="p-3 text-[11px] text-muted-foreground leading-relaxed border-t border-border">
          НИУ ВШЭ · 2026 · Калабай Михаил Иванович
        </div>
      </aside>

      <main className="flex-1 flex flex-col min-w-0">
        <HeaderBar
          connected={connected}
          stage={st.pipeline_stage}
          emotion={st.emotion}
          title={active?.title ?? "Чат"}
          onRename={(t) => activeId && renameChat(activeId, t)}
          onStart={startDemo}
          onStop={stop}
          isActive={isActive}
        />

        <div className="flex-1 overflow-y-auto scrollbar-thin tg-wallpaper">
          <div className="max-w-3xl mx-auto min-h-full flex flex-col">
            <TranscriptPanel segments={st.transcript} onOpenSource={openArticle} />
          </div>
        </div>

        <div className="shrink-0 border-t border-border bg-card px-3 sm:px-6 py-3">
          <div className="max-w-3xl mx-auto space-y-3">
            <SuggestionsPanel
              sets={active?.pending ?? []}
              demo={active?.demo}
              onSend={(text, sources, setId) => {
                if (active?.demo) return;
                commitOperator(text, sources);
                if (activeId) dismissSet(activeId, setId);
              }}
              onOpenSource={openArticle}
              onDismiss={(setId) => activeId && dismissSet(activeId, setId)}
            />
            <MicRecorder
              onAudio={sendVoice}
              disabled={!connected}
              busy={isActive}
              speaker={voiceSpeaker}
              onSpeakerChange={changeSpeaker}
            />
          </div>
        </div>
      </main>

      <aside className="hidden xl:flex w-[380px] shrink-0 flex-col border-l border-border bg-card">
        <div className="h-14 shrink-0 flex items-center px-4 border-b border-border">
          <span className="font-semibold text-[15px]">Сведения о звонке</span>
        </div>
        <div className="flex-1 overflow-y-auto scrollbar-thin p-4 space-y-4">
          <div className="glass p-4">
            <p className="eyebrow mb-2">База знаний оператора</p>
            <select
              value={active?.company ?? "mts"}
              onChange={(e) => activeId && setCompany(activeId, e.target.value)}
              className="w-full bg-secondary text-foreground text-sm rounded-lg px-3 py-2 outline-none border border-border focus:border-primary/60"
            >
              <option value="mts">МТС</option>
              <option value="beeline">Билайн</option>
              <option value="megafon">МегаФон</option>
              <option value="tele2">Tele2</option>
              <option value="rostelecom">Ростелеком</option>
              <option value="vektor">Вектор (синтетическая, 90 док)</option>
            </select>
            <p className="text-xs text-muted-foreground mt-2">
              {active?.company === "vektor"
                ? "Синтетическая база вымышленного оператора «Вектор»"
                : "Поиск только в документах выбранного оператора"}
            </p>
          </div>
          <AsrSelector />
          <EmotionPanel emotion={st.emotion} />
          <LatencyPanel latency={st.latency} />
          <SerUploadPanel />
        </div>
      </aside>

      {article && (
        <>
          <div
            className="fixed inset-0 bg-black/50 z-40"
            onClick={() => setArticle(null)}
          />
          <aside className="fixed right-0 top-0 h-full w-full sm:w-[440px] bg-card border-l border-border z-50 flex flex-col shadow-2xl animate-slide_up">
            <div className="h-14 shrink-0 flex items-center justify-between gap-2 px-4 border-b border-border">
              <div className="flex items-center gap-2 min-w-0">
                <BookOpen className="w-4 h-4 text-primary shrink-0" />
                <span className="font-semibold text-sm truncate">
                  {article.title || "Статья базы знаний"}
                </span>
              </div>
              <button
                type="button"
                onClick={() => setArticle(null)}
                className="w-8 h-8 rounded-full flex items-center justify-center text-muted-foreground hover:bg-secondary hover:text-foreground transition"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="shrink-0 flex gap-1.5 px-4 pt-3">
              {([["original", "Оригинал"], ["doc", "Переработанный"], ["instruction", "Инструкция"]] as const).map(
                ([id, label]) => (
                  <button
                    key={id}
                    type="button"
                    onClick={() => setArticleView(id)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                      articleView === id
                        ? "bg-primary text-primary-foreground"
                        : "bg-secondary text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {label}
                  </button>
                ),
              )}
            </div>
            <div className="flex-1 overflow-y-auto scrollbar-thin p-5">
              {articleLoading && !article.text ? (
                <div className="text-sm text-muted-foreground">Загрузка статьи…</div>
              ) : articleView === "original" ? (
                article.original ? (
                  <>
                    {article.source_url && (
                      <a
                        href={article.source_url}
                        target="_blank"
                        rel="noreferrer"
                        className="inline-block mb-3 text-xs text-primary hover:underline break-all"
                      >
                        Источник: {article.source_url}
                      </a>
                    )}
                    <pre className="whitespace-pre-wrap break-words font-sans text-[14px] leading-relaxed text-foreground">
                      {article.original}
                    </pre>
                  </>
                ) : (
                  <div className="text-sm text-muted-foreground">Оригинальный текст готовится</div>
                )
              ) : articleView === "instruction" ? (
                article.instruction ? (
                  <pre className="whitespace-pre-wrap break-words font-sans text-[14px] leading-relaxed text-foreground">
                    {article.instruction}
                  </pre>
                ) : (
                  <div className="text-sm text-muted-foreground">
                    Для этого документа нет готовой инструкции с ответами
                  </div>
                )
              ) : (
                <pre className="whitespace-pre-wrap break-words font-sans text-[14px] leading-relaxed text-foreground">
                  {article.text || "Текст статьи недоступен"}
                </pre>
              )}
            </div>
            <div className="shrink-0 px-4 py-2.5 border-t border-border text-[11px] text-muted-foreground">
              {article.doc_id}
            </div>
          </aside>
        </>
      )}

      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        settings={settings}
        update={updateSettings}
      />
    </div>
  );
}
