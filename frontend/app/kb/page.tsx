"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { ArrowLeft, FileText, Search, Upload } from "lucide-react";

import { Markdown } from "@/components/Markdown";
import { cn } from "@/lib/utils";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const COMPANY: Record<string, string> = {
  mts: "МТС", beeline: "Билайн", megafon: "МегаФон",
  tele2: "Tele2", rostelecom: "Ростелеком", vektor: "Вектор (синт.)",
};
const ORDER = ["mts", "beeline", "megafon", "tele2", "rostelecom", "vektor"];

interface Doc {
  doc_id: string;
  title: string;
  company: string;
}

export default function KbPage() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [sel, setSel] = useState<string | null>(null);
  const [article, setArticle] = useState<{ title: string; text: string; company: string; instruction?: string; original?: string; source_url?: string } | null>(null);
  const [view, setView] = useState<"original" | "doc" | "instruction">("original");
  const [loading, setLoading] = useState(false);
  const [q, setQ] = useState("");

  const [file, setFile] = useState<File | null>(null);
  const [upCompany, setUpCompany] = useState("");
  const [uploading, setUploading] = useState(false);
  const [upMsg, setUpMsg] = useState<string | null>(null);

  const loadDocs = () =>
    fetch(`${API_BASE}/api/kb/list`)
      .then((r) => r.json())
      .then((d) => setDocs(d.documents || []))
      .catch(() => {});

  useEffect(() => {
    loadDocs();
  }, []);

  const upload = async () => {
    if (!file) return;
    setUploading(true);
    setUpMsg(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      if (upCompany.trim()) fd.append("company", upCompany.trim());
      const r = await fetch(`${API_BASE}/api/kb/upload`, { method: "POST", body: fd });
      const d = await r.json();
      if (r.ok) {
        setUpMsg(`✓ добавлено ${d.doc_ids?.length ?? 0} док (${d.chunks_added} фрагм.)`);
        setFile(null);
        setUpCompany("");
        await loadDocs();
      } else {
        setUpMsg(`✗ ${d.detail ?? "ошибка загрузки"}`);
      }
    } catch {
      setUpMsg("✗ ошибка сети");
    }
    setUploading(false);
  };

  const open = async (docId: string) => {
    setSel(docId);
    setView("original");
    setLoading(true);
    setArticle(null);
    try {
      const r = await fetch(`${API_BASE}/api/kb/doc?doc_id=${encodeURIComponent(docId)}`);
      if (r.ok) setArticle(await r.json());
    } catch {
    }
    setLoading(false);
  };

  const grouped = useMemo(() => {
    const f = docs.filter(
      (d) =>
        !q ||
        d.title.toLowerCase().includes(q.toLowerCase()) ||
        d.doc_id.toLowerCase().includes(q.toLowerCase()),
    );
    const g: Record<string, Doc[]> = {};
    for (const d of f) (g[d.company || "—"] ??= []).push(d);
    const companies = Object.keys(g).sort(
      (a, b) => (ORDER.indexOf(a) + 1 || 99) - (ORDER.indexOf(b) + 1 || 99),
    );
    return { g, companies };
  }, [docs, q]);

  return (
    <div className="h-screen flex bg-background text-foreground overflow-hidden">
      <aside className="w-[330px] shrink-0 flex flex-col border-r border-border bg-card">
        <div className="h-14 shrink-0 flex items-center gap-3 px-4 border-b border-border">
          <Link href="/" className="text-muted-foreground hover:text-foreground transition" title="Назад">
            <ArrowLeft className="w-4 h-4" />
          </Link>
          <span className="font-semibold text-[15px]">База знаний</span>
          <span className="ml-auto text-xs text-muted-foreground">{docs.length} док</span>
        </div>
        <div className="p-2">
          <div className="flex items-center gap-2 px-3 py-2 rounded-full bg-secondary text-sm">
            <Search className="w-4 h-4 text-muted-foreground" />
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Поиск документа"
              className="bg-transparent outline-none flex-1 text-foreground placeholder:text-muted-foreground"
            />
          </div>
        </div>

        <div className="px-2 pb-2 border-b border-border">
          <label className="flex items-center gap-2 px-3 py-2 rounded-lg bg-secondary text-sm cursor-pointer hover:bg-secondary/80 transition">
            <Upload className="w-4 h-4 text-muted-foreground shrink-0" />
            <span className="truncate flex-1">{file ? file.name : "Загрузить .md документ"}</span>
            <input
              type="file"
              accept=".md"
              className="hidden"
              onChange={(e) => {
                setFile(e.target.files?.[0] ?? null);
                setUpMsg(null);
              }}
            />
          </label>
          {file && (
            <div className="mt-2 flex items-center gap-2">
              <input
                value={upCompany}
                onChange={(e) => setUpCompany(e.target.value)}
                placeholder="компания (опц.)"
                className="flex-1 min-w-0 px-2.5 py-1.5 rounded-lg bg-secondary text-sm outline-none placeholder:text-muted-foreground"
              />
              <button
                type="button"
                disabled={uploading}
                onClick={upload}
                className="px-3 py-1.5 rounded-lg bg-primary text-primary-foreground text-sm font-medium disabled:opacity-50 hover:opacity-90 transition shrink-0"
              >
                {uploading ? "…" : "Загрузить"}
              </button>
            </div>
          )}
          {upMsg && <div className="mt-1.5 text-[11px] text-muted-foreground">{upMsg}</div>}
        </div>

        <div className="flex-1 overflow-y-auto scrollbar-thin px-2 pb-3">
          {grouped.companies.map((c) => (
            <div key={c} className="mb-2">
              <div className="px-2 py-1.5 text-[11px] uppercase tracking-wider text-muted-foreground font-semibold">
                {COMPANY[c] ?? c} · {grouped.g[c].length}
              </div>
              {grouped.g[c].map((d) => (
                <button
                  key={d.doc_id}
                  type="button"
                  onClick={() => open(d.doc_id)}
                  className={cn(
                    "w-full flex items-start gap-2 px-2.5 py-2 rounded-lg text-left transition",
                    sel === d.doc_id ? "bg-primary text-primary-foreground" : "hover:bg-secondary",
                  )}
                >
                  <FileText className="w-3.5 h-3.5 mt-0.5 shrink-0 opacity-70" />
                  <span className="min-w-0">
                    <span className="block text-sm truncate">{d.title}</span>
                    <span
                      className={cn(
                        "block text-[11px] truncate",
                        sel === d.doc_id ? "opacity-80" : "text-muted-foreground",
                      )}
                    >
                      {d.doc_id}
                    </span>
                  </span>
                </button>
              ))}
            </div>
          ))}
        </div>
      </aside>

      <main className="flex-1 overflow-y-auto scrollbar-thin">
        {!sel ? (
          <div className="h-full flex items-center justify-center text-muted-foreground text-sm">
            Выберите документ слева, чтобы посмотреть его содержимое.
          </div>
        ) : (
          <div className="max-w-2xl mx-auto p-6">
            {loading || !article ? (
              <div className="text-muted-foreground text-sm">Загрузка…</div>
            ) : (
              <>
                <div className="flex items-center gap-2 mb-3 text-xs text-muted-foreground">
                  <span className="px-2 py-0.5 rounded-full bg-secondary">
                    {COMPANY[article.company] ?? article.company}
                  </span>
                  <span className="font-mono">{sel}</span>
                </div>
                <div className="flex gap-1.5 mb-4">
                  {([["original", "Оригинал"], ["doc", "Переработанный"], ["instruction", "Инструкция"]] as const).map(
                    ([id, label]) => (
                      <button
                        key={id}
                        type="button"
                        onClick={() => setView(id)}
                        className={cn(
                          "px-3 py-1.5 rounded-lg text-xs font-medium transition",
                          view === id
                            ? "bg-primary text-primary-foreground"
                            : "bg-secondary text-muted-foreground hover:text-foreground",
                        )}
                      >
                        {label}
                      </button>
                    ),
                  )}
                </div>
                {view === "original" ? (
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
                ) : view === "instruction" ? (
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
                  <Markdown text={article.text} />
                )}
              </>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
