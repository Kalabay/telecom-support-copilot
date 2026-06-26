"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import type { CopilotUpdate, KBSource, Suggestion } from "./types";

const EMPTY: CopilotUpdate = {
  timestamp: "",
  transcript: [],
  emotion: null,
  suggestions: [],
  latency: { asr_ms: 0, ser_ms: 0, retrieval_ms: 0, llm_ms: 0, total_ms: 0 },
  pipeline_stage: "idle",
};

const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws/copilot";
const API_HTTP = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface SuggestionSet {
  id: number;
  suggestions: Suggestion[];
  replyTo: string;
}

export interface Chat {
  id: string;
  title: string;
  state: CopilotUpdate;
  connected: boolean;
  company: string;
  pending: SuggestionSet[];
  demo: boolean;
}

export function useChats() {
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeId, setActiveId] = useState<string>("");
  const wsMap = useRef<Map<string, WebSocket>>(new Map());
  const seq = useRef(0);
  const setSeq = useRef(0);
  const didInit = useRef(false);

  const patch = useCallback((id: string, p: Partial<Chat>) => {
    setChats((prev) => prev.map((c) => (c.id === id ? { ...c, ...p } : c)));
  }, []);

  const createChat = useCallback(
    (title?: string) => {
      const id = `c${seq.current++}`;
      const ws = new WebSocket(WS_URL);
      wsMap.current.set(id, ws);
      ws.binaryType = "arraybuffer";
      ws.onopen = () => {
        patch(id, { connected: true });
        ws.send(JSON.stringify({ type: "set_company", payload: { company: "mts" } }));
      };
      ws.onclose = () => patch(id, { connected: false });
      ws.onerror = () => patch(id, { connected: false });
      ws.onmessage = (e) => {
        if (typeof e.data !== "string") return;
        try {
          const update = JSON.parse(e.data) as CopilotUpdate;
          if (update.transcript === undefined) return;
          if (update.audio_url) {
            try {
              void new Audio(`${API_HTTP}${update.audio_url}`).play().catch(() => {});
            } catch {
              /* нет аудио — не страшно */
            }
          }
          setChats((prev) =>
            prev.map((c) => {
              if (c.id !== id) return c;
              const demo = c.demo && update.pipeline_stage !== "idle";
              const lastSpeaker =
                update.transcript[update.transcript.length - 1]?.speaker;
              let pending = c.pending;
              if (update.transcript.length === 0) {
                pending = [];
              } else if (demo && lastSpeaker === "operator") {
                pending = [];
              } else if (update.suggestions && update.suggestions.length > 0) {
                const replyTo =
                  [...update.transcript].reverse().find((s) => s.speaker === "customer")
                    ?.text ?? "";
                pending = [
                  { id: setSeq.current++, suggestions: update.suggestions, replyTo },
                  ...c.pending,
                ].slice(0, 8);
              }
              const emotion =
                update.transcript.length === 0
                  ? null
                  : update.emotion ?? c.state.emotion;
              return { ...c, demo, state: { ...update, emotion }, pending };
            }),
          );
        } catch {
        }
      };
      setChats((prev) => [
        ...prev,
        {
          id,
          title: title ?? `Клиент ${prev.length + 1}`,
          state: EMPTY,
          connected: false,
          company: "mts",
          pending: [],
          demo: false,
        },
      ]);
      setActiveId(id);
      return id;
    },
    [patch],
  );

  const closeChat = useCallback((id: string) => {
    wsMap.current.get(id)?.close();
    wsMap.current.delete(id);
    setChats((prev) => {
      const next = prev.filter((c) => c.id !== id);
      setActiveId((a) => (a === id ? next[next.length - 1]?.id ?? "" : a));
      return next;
    });
  }, []);

  useEffect(() => {
    if (didInit.current) return;
    didInit.current = true;
    createChat();
  }, []);

  const sendJson = useCallback(
    (obj: unknown) => {
      const ws = wsMap.current.get(activeId);
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
    },
    [activeId],
  );

  const startDemo = useCallback(() => {
    if (activeId) patch(activeId, { demo: true });
    sendJson({ type: "demo_trigger" });
  }, [sendJson, activeId, patch]);
  const stop = useCallback(() => sendJson({ type: "stop" }), [sendJson]);
  const commitOperator = useCallback(
    (text: string, sources: KBSource[] = []) =>
      sendJson({ type: "operator_said", payload: { text, sources } }),
    [sendJson],
  );
  const setVoiceSpeaker = useCallback(
    (speaker: "customer" | "operator") =>
      sendJson({ type: "voice_speaker", payload: { speaker } }),
    [sendJson],
  );
  const sendVoice = useCallback(
    (blob: Blob) => {
      const ws = wsMap.current.get(activeId);
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(blob);
    },
    [activeId],
  );

  const renameChat = useCallback(
    (id: string, title: string) => {
      const t = title.trim();
      if (t) patch(id, { title: t });
    },
    [patch],
  );

  const setCompany = useCallback(
    (id: string, company: string) => {
      patch(id, { company });
      const ws = wsMap.current.get(id);
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "set_company", payload: { company } }));
      }
    },
    [patch],
  );

  const dismissSet = useCallback((chatId: string, setId: number) => {
    setChats((prev) =>
      prev.map((c) =>
        c.id === chatId
          ? { ...c, pending: c.pending.filter((s) => s.id !== setId) }
          : c,
      ),
    );
  }, []);

  const active = chats.find((c) => c.id === activeId) ?? null;

  return {
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
  };
}
