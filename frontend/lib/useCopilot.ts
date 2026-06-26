"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import type { CopilotUpdate } from "./types";

const EMPTY: CopilotUpdate = {
  timestamp: new Date().toISOString(),
  transcript: [],
  emotion: null,
  suggestions: [],
  latency: { asr_ms: 0, ser_ms: 0, retrieval_ms: 0, llm_ms: 0, total_ms: 0 },
  pipeline_stage: "idle",
};

const WS_URL =
  process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws/copilot";

export function useCopilot() {
  const [state, setState] = useState<CopilotUpdate>(EMPTY);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let stopped = false;
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);
    ws.onmessage = (e) => {
      try {
        const update = JSON.parse(e.data) as CopilotUpdate;
        if (update.transcript !== undefined) setState(update);
      } catch {
      }
    };

    return () => {
      stopped = true;
      ws.close();
    };
  }, []);

  const startDemo = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ type: "demo_trigger" }));
  }, []);

  const stop = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ type: "stop" }));
  }, []);

  const sendVoice = useCallback((blob: Blob) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(blob);
  }, []);

  const commitOperator = useCallback((text: string) => {
    wsRef.current?.send(
      JSON.stringify({ type: "operator_said", payload: { text } }),
    );
  }, []);

  const setVoiceSpeaker = useCallback((speaker: "customer" | "operator") => {
    wsRef.current?.send(
      JSON.stringify({ type: "voice_speaker", payload: { speaker } }),
    );
  }, []);

  return {
    state,
    connected,
    startDemo,
    stop,
    sendVoice,
    commitOperator,
    setVoiceSpeaker,
  };
}
