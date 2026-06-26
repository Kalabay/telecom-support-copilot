"use client";

import { useEffect, useRef, useState } from "react";
import { Loader2, Mic, MicOff, Radio, Square } from "lucide-react";

import { cn } from "@/lib/utils";

interface Props {
  onAudio: (blob: Blob) => void;
  disabled?: boolean;
  busy?: boolean;
  speaker?: "customer" | "operator";
  onSpeakerChange?: (s: "customer" | "operator") => void;
}

const SPEECH_THRESHOLD = 0.045;
const SILENCE_HANGOVER_MS = 1200;
const MIN_SPEECH_MS = 350;
const MAX_UTTERANCE_MS = 15000;

type LiveState = "off" | "listening" | "speaking";

export function MicRecorder({
  onAudio,
  disabled,
  busy,
  speaker = "customer",
  onSpeakerChange,
}: Props) {
  const [live, setLive] = useState<LiveState>("off");
  const [level, setLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [lastSent, setLastSent] = useState<number>(0);
  const [mounted, setMounted] = useState(false);

  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);
  const mimeRef = useRef<string>("audio/webm");

  const speakingRef = useRef(false);
  const lastVoiceAtRef = useRef(0);
  const segmentStartRef = useRef(0);
  const hadSpeechRef = useRef(false);
  const liveRef = useRef<LiveState>("off");

  useEffect(() => {
    liveRef.current = live;
  }, [live]);

  const fullCleanup = () => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    try {
      recorderRef.current?.state !== "inactive" && recorderRef.current?.stop();
    } catch {
    }
    recorderRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    audioCtxRef.current?.close().catch(() => {});
    audioCtxRef.current = null;
    analyserRef.current = null;
    setLevel(0);
  };

  useEffect(() => () => fullCleanup(), []);
  useEffect(() => setMounted(true), []);

  const startSegment = () => {
    const stream = streamRef.current;
    if (!stream) return;
    const recorder = new MediaRecorder(stream, { mimeType: mimeRef.current });
    chunksRef.current = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size) chunksRef.current.push(e.data);
    };
    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: mimeRef.current });
      if (hadSpeechRef.current && blob.size > 2000) {
        setLastSent(Date.now());
        onAudio(blob);
      }
      if (liveRef.current !== "off") {
        hadSpeechRef.current = false;
        speakingRef.current = false;
        startSegment();
      }
    };
    recorder.start();
    recorderRef.current = recorder;
    segmentStartRef.current = performance.now();
    hadSpeechRef.current = false;
    speakingRef.current = false;
  };

  const tick = () => {
    const analyser = analyserRef.current;
    if (!analyser) return;
    const buf = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteTimeDomainData(buf);
    let sum = 0;
    for (let i = 0; i < buf.length; i++) {
      const v = (buf[i] - 128) / 128;
      sum += v * v;
    }
    const rms = Math.sqrt(sum / buf.length);
    setLevel(Math.min(1, rms * 3));

    const now = performance.now();
    const recorder = recorderRef.current;

    if (rms > SPEECH_THRESHOLD) {
      lastVoiceAtRef.current = now;
      if (!speakingRef.current) {
        speakingRef.current = true;
        hadSpeechRef.current = true;
        setLive("speaking");
      }
    } else {
      if (speakingRef.current && now - lastVoiceAtRef.current > SILENCE_HANGOVER_MS) {
        const segLen = now - segmentStartRef.current;
        if (segLen > MIN_SPEECH_MS && recorder && recorder.state !== "inactive") {
          recorder.stop();
          setLive("listening");
        }
      }
    }

    if (
      recorder &&
      recorder.state !== "inactive" &&
      now - segmentStartRef.current > MAX_UTTERANCE_MS &&
      hadSpeechRef.current
    ) {
      recorder.stop();
    }

    rafRef.current = requestAnimationFrame(tick);
  };

  const startLive = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000,
          channelCount: 1,
        },
      });
      streamRef.current = stream;
      mimeRef.current = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";

      const ctx = new AudioContext();
      audioCtxRef.current = ctx;
      const src = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      src.connect(analyser);
      analyserRef.current = analyser;

      lastVoiceAtRef.current = performance.now();
      setLive("listening");
      liveRef.current = "listening";
      startSegment();
      rafRef.current = requestAnimationFrame(tick);
    } catch (e) {
      setError(e instanceof Error ? e.message : "mic access failed");
      fullCleanup();
      setLive("off");
    }
  };

  const stopLive = () => {
    liveRef.current = "off";
    setLive("off");
    fullCleanup();
  };

  const isMicSupported =
    !mounted ||
    (typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia);

  const sinceSent = lastSent ? Math.round((Date.now() - lastSent) / 1000) : null;

  return (
    <div className="glass px-5 py-4 flex items-center gap-4 flex-wrap">
      {!isMicSupported ? (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <MicOff className="w-4 h-4" /> Микрофон не поддерживается браузером
        </div>
      ) : (
        <>
          <div className="flex-1">
            <p className="text-sm font-medium flex items-center gap-2">
              <Radio
                className={cn(
                  "w-4 h-4",
                  live === "off"
                    ? "text-muted-foreground"
                    : "text-rose-500 animate-pulse_glow",
                )}
              />
              Живой режим
            </p>
            <p className="text-xs text-muted-foreground">
              {live === "off" &&
                "Говорите в микрофон — реплика обработается автоматически"}
              {live === "listening" &&
                (busy ? "Обрабатываю предыдущую реплику…" : "Слушаю… говори, когда готов")}
              {live === "speaking" && "🔴 Речь — закончи фразу, дальше сработает автоматически"}
            </p>
          </div>

          <div className="flex items-center gap-1 p-1 rounded-xl bg-secondary shrink-0">
            <button
              onClick={() => onSpeakerChange?.("customer")}
              className={cn(
                "px-2.5 py-1 rounded-lg text-xs font-medium transition",
                speaker === "customer"
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
              title="Речь будет распознана как реплика клиента (полный пайплайн)"
            >
              Я клиент
            </button>
            <button
              onClick={() => onSpeakerChange?.("operator")}
              className={cn(
                "px-2.5 py-1 rounded-lg text-xs font-medium transition",
                speaker === "operator"
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
              title="Речь зафиксируется как реплика оператора (без генерации подсказок)"
            >
              Я оператор
            </button>
          </div>

          {live !== "off" && (
            <div className="w-24 h-2 rounded-full bg-secondary overflow-hidden">
              <div
                className={cn(
                  "h-full transition-[width] duration-75",
                  level > SPEECH_THRESHOLD * 3 ? "bg-rose-500" : "bg-primary/60",
                )}
                style={{ width: `${level * 100}%` }}
              />
            </div>
          )}

          {busy && live !== "off" && (
            <Loader2 className="w-4 h-4 animate-spin text-primary" />
          )}

          {live === "off" ? (
            <button
              onClick={startLive}
              disabled={disabled}
              className={cn(
                "inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition",
                "bg-rose-500 text-white hover:bg-rose-600",
                "disabled:opacity-40 disabled:cursor-not-allowed",
              )}
            >
              <Mic className="w-4 h-4" />
              Начать разговор
            </button>
          ) : (
            <button
              onClick={stopLive}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium bg-secondary text-secondary-foreground hover:bg-secondary/80 transition"
            >
              <Square className="w-4 h-4" />
              Завершить
            </button>
          )}

          {error && <span className="text-xs text-rose-600">{error}</span>}
        </>
      )}
    </div>
  );
}
