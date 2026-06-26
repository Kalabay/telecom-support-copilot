export type Emotion = "neutral" | "angry" | "positive" | "sad" | "other";

export type PipelineStage =
  | "idle"
  | "listening"
  | "transcribing"
  | "analyzing"
  | "retrieving"
  | "generating"
  | "ready";

export interface EmotionState {
  label: Emotion;
  confidence: number;
  arousal: number;
  valence: number;
  escalation_risk: boolean;
}

export interface TranscriptSegment {
  text: string;
  speaker: "customer" | "operator";
  start_ms: number;
  end_ms: number;
  confidence: number;
  sources?: KBSource[];
}

export interface KBSource {
  doc_id: string;
  title: string;
  snippet: string;
  score: number;
}

export interface Suggestion {
  text: string;
  rank: number;
  sources: KBSource[];
  intent?: string;
}

export interface LatencyBreakdown {
  asr_ms: number;
  ser_ms: number;
  retrieval_ms: number;
  llm_ms: number;
  total_ms: number;
}

export interface CopilotUpdate {
  timestamp: string;
  transcript: TranscriptSegment[];
  emotion: EmotionState | null;
  suggestions: Suggestion[];
  latency: LatencyBreakdown;
  pipeline_stage: PipelineStage;
  audio_url?: string | null;
}
