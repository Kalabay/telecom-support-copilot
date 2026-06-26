import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatLatency(ms: number): string {
  if (ms < 1000) return `${ms} мс`;
  return `${(ms / 1000).toFixed(2)} с`;
}

export function emotionColor(label: string): string {
  return `hsl(var(--emotion-${label}, 220 14% 60%))`;
}

export const EMOTION_LABEL_RU: Record<string, string> = {
  neutral: "Нейтрально",
  angry: "Раздражён",
  positive: "Позитив",
  sad: "Расстроен",
  other: "Иное",
};

export const STAGE_LABEL_RU: Record<string, string> = {
  idle: "Ожидание",
  listening: "Слушаю",
  transcribing: "Распознаю речь",
  analyzing: "Анализирую эмоцию",
  retrieving: "Ищу в базе",
  generating: "Генерирую подсказку",
  ready: "Готово",
};
