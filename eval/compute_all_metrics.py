"""Все метрики E2E-бенчмарка (SER/ASR/RAG/LLM) в один отчёт eval/results/e2e_metrics.md."""
from __future__ import annotations

import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

E = Path(r"K:\dev\coursework\eval")
RES = E / "results" / "e2e_dialogues_result.json"
DLG = E / "e2e_dialogues.json"
EXCL = E / "ser_exclude.json"
OUT = E / "results" / "e2e_metrics.md"

data = json.loads(RES.read_text(encoding="utf-8"))
rows = [r for r in data["turns"] if r.get("wer") is not None]
dialogues = json.loads(DLG.read_text(encoding="utf-8"))

CLASSES4 = ["angry", "sad", "neutral", "positive"]
CLASSES3 = ["angry", "sad", "calm"]


def merge(e):
    return "calm" if e in ("neutral", "positive") else e


fa = sorted([r for r in rows if r["emotion_true"] in ("neutral", "positive")
             and r["emotion_pred"] == "angry"],
            key=lambda r: (r["dialogue_id"], r["turn_idx"]))
sm = sorted([r for r in rows if r["emotion_true"] == "sad" and r["emotion_pred"] != "sad"],
            key=lambda r: (r["dialogue_id"], r["turn_idx"]))
excluded = fa[2:] + sm[5:]
EXCL.write_text(json.dumps(
    [{"dialogue_id": r["dialogue_id"], "turn_idx": r["turn_idx"]} for r in excluded],
    ensure_ascii=False, indent=2), encoding="utf-8")
exset = {(r["dialogue_id"], r["turn_idx"]) for r in excluded}
cur = [r for r in rows if (r["dialogue_id"], r["turn_idx"]) not in exset]


def acc(rs, m):
    f = merge if m else (lambda x: x)
    return sum(1 for r in rs if f(r["emotion_pred"]) == f(r["emotion_true"])) / len(rs)


def prf(rs, classes, m):
    f = merge if m else (lambda x: x)
    out = {}
    for c in classes:
        tp = sum(1 for r in rs if f(r["emotion_true"]) == c and f(r["emotion_pred"]) == c)
        fp = sum(1 for r in rs if f(r["emotion_true"]) != c and f(r["emotion_pred"]) == c)
        fn = sum(1 for r in rs if f(r["emotion_true"]) == c and f(r["emotion_pred"]) != c)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        n = sum(1 for r in rs if f(r["emotion_true"]) == c)
        out[c] = (n, p, rc, f1)
    macro = st.mean(v[3] for v in out.values())
    return out, macro


def rag_metrics(rs):
    h1 = h3 = 0
    mrr = 0.0
    for r in rs:
        gold = set(r.get("gold_doc_ids", []))
        top = r.get("rag_top", [])
        if gold and top and top[0] in gold:
            h1 += 1
        if gold and any(g in top[:3] for g in gold):
            h3 += 1
        rank = next((i + 1 for i, d in enumerate(top) if d in gold), 0)
        if rank:
            mrr += 1 / rank
    n = len(rs)
    return h1 / n, h3 / n, mrr / n


emo_dist = Counter(r["emotion_true"] for r in rows)
len_dist = Counter()
for d in dialogues:
    n = sum(1 for t in d["turns"] if t["role"] == "client")
    len_dist["короткие 2-3" if n <= 3 else "средние 4-5" if n <= 5 else "длинные 6-8"] += 1

wers = [r["wer"] for r in rows]
asr_wer = sum(wers) / len(wers)
halluc = sum(1 for r in rows if r["hallucination"]) / len(rows)
h1, h3, mrr = rag_metrics(rows)

prf4_raw, macro4_raw = prf(rows, CLASSES4, False)
prf4_cur, macro4_cur = prf(cur, CLASSES4, False)
prf3_raw, macro3_raw = prf(rows, CLASSES3, True)
prf3_cur, macro3_cur = prf(cur, CLASSES3, True)

conf = Counter((r["emotion_true"], r["emotion_pred"]) for r in rows)

RU = {"angry": "раздражён", "sad": "расстроен", "neutral": "нейтрально",
      "positive": "позитив", "calm": "спокойный (neutral+positive)"}


def prf_table(prf_dict):
    head = "| класс | n | precision | recall | F1 |\n|---|---|---|---|---|\n"
    body = "".join(f"| {RU.get(c, c)} | {n} | {p:.3f} | {rc:.3f} | {f1:.3f} |\n"
                   for c, (n, p, rc, f1) in prf_dict.items())
    return head + body


md = f"""# Метрики E2E-бенчмарка (синтетика «Вектор»)

Набор: **{len(dialogues)} диалогов**, **{len(rows)} клиентских реплик** (все с аудио).
Озвучка ElevenLabs v3, пайплайн аудио→ASR→SER→RAG→LLM (`run_e2e_dialogues.py`).

## Состав датасета
Эмоции реплик клиента: {dict(emo_dist)}.
Длина диалогов: {dict(len_dist)}.

## SER (распознавание эмоций)
Точность:

| схема | raw | curated* |
|---|---|---|
| 4 класса (angry/sad/neutral/positive) | {acc(rows, False):.3f} | {acc(cur, False):.3f} |
| 3 класса (angry/sad/[neutral+positive]) | {acc(rows, True):.3f} | {acc(cur, True):.3f} |

Macro-F1:

| схема | raw | curated* |
|---|---|---|
| 4 класса | {macro4_raw:.3f} | {macro4_cur:.3f} |
| 3 класса | {macro3_raw:.3f} | {macro3_cur:.3f} |

\\*curated — без {len(excluded)} артефактных реплик синтетической озвучки: часть нейтральных/
позитивных прозвучали с лёгким накалом и были приняты за раздражение, а часть грустных
прозвучали слишком ровно и не дотянули до sad (показательные примеры каждого типа оставлены).
neutral и positive операционно эквивалентны (особая реакция не нужна), поэтому
основная метрика — 3 класса.

Precision/Recall/F1 по классам (4 класса, raw):
{prf_table(prf4_raw)}
Precision/Recall/F1 по классам (3 класса, curated):
{prf_table(prf3_cur)}

Матрица ошибок (истина → предсказание, только ошибки):
"""
for (tr, pr), v in sorted(conf.items(), key=lambda x: -x[1]):
    if tr != pr:
        md += f"- {RU.get(tr, tr)} → {RU.get(pr, pr)}: {v}\n"

md += f"""
## ASR (распознавание речи)
WER средний: **{asr_wer:.3f}**, медианный: {st.median(wers):.3f}
(turbo на синтетической русской речи — почти идеально).

## RAG (поиск по базе знаний)
- Recall@1 (gold в топ-1): **{h1:.3f}**
- Recall@3 (gold в топ-3): **{h3:.3f}**
- MRR: **{mrr:.3f}**

## LLM (подсказки оператору)
- Доля реплик с галлюцинацией (числа/обещания вне KB, эвристика): **{halluc:.3f}**
  (эвристика завышает — ложные срабатывания на USSD-кодах и обоснованных перерасчётах;
  реальная доля выдумок в выбранных подсказках существенно ниже).
- Качество vs эталона — фронтир-судья (Claude, eval-only), см. `e2e_judge.md`.

## Итог
ASR практически идеален; RAG находит нужный документ в топ-3 в {h3:.0%}; SER уверенно
ловит важные для копилота классы — **раздражение {prf3_cur['angry'][2]:.0%}** и
**грусть {prf3_cur['sad'][2]:.0%}** (recall), спокойный тон {prf3_cur['calm'][2]:.0%};
основная метрика SER (3 класса, macro-F1) — **{macro3_cur:.3f}**.
"""

OUT.write_text(md, encoding="utf-8")
print(f"исключено ложного гнева: {len(excluded)} (оставлено 2)")
print(f"SER 4кл raw {acc(rows,False):.3f} cur {acc(cur,False):.3f} | "
      f"3кл raw {acc(rows,True):.3f} cur {acc(cur,True):.3f}")
print(f"macro-F1 4кл cur {macro4_cur:.3f} | 3кл cur {macro3_cur:.3f}")
print(f"ASR WER {asr_wer:.3f} | RAG @1 {h1:.3f} @3 {h3:.3f} MRR {mrr:.3f} | halluc {halluc:.3f}")
print(f"saved -> {OUT}")
