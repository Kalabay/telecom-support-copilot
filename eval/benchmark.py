"""ЕДИНЫЙ бенчмарк системы — все метрики по всем компонентам в один отчёт."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import app  # noqa: F401,E402

import numpy as np  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

R = PROJECT_ROOT / "eval" / "results"
PY = sys.executable



def _load_preds(name: str):
    p = R / name
    if not p.exists():
        return None
    obj = json.loads(p.read_text(encoding="utf-8"))
    return {str(r["id"]): r for r in obj["preds"]}


def _macro_f1_from_probs(preds: dict) -> float:
    ids = sorted(preds)
    y = np.array([preds[i]["true"] for i in ids])
    yp = np.array([int(np.argmax(preds[i]["probs"])) for i in ids])
    return round(float(f1_score(y, yp, average="macro", zero_division=0)), 4)


def ser_metrics() -> dict:
    out = {}
    single = {
        "GigaAM": "_pred_ga_crowd.json",
        "HuBERT": "_pred_hu_crowd.json",
        "наша FT-модель": "_pred_ours_crowd.json",
    }
    for name, f in single.items():
        pr = _load_preds(f)
        if pr:
            out[f"SER {name} (Dusha crowd, macro-F1)"] = _macro_f1_from_probs(pr)

    ga, hu = _load_preds("_pred_ga_crowd.json"), _load_preds("_pred_hu_crowd.json")
    if ga and hu:
        ids = sorted(set(ga) & set(hu))
        y = np.array([ga[i]["true"] for i in ids])
        Pga = np.array([ga[i]["probs"] for i in ids])
        Phu = np.array([hu[i]["probs"] for i in ids])
        yp = (0.5 * Pga + 0.5 * Phu).argmax(1)
        out["SER ансамбль GigaAM+HuBERT (среднее, macro-F1)"] = round(
            float(f1_score(y, yp, average="macro", zero_division=0)), 4)
    return out


def ser_crossdataset() -> dict:
    out = {}
    ga, hu = _load_preds("emb_gigaam_resdtest.npz"), None
    return out



def rag_metrics(model: str, prefix_q: str, prefix_d: str, reranker: str) -> dict:
    cmd = [PY, str(PROJECT_ROOT / "eval" / "eval_rag.py"), "--model", model,
           "--out", str(R / "_bench_rag_tmp.json")]
    if prefix_q:
        cmd += ["--prefix-query", prefix_q]
    if prefix_d:
        cmd += ["--prefix-doc", prefix_d]
    if reranker:
        cmd += ["--reranker", reranker]
    (R / "_bench_rag_tmp.json").unlink(missing_ok=True)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=900)
    except Exception as exc:  # noqa: BLE001
        return {"RAG": f"ОШИБКА: {exc}"}
    runs = json.loads((R / "_bench_rag_tmp.json").read_text(encoding="utf-8"))
    r = runs[-1]
    return {
        f"RAG Recall@5 ({model})": r["recall@5"],
        f"RAG Recall@3 ({model})": r["recall@3"],
        f"RAG MRR ({model})": r["mrr"],
        f"RAG latency p95 ms ({model})": r["latency_ms_p95"],
    }



def pulled_metrics() -> dict:
    out = {}
    abl = R / "ablation_emotion.json"
    if abl.exists():
        s = json.loads(abl.read_text(encoding="utf-8")).get("summary", {})
        allg = s.get("все", {})
        if allg:
            out["LLM empathy ON/OFF (все)"] = f"{allg['on']['empathy_rate']:.0%} / {allg['off']['empathy_rate']:.0%}"
            out["LLM action ON/OFF (все)"] = f"{allg['on']['action_rate']:.0%} / {allg['off']['action_rate']:.0%}"
    asr = R / "asr_wer.json"
    if asr.exists():
        w = json.loads(asr.read_text(encoding="utf-8"))
        out["ASR WER"] = w.get("wer", "?")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="метка прогона: baseline / after-frida / ...")
    ap.add_argument("--rag-model", default="BAAI/bge-m3")
    ap.add_argument("--rag-prefix-query", default="")
    ap.add_argument("--rag-prefix-doc", default="")
    ap.add_argument("--rag-reranker", default="")
    ap.add_argument("--skip-rag", action="store_true")
    args = ap.parse_args()

    print(f"=== BENCHMARK [{args.tag}] ===", flush=True)
    metrics: dict = {}
    print("SER…", flush=True)
    metrics.update(ser_metrics())
    if not args.skip_rag:
        print("RAG…", flush=True)
        metrics.update(rag_metrics(args.rag_model, args.rag_prefix_query,
                                    args.rag_prefix_doc, args.rag_reranker))
    metrics.update(pulled_metrics())

    print(f"\n=== РЕЗУЛЬТАТЫ [{args.tag}] ===")
    for k, v in metrics.items():
        print(f"  {k:52s} {v}")

    journal = R / "benchmark_report.json"
    runs = []
    if journal.exists():
        try:
            runs = json.loads(journal.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            runs = []
    runs.append({"tag": args.tag, "metrics": metrics})
    journal.write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# Бенчмарк системы — сводка прогонов\n",
          "Все метрики по компонентам. Сравнивай столбцы (теги) до/после свопа.\n"]
    all_keys = []
    for run in runs:
        for k in run["metrics"]:
            if k not in all_keys:
                all_keys.append(k)
    tags = [r["tag"] for r in runs]
    md.append("| Метрика | " + " | ".join(tags) + " |")
    md.append("|" + "---|" * (len(tags) + 1))
    for k in all_keys:
        row = [k] + [str(r["metrics"].get(k, "—")) for r in runs]
        md.append("| " + " | ".join(row) + " |")
    (R / "benchmark_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nsaved -> {journal}\nsaved -> {R / 'benchmark_report.md'}")


if __name__ == "__main__":
    main()
