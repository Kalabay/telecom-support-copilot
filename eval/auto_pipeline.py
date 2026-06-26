"""Автоконвейер: для каждой модели — скачать GGUF (с докачкой) и сразу забенчмаркать."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", r"K:\dev\coursework\.hf_cache")
os.environ.setdefault("TORCH_HOME", r"K:\dev\coursework\.caches\torch")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY = PROJECT_ROOT / "backend" / ".venv" / "Scripts" / "python.exe"
R = PROJECT_ROOT / "eval" / "results"

from huggingface_hub import hf_hub_download  # noqa: E402

MODELS = [
    ("qwen3_14b", "Qwen/Qwen3-14B-GGUF", "Qwen3-14B-Q4_K_M.gguf"),
    ("vikhr_nemo12", "bartowski/Vikhr-Nemo-12B-Instruct-R-21-09-24-GGUF",
     "Vikhr-Nemo-12B-Instruct-R-21-09-24-Q4_K_M.gguf"),
    ("gigachat20_v15", "ai-sage/GigaChat-20B-A3B-instruct-v1.5-GGUF",
     "GigaChat-20B-A3B-instruct-v1.5-q4_K_M.gguf"),
]


def download(repo: str, fname: str) -> str | None:
    att = 0
    while att < 80:
        att += 1
        try:
            return hf_hub_download(repo_id=repo, filename=fname)
        except Exception as exc:  # noqa: BLE001
            print(f"    [retry {att}] {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            time.sleep(min(10 + att * 3, 60))
    return None


def main() -> None:
    status: dict[str, str] = {}
    for name, repo, fname in MODELS:
        if (R / f"bench_{name}.json").exists():
            status[name] = "уже есть"
            print(f"=== [{name}] уже забенчмаркан, пропуск ===", flush=True)
            continue
        print(f"\n=== [{name}] скачиваю {fname} ===", flush=True)
        path = download(repo, fname)
        if not path:
            status[name] = "download_failed"
            print(f"  [{name}] НЕ скачался", flush=True)
            continue
        print(f"  готов: {path}\n=== [{name}] бенчмарк ===", flush=True)
        env = dict(os.environ); env["LLM_BACKEND"] = name
        r = subprocess.run([str(PY), str(PROJECT_ROOT / "eval" / "bench_model.py")], env=env)
        ok = (R / f"bench_{name}.json").exists() and r.returncode == 0
        status[name] = "ok" if ok else f"bench_failed(rc={r.returncode})"
        print(f"  [{name}] -> {status[name]}", flush=True)

    print("\n==================== AUTO-PIPELINE DONE ====================", flush=True)
    for k, v in status.items():
        print(f"  {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
