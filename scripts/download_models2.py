"""Скачивание моделей из Deep Research: эмбеддеры (целиком) + большие LLM (GGUF)."""
from __future__ import annotations

import os
import time

os.environ.setdefault("HF_HOME", r"K:\dev\coursework\.hf_cache")

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download  # noqa: E402

EMBED = [
    "deepvk/USER2-base",
    "deepvk/USER-bge-m3",
    "Qwen/Qwen3-Embedding-0.6B",
    "ai-sage/Giga-Embeddings-instruct",
]
GGUF = [
    ("t-tech/T-pro-it-2.1-GGUF", "q3_k_m"),
    ("bartowski/Mistral-Small-3.2-24B-Instruct-2506-GGUF", "q4_k_m"),
    ("unsloth/gemma-3-27b-it-GGUF", "q3_k_m"),
    ("unsloth/Qwen3-30B-A3B-GGUF", "q4_k_m"),
]


def retry(fn, *a, **k):
    for att in range(40):
        try:
            return fn(*a, **k)
        except Exception as e:  # noqa: BLE001
            print(f"    retry {att}: {type(e).__name__} {str(e)[:110]}", flush=True)
            time.sleep(min(8 + att * 3, 60))
    return None


def main() -> None:
    for repo in EMBED:
        print(f"\n=== ЭМБЕДДЕР {repo} ===", flush=True)
        p = retry(snapshot_download, repo_id=repo)
        print(f"  {'ok: ' + str(p) if p else 'FAIL'}", flush=True)

    for repo, q in GGUF:
        print(f"\n=== LLM {repo} [{q}] ===", flush=True)
        files = retry(list_repo_files, repo) or []
        ggufs = [f for f in files if f.lower().endswith(".gguf")
                 and q in f.lower() and "mmproj" not in f.lower()]
        if not ggufs:
            print(f"  нет {q}; gguf-файлы:", [f for f in files if f.endswith('.gguf')][:6], flush=True)
            continue
        for f in ggufs:
            print(f"  файл {f}", flush=True)
            path = retry(hf_hub_download, repo_id=repo, filename=f)
            print(f"    {'ok' if path else 'FAIL'}", flush=True)

    print("\n==================== ВСЁ СКАЧАНО ====================", flush=True)


if __name__ == "__main__":
    main()
