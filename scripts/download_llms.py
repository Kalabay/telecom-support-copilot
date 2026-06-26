"""Закачка больших LLM (GGUF) из Deep Research — исправленные репозитории, кванты под 16 ГБ."""
from __future__ import annotations

import os
import time

os.environ.setdefault("HF_HOME", r"K:\dev\coursework\.hf_cache")

from huggingface_hub import hf_hub_download, list_repo_files  # noqa: E402
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError  # noqa: E402

GGUF = [
    ("RefalMachine/RuadaptQwen3-32B-Instruct-GGUF", "q3_k_s"),
    ("unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF", "q4_k_m"),
    ("unsloth/gemma-3-27b-it-GGUF", "q3_k_m"),
    ("unsloth/Qwen3-30B-A3B-GGUF", "q3_k_m"),
]


def retry(fn, *a, **k):
    for att in range(40):
        try:
            return fn(*a, **k)
        except (RepositoryNotFoundError, EntryNotFoundError) as e:
            print(f"    пропуск (404/нет файла): {str(e)[:80]}", flush=True)
            return None
        except Exception as e:  # noqa: BLE001
            print(f"    retry {att}: {type(e).__name__} {str(e)[:90]}", flush=True)
            time.sleep(min(8 + att * 3, 60))
    return None


def main() -> None:
    for repo, q in GGUF:
        print(f"\n=== {repo} [{q}] ===", flush=True)
        files = retry(list_repo_files, repo)
        if not files:
            continue
        ggufs = [f for f in files if f.lower().endswith(".gguf")
                 and q in f.lower() and "mmproj" not in f.lower()]
        if not ggufs:
            for alt in ("q3_k_m", "q4_k_m", "q3_k_l"):
                ggufs = [f for f in files if f.lower().endswith(".gguf")
                         and alt in f.lower() and "mmproj" not in f.lower()]
                if ggufs:
                    print(f"    {q} нет, беру {alt}", flush=True)
                    break
        if not ggufs:
            print(f"    нет подходящего кванта; gguf:", [f for f in files if f.endswith('.gguf')][:6], flush=True)
            continue
        for f in ggufs:
            print(f"    файл {f}", flush=True)
            p = retry(hf_hub_download, repo_id=repo, filename=f)
            print(f"      {'ok' if p else 'FAIL'}", flush=True)
    print("\n==================== LLM СКАЧАНЫ ====================", flush=True)


if __name__ == "__main__":
    main()
