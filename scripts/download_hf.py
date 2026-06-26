"""Скачать GGUF кандидатов через hf_transfer (параллельно, с докачкой) в HF-кэш на K:."""
from __future__ import annotations

import os
import time

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ.setdefault("HF_HOME", r"K:\dev\coursework\.hf_cache")

from huggingface_hub import hf_hub_download  # noqa: E402

MODELS = [
    ("Qwen/Qwen3-14B-GGUF", "Qwen3-14B-Q4_K_M.gguf"),
    ("bartowski/Vikhr-Nemo-12B-Instruct-R-21-09-24-GGUF",
     "Vikhr-Nemo-12B-Instruct-R-21-09-24-Q4_K_M.gguf"),
    ("ai-sage/GigaChat-20B-A3B-instruct-v1.5-GGUF",
     "GigaChat-20B-A3B-instruct-v1.5-q4_K_M.gguf"),
]


def main() -> None:
    for repo, fname in MODELS:
        print(f"\n=== {repo} / {fname} ===", flush=True)
        attempt = 0
        while True:
            attempt += 1
            try:
                t0 = time.time()
                path = hf_hub_download(repo_id=repo, filename=fname)
                print(f"  OK ({time.time()-t0:.0f}s): {path}", flush=True)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  [retry {attempt}] {type(exc).__name__}: {str(exc)[:160]}", flush=True)
                if attempt >= 30:
                    print("  СДАЮСЬ после 30 попыток", flush=True)
                    break
                time.sleep(min(5 + attempt * 2, 30))
    print("\nЗАКАЧКА ЗАВЕРШЕНА", flush=True)


if __name__ == "__main__":
    main()
