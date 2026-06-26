"""Скачать GGUF моделей-кандидатов (устойчиво к обрывам, с докачкой)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx
from huggingface_hub import list_repo_files

DEST = Path(r"K:\dev\coursework\.models")
MODELS = [
    ("qwen3_14b", "Qwen/Qwen3-14B-GGUF", ["Q4_K_M"]),
    ("vikhr_nemo12", "bartowski/Vikhr-Nemo-12B-Instruct-R-21-09-24-GGUF", ["Q4_K_M", "Q4_K_S"]),
    ("gigachat20_v15", "ai-sage/GigaChat-20B-A3B-instruct-v1.5-GGUF",
     ["Q4_K_M", "Q4_K_S", "q4_k_m", "Q4_0", "q4"]),
]


def pick_file(repo: str, quants: list[str]) -> str | None:
    files = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
    single = [f for f in files if "of-0" not in f]
    pool = single or files
    for q in quants:
        for f in pool:
            if q.lower() in f.lower():
                return f
    return pool[0] if pool else None


def resumable(client: httpx.Client, url: str, dest: Path, total: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        have = dest.stat().st_size if dest.exists() else 0
        if total and have >= total:
            return
        attempt += 1
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with client.stream("GET", url, headers=headers, follow_redirects=True,
                               timeout=httpx.Timeout(30.0, read=120.0)) as r:
                if r.status_code not in (200, 206):
                    raise httpx.HTTPError(f"status {r.status_code}")
                with open(dest, "ab" if have else "wb") as f:
                    last = time.time()
                    for chunk in r.iter_bytes(chunk_size=1 << 20):
                        f.write(chunk)
                        if time.time() - last > 8:
                            sz = dest.stat().st_size
                            print(f"    {dest.name}: {sz/1e9:.1f}/{total/1e9:.1f} ГБ "
                                  f"({sz/total*100:.0f}%)", flush=True)
                            last = time.time()
        except Exception as exc:  # noqa: BLE001
            print(f"    [retry {attempt}] {type(exc).__name__}: докачиваю с "
                  f"{dest.stat().st_size if dest.exists() else 0} б", flush=True)
            time.sleep(min(5 + attempt, 20))


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    with httpx.Client() as client:
        for name, repo, quants in MODELS:
            print(f"\n=== {name} <- {repo} ===", flush=True)
            try:
                fname = pick_file(repo, quants)
            except Exception as exc:  # noqa: BLE001
                print(f"  не смог получить список файлов: {exc}", flush=True)
                continue
            if not fname:
                print("  GGUF не найден", flush=True)
                continue
            url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
            dest = DEST / f"{name}.gguf"
            try:
                total = int(client.head(url, follow_redirects=True, timeout=30)
                            .headers.get("content-length", 0))
            except Exception as exc:  # noqa: BLE001
                print(f"  HEAD не прошёл: {exc}", flush=True)
                continue
            if dest.exists() and total and dest.stat().st_size >= total:
                print(f"  уже скачан: {fname} ({total/1e9:.1f} ГБ)", flush=True)
                continue
            print(f"  файл: {fname} ({total/1e9:.1f} ГБ) -> {dest.name}", flush=True)
            t0 = time.time()
            resumable(client, url, dest, total)
            print(f"  готово за {time.time()-t0:.0f}s", flush=True)
    print("\nВСЕ КАНДИДАТЫ СКАЧАНЫ")


if __name__ == "__main__":
    main()
