"""Устойчивый к обрывам загрузчик файлов (HTTP Range + ретраи)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

FILES = ["model.bin", "config.json", "tokenizer.json", "vocabulary.txt"]


def file_size(client: httpx.Client, url: str) -> int:
    r = client.head(url, follow_redirects=True, timeout=30)
    r.raise_for_status()
    return int(r.headers.get("content-length", 0))


def resumable_download(client: httpx.Client, url: str, dest: Path, total: int) -> None:
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
                               timeout=httpx.Timeout(30.0, read=60.0)) as r:
                if r.status_code not in (200, 206):
                    raise httpx.HTTPError(f"status {r.status_code}")
                mode = "ab" if have else "wb"
                with open(dest, mode) as f:
                    last = time.time()
                    for chunk in r.iter_bytes(chunk_size=1 << 20):
                        f.write(chunk)
                        now = time.time()
                        if now - last > 5:
                            sz = dest.stat().st_size
                            pct = (sz / total * 100) if total else 0
                            print(f"    {dest.name}: {sz/1e6:.0f}/{total/1e6:.0f} MB "
                                  f"({pct:.0f}%)", flush=True)
                            last = now
        except Exception as exc:  # noqa: BLE001
            print(f"    [retry {attempt}] {dest.name}: {type(exc).__name__} {exc}; "
                  f"докачиваю с {dest.stat().st_size if dest.exists() else 0} б", flush=True)
            time.sleep(min(5 + attempt, 20))


def main() -> None:
    repo = sys.argv[1]
    out = Path(sys.argv[2])
    base = f"https://huggingface.co/{repo}/resolve/main"
    print(f"Качаю {repo} -> {out}")
    with httpx.Client() as client:
        for fname in FILES:
            url = f"{base}/{fname}"
            dest = out / fname
            try:
                total = file_size(client, url)
            except Exception as exc:  # noqa: BLE001
                print(f"  {fname}: нет файла ({exc}), пропускаю")
                continue
            if dest.exists() and total and dest.stat().st_size >= total:
                print(f"  {fname}: уже есть ({total/1e6:.0f} MB)")
                continue
            print(f"  {fname}: {total/1e6:.0f} MB")
            t0 = time.time()
            resumable_download(client, url, dest, total)
            print(f"  {fname}: готово за {time.time()-t0:.0f}s", flush=True)
    print("ВСЁ СКАЧАНО")


if __name__ == "__main__":
    main()
