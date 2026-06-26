"""Докачать застрявший HF-blob через Range и финализировать (blob + snapshot symlink)."""

from __future__ import annotations

import time
from pathlib import Path

import httpx

HUB = Path(r"K:\dev\coursework\.hf_cache\hub\models--t-tech--T-lite-it-2.1-GGUF")
HASH = "5ff7c6c37f3046b92ca65cd16f6161ed47d73395a9c4b9721db11e685d017a38"
FNAME = "T-lite-it-2.1-Q5_K_M.gguf"
URL = f"https://huggingface.co/t-tech/T-lite-it-2.1-GGUF/resolve/main/{FNAME}"

blob = HUB / "blobs" / HASH
inc = HUB / "blobs" / f"{HASH}.incomplete"


def main() -> None:
    src = inc if inc.exists() else blob
    with httpx.Client(follow_redirects=True) as c:
        total = int(c.head(URL, timeout=30).headers.get("content-length", 0))
        print(f"remote total: {total/1e9:.2f} GB", flush=True)
        attempt = 0
        while True:
            have = src.stat().st_size if src.exists() else 0
            print(f"have: {have/1e9:.2f} GB ({have/total*100:.1f}%)", flush=True)
            if total and have >= total:
                break
            attempt += 1
            try:
                with c.stream("GET", URL, headers={"Range": f"bytes={have}-"},
                              timeout=httpx.Timeout(30.0, read=120.0)) as r:
                    if r.status_code not in (200, 206):
                        raise httpx.HTTPError(f"status {r.status_code}")
                    with open(src, "ab") as f:
                        last = time.time()
                        for chunk in r.iter_bytes(1 << 20):
                            f.write(chunk)
                            if time.time() - last > 10:
                                sz = src.stat().st_size
                                print(f"  {sz/1e9:.2f}/{total/1e9:.2f} GB "
                                      f"({sz/total*100:.0f}%)", flush=True)
                                last = time.time()
            except Exception as exc:  # noqa: BLE001
                print(f"  [retry {attempt}] {type(exc).__name__}: {exc}; докачиваю", flush=True)
                time.sleep(min(3 + attempt, 15))

    if inc.exists():
        inc.rename(blob)
        print(f"renamed -> {blob.name}", flush=True)
    ref = (HUB / "refs" / "main").read_text().strip()
    snap = HUB / "snapshots" / ref
    snap.mkdir(parents=True, exist_ok=True)
    link = snap / FNAME
    if link.exists() or link.is_symlink():
        link.unlink()
    try:
        link.symlink_to(blob)
    except OSError:
        import shutil
        shutil.copy2(blob, link)
    print(f"FINALIZED -> {link}  ({blob.stat().st_size/1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
