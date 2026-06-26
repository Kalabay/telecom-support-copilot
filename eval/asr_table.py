"""Сводная ASR-таблица: движок x набор (WER)."""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "eval" / "results"
NAMEMAP = {
    "large-v3-turbo": "Whisper large-v3-turbo",
    "medium": "Whisper medium",
    "gigaam-v2": "GigaAM-v2 (Сбер)",
    "gigaam-v2-ctc": "GigaAM-v2 (Сбер)",
    "t-tech/t-one": "T-one (Т-банк)",
}
data = {}
for f in list(R.glob("asr_*.json")) + [R / "asr_cmp_whisper_synth.json"]:
    if not f.exists():
        continue
    d = json.loads(f.read_text(encoding="utf-8"))
    if "wer" not in d:
        continue
    model = d.get("model") or "large-v3-turbo"
    name = NAMEMAP.get(model, model)
    dset = d.get("set", "synth")
    data.setdefault(name, {})[dset] = (d["wer"], d.get("n"))

order = ["Whisper large-v3-turbo", "Whisper medium", "GigaAM-v2 (Сбер)", "T-one (Т-банк)"]
print(f"{'Движок':28s} | {'синтетика':>12s} | {'Dusha (реальная)':>16s}")
print("-" * 62)
for name in order + [k for k in data if k not in order]:
    if name not in data:
        continue
    s = data[name].get("synth"); du = data[name].get("dusha")
    sv = f"{s[0]:.4f}" if s else "—"
    dv = f"{du[0]:.4f}" if du else "—"
    print(f"{name:28s} | {sv:>12s} | {dv:>16s}")
