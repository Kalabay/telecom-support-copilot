"""V/A-регрессия (audeering MSP-dim) на Dusha crowd: silver arousal/valence/dominance."""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import app  # noqa: F401,E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import librosa  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import transformers.modeling_utils as mu  # noqa: E402

mu.PreTrainedModel.all_tied_weights_keys = {}
from transformers import Wav2Vec2Processor  # noqa: E402
from transformers.models.wav2vec2.modeling_wav2vec2 import (  # noqa: E402
    Wav2Vec2Model, Wav2Vec2PreTrainedModel)

R = PROJECT_ROOT / "eval" / "results"
SETUPS = Path(r"K:\.caches\dusha_setups\paper_setups\test")
CLASSES = ["neutral", "angry", "positive", "sad"]
C2I = {c: i for i, c in enumerate(CLASSES)}
MID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"


class Head(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.final_dropout)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.num_labels)

    def forward(self, x):
        x = self.dropout(x); x = torch.tanh(self.dense(x)); x = self.dropout(x)
        return self.out_proj(x)


class VADModel(Wav2Vec2PreTrainedModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.wav2vec2 = Wav2Vec2Model(cfg); self.classifier = Head(cfg); self.init_weights()

    def forward(self, x):
        return self.classifier(self.wav2vec2(x)[0].mean(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = Wav2Vec2Processor.from_pretrained(MID)
    m = VADModel.from_pretrained(MID).to(dev).eval()
    print("V/A model loaded", flush=True)

    man_path = Path(r"K:\.caches\dusha_test\manifest.jsonl")
    recs = [json.loads(l) for l in man_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    rows = []
    t0 = time.perf_counter(); cnt = 0
    for rec in recs:
        if cnt >= args.n:
            break
        fid, emo = rec["id"], rec["emotion"]
        if emo not in C2I:
            continue
        try:
            w, sr = sf.read(rec["path"], dtype="float32")
            if w.ndim > 1:
                w = w.mean(1)
            if sr != 16000:
                w = librosa.resample(w, orig_sr=sr, target_sr=16000)
            x = proc(w, sampling_rate=16000, return_tensors="pt").input_values.to(dev)
            with torch.inference_mode():
                a, d, v = m(x)[0].cpu().numpy().tolist()
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {fid}: {exc}", flush=True); continue
        rows.append({"id": fid, "emotion": emo,
                     "arousal": round(a, 4), "dominance": round(d, 4), "valence": round(v, 4)})
        cnt += 1
        if cnt % 200 == 0:
            print(f"  {cnt}/{args.n}  {time.perf_counter()-t0:.0f}s", flush=True)

    (R / "vad_crowd.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(f"saved -> vad_crowd.json ({len(rows)})", flush=True)


if __name__ == "__main__":
    main()
