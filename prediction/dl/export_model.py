"""Train the DL risk model once and save it for the agent (Block 2 adapter).

Reuses the training code of dl_risk.py unchanged; only the dataset path is
set to the project's copy. Saves:

    prediction/dl/models/gru.pt          model weights
    prediction/dl/models/gru_prep.json   sequence length + normalisation stats

Usage:  python prediction/dl/export_model.py [--kind GRU|LSTM] [--seed 0]
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

spec = importlib.util.spec_from_file_location("dl_risk", HERE / "dl_risk.py")
dl = importlib.util.module_from_spec(spec)
sys.modules["dl_risk"] = dl
spec.loader.exec_module(dl)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="GRU", choices=["GRU", "LSTM"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dl.CSV = str(ROOT / "data" / "synthetic_emergency_timeseries_2000_patients.csv")
    d = dl.preparer()
    model, f1_val = dl.entrainer(args.kind, args.seed, d["X"], d["y"], d["tr"], d["va"])
    pred = dl.predire(model, d["X"][d["te"]]).argmax(1)
    f1_test = f1_score(d["y"][d["te"]], pred, average="macro")

    out = HERE / "models"
    out.mkdir(exist_ok=True)
    name = args.kind.lower()
    torch.save(model.state_dict(), out / f"{name}.pt")
    prep = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d["prep"].items()}
    prep.update({"kind": args.kind, "hidden": dl.HIDDEN, "features": dl.FEATURES,
                 "levels": dl.NIVEAUX, "seed": args.seed,
                 "f1_val": float(f1_val), "f1_test": float(f1_test)})
    (out / f"{name}_prep.json").write_text(json.dumps(prep, indent=2))
    print(f"{args.kind}: macro-F1 validation {f1_val:.3f}, test {f1_test:.3f} -> {out / (name + '.pt')}")


if __name__ == "__main__":
    main()
