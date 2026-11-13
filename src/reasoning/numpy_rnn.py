"""Run the DL team's GRU/LSTM without PyTorch.

PyTorch is only needed to *train* the model (prediction/dl/export_model.py).
To *use* it, this module

1. reads the weights from the ``.pt`` file directly. A ``.pt`` file is a zip
   holding a pickle (the state dict's structure) and the raw tensor bytes.
   The unpickler below accepts only the handful of classes a state dict
   needs and refuses everything else, so loading cannot run arbitrary code;
2. re-implements the forward pass of ``dl_risk.Modele`` in NumPy, with the
   exact gate equations of ``torch.nn.GRU`` / ``torch.nn.LSTM``:

   GRU   r = σ(W_ir x + b_ir + W_hr h + b_hr)
         z = σ(W_iz x + b_iz + W_hz h + b_hz)
         n = tanh(W_in x + b_in + r ⊙ (W_hn h + b_hn))
         h = (1 − z) ⊙ n + z ⊙ h

   LSTM  i, f, g, o from W_ih x + b_ih + W_hh h + b_hh (gate order i, f, g, o)
         c = f ⊙ c + i ⊙ tanh(g),   h = o ⊙ tanh(c)

   then the linear layer on the last hidden state and a softmax, as
   ``dl_risk.predire`` does.
"""

from __future__ import annotations

import io
import pickle
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

_STORAGE_DTYPES = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


class _StorageType:
    def __init__(self, dtype):
        self.dtype = dtype


def _rebuild_tensor(storage, offset, size, stride, *_):
    item = storage.dtype.itemsize
    view = np.lib.stride_tricks.as_strided(storage[offset:], shape=tuple(size),
                                           strides=tuple(s * item for s in stride))
    return np.array(view)  # copy, so the array owns its memory


def _rebuild_parameter(data, *_):
    return data


def load_state_dict(path: str | Path) -> dict[str, np.ndarray]:
    """Read a ``torch.save(model.state_dict(), path)`` file into NumPy arrays."""
    with zipfile.ZipFile(path) as z:
        pkl = next(n for n in z.namelist() if n.endswith("/data.pkl"))
        prefix = pkl[: -len("data.pkl")]
        order = z.read(prefix + "byteorder").decode().strip() if prefix + "byteorder" in z.namelist() else "little"
        if order != "little":
            raise ValueError(f"{path}: unsupported byte order {order!r}")

        class Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if (module, name) == ("collections", "OrderedDict"):
                    return OrderedDict
                if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
                    return _rebuild_tensor
                if (module, name) == ("torch._utils", "_rebuild_parameter"):
                    return _rebuild_parameter
                if module == "torch" and name in _STORAGE_DTYPES:
                    return _StorageType(_STORAGE_DTYPES[name])
                raise pickle.UnpicklingError(f"{path}: refusing to load {module}.{name}")

            def persistent_load(self, pid):
                kind, stype, key, _location, _numel = pid
                if kind != "storage":
                    raise pickle.UnpicklingError(f"{path}: unknown persistent id {kind!r}")
                return np.frombuffer(z.read(f"{prefix}data/{key}"), dtype=stype.dtype)

        state = Unpickler(io.BytesIO(z.read(pkl))).load()
    return {k: np.asarray(v) for k, v in state.items()}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class NumpyRNN:
    """Same computation as ``dl_risk.Modele`` (one recurrent layer + linear)."""

    def __init__(self, state: dict[str, np.ndarray], kind: str = "GRU") -> None:
        if kind not in ("GRU", "LSTM"):
            raise ValueError(f"unknown kind {kind!r}")
        self.kind = kind
        self.w_ih = state["rnn.weight_ih_l0"].astype(np.float64)
        self.w_hh = state["rnn.weight_hh_l0"].astype(np.float64)
        self.b_ih = state["rnn.bias_ih_l0"].astype(np.float64)
        self.b_hh = state["rnn.bias_hh_l0"].astype(np.float64)
        self.w_out = state["sortie.weight"].astype(np.float64)
        self.b_out = state["sortie.bias"].astype(np.float64)
        self.hidden = self.w_hh.shape[1]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, time, features), already normalised. Returns (batch, classes)."""
        x = np.asarray(x, dtype=np.float64)
        n, steps, _ = x.shape
        h = np.zeros((n, self.hidden))
        c = np.zeros((n, self.hidden))
        H = self.hidden
        for t in range(steps):
            gi = x[:, t, :] @ self.w_ih.T + self.b_ih
            gh = h @ self.w_hh.T + self.b_hh
            if self.kind == "GRU":
                r = _sigmoid(gi[:, :H] + gh[:, :H])
                z = _sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
                cand = np.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
                h = (1 - z) * cand + z * h
            else:
                g = gi + gh
                i, f = _sigmoid(g[:, :H]), _sigmoid(g[:, H:2 * H])
                cell, o = np.tanh(g[:, 2 * H:3 * H]), _sigmoid(g[:, 3 * H:])
                c = f * c + i * cell
                h = o * np.tanh(c)
        logits = h @ self.w_out.T + self.b_out
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)
