"""Classifier head that sits on top of the frozen fused features.

Ported from the head-training cell of ``run_on_colab.ipynb``.

The architecture is deliberately tiny - Linear -> ReLU -> Dropout -> Linear ->
one logit. All the representational work was already done by the frozen
DINOv2 backbone in ``src/features.py``; this head only has to find a decision
boundary in the 800-d fused space. Keeping it small is what makes the whole
pipeline trainable in seconds on cached embeddings, and it's also the reason
the backbone stays frozen: with ~103k parameters here versus 86M there, a
fine-tuned backbone on a hackathon-sized subset would overfit immediately.

Single logit + BCEWithLogitsLoss (rather than 2 classes + cross-entropy) is
the notebook's choice and is kept: it makes P(fake) a single number, so the
decision threshold is explicit and tunable at inference time rather than
hidden inside an argmax.

Label convention (from the manifest): 0 = real, 1 = fake. So the sigmoid of
the logit is P(fake).
"""

import importlib
import os
import pickle

import joblib
import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "Head",
    "predict_proba",
    "save_head",
    "load_head",
    "SCALER_SUFFIX",
    "scaler_path_for",
    "save_scaler",
    "load_scaler",
    "resolve_feature_scaler",
    "apply_scaler",
]

# Notebook defaults, kept as the module defaults so a bare Head(in_dim)
# reproduces the notebook's model exactly.
DEFAULT_HIDDEN = 128
DEFAULT_DROPOUT = 0.2


class Head(nn.Module):
    """Two-layer MLP producing one logit per sample.

    Args:
        in_dim: width of the input feature vector. For the standard fused
            layout this is ``src.features.FUSED_DIM`` (800). Passed in rather
            than hardcoded so a different FFT bin count or backbone still
            works.
        hidden: width of the hidden layer.
        dropout: dropout probability applied between the two Linear layers.
            Note this only has an effect in ``.train()`` mode - remember to
            call ``.eval()`` before measuring anything.
    """

    def __init__(self, in_dim, hidden=DEFAULT_HIDDEN, dropout=DEFAULT_DROPOUT):
        super().__init__()
        # Kept in a single `self.net` Sequential to match the notebook's
        # state_dict key names ("net.0.weight", ...), so a checkpoint saved by
        # the notebook loads into this class unchanged.
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        # Remembered so save_head can record the shape the weights expect.
        self.in_dim = in_dim
        self.hidden = hidden
        self.dropout = dropout

    def forward(self, x):
        """Returns raw logits, shape (N, 1). Apply sigmoid for probabilities."""
        return self.net(x)


@torch.no_grad()
def predict_proba(model, features_np, batch_size=4096, device=None):
    """Run the head over a numpy feature matrix and return P(fake).

    Handles the eval-mode / no-grad / device / sigmoid boilerplate that is
    easy to forget - notably ``.eval()``, without which dropout would randomly
    perturb predictions and make them non-reproducible.

    Args:
        model: a Head (or anything returning one logit per row).
        features_np: (N, in_dim) array-like of fused features.
        batch_size: rows per forward pass, to bound memory on large sets.
        device: torch device. Defaults to the device the model is already on.

    Returns:
        float64 numpy array of shape (N,) with probabilities in [0, 1].
        Threshold at 0.5 for a hard label (1 = fake).
    """
    was_training = model.training
    model.eval()

    if device is None:
        # Infer from the model's own parameters rather than guessing.
        device = next(model.parameters()).device

    features_np = np.asarray(features_np, dtype=np.float32)
    if features_np.ndim == 1:
        # Tolerate a single un-batched vector.
        features_np = features_np[None, :]

    probs = []
    for i in range(0, len(features_np), batch_size):
        chunk = torch.from_numpy(features_np[i:i + batch_size]).to(device)
        logits = model(chunk)
        probs.append(torch.sigmoid(logits).squeeze(-1).cpu().numpy())

    if was_training:
        # Leave the model as we found it.
        model.train()

    if not probs:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(probs).astype(np.float64)


def save_head(model, path, extra=None):
    """Save a head checkpoint that carries its own architecture.

    The notebook saved a bare ``state_dict``, which silently loses the input
    width. Reloading it against a differently-shaped feature vector (a changed
    FFT bin count, a different backbone) then fails with an opaque shape
    mismatch - or worse, appears to work. Recording in_dim / hidden / dropout
    alongside the weights means ``load_head`` can rebuild the exact
    architecture and fail loudly and early when features don't match.

    Args:
        model: the Head to save.
        path: destination .pt path.
        extra: optional dict of metadata to embed (e.g. metrics, config).
    """
    in_dim = getattr(model, "in_dim", None)
    hidden = getattr(model, "hidden", DEFAULT_HIDDEN)
    dropout = getattr(model, "dropout", DEFAULT_DROPOUT)

    # Explicit casts to plain Python types. Any numpy-flavoured value pickled
    # here (a np.float64 metric, a np.int64 dimension) becomes a
    # `numpy._core.multiarray.scalar` global in the archive, which torch's
    # strict weights_only unpickler refuses to reconstruct - see _torch_load.
    ckpt = {
        "state_dict": model.state_dict(),
        "in_dim": None if in_dim is None else int(in_dim),
        "hidden": int(hidden),
        "dropout": float(dropout),
    }
    if extra:
        # In practice this is where the numpy scalars actually come from:
        # callers pass metrics straight through from sklearn, and older
        # sklearn returns np.float64 from accuracy_score / roc_auc_score.
        # Coerced recursively, so no numpy object reaches the pickle no matter
        # what the caller hands over.
        ckpt["extra"] = _to_plain_python(extra)

    torch.save(ckpt, path)


def _to_plain_python(obj):
    """Recursively convert numpy scalars/arrays into plain Python equivalents.

    Applied to checkpoint metadata before pickling. Containers are walked, so a
    nested dict of metrics is handled and not just top-level values.
    """
    # numpy scalar (np.float64, np.int64, np.bool_, ...) -> float / int / bool
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {_to_plain_python(k): _to_plain_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return type(obj)(_to_plain_python(v) for v in obj)
    return obj


def _numpy_safe_globals():
    """numpy globals a legacy checkpoint may reference, for the allowlist.

    Built defensively: the private scalar-reconstruction helper moved from
    ``numpy.core.multiarray`` (numpy 1.x) to ``numpy._core.multiarray``
    (numpy 2.x), and the concrete dtype classes differ by version, so anything
    absent is skipped rather than raising at import time.
    """
    allow = []

    for mod_name in ("numpy._core.multiarray", "numpy.core.multiarray"):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:  # noqa: BLE001 - a missing/renamed module is expected
            continue
        fn = getattr(mod, "scalar", None)
        if fn is not None:
            allow.append(fn)

    # Reconstructing a scalar also pulls in its dtype.
    allow.append(np.dtype)
    for name in ("float64", "float32", "int64", "int32", "bool_"):
        t = getattr(np, name, None)
        if t is not None:
            allow.append(t)

    dtypes_mod = getattr(np, "dtypes", None)
    if dtypes_mod is not None:
        for name in ("Float64DType", "Float32DType", "Int64DType", "Int32DType",
                     "BoolDType"):
            t = getattr(dtypes_mod, name, None)
            if t is not None:
                allow.append(t)

    return allow


def _torch_load(path, map_location):
    """torch.load, robust across torch versions and checkpoint vintages.

    WHY THIS IS NOT JUST torch.load()
    ---------------------------------
    torch 2.6 flipped the ``weights_only`` default from False to True. Under
    the strict unpickler only a small allowlist of types can be
    reconstructed, so a checkpoint whose *metadata* holds a numpy scalar now
    fails with:

        _pickle.UnpicklingError: Weights only load failed ...
        Unsupported global: GLOBAL numpy._core.multiarray.scalar was not an
        allowed global by default

    The weights themselves are fine - it is the metadata dict beside them that
    trips the check. ``save_head`` no longer writes numpy objects, but head
    files saved before that fix still exist, and retraining purely to reload
    them would be absurd. So this degrades in three steps:

      1. strict ``weights_only=True``           - new files; no trust needed
      2. strict + numpy scalars allowlisted     - legacy files, still strict
                                                  about everything else
      3. ``weights_only=False``, with a warning - last resort, full unpickle

    Step 3 executes arbitrary pickled code, which is why it is last, loud, and
    reached only for files the first two steps cannot read. These are your own
    checkpoints, so that is acceptable here - it would not be for a checkpoint
    downloaded from a stranger.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # torch too old to know the kwarg at all; its default is a full
        # unpickle, so nothing further is needed.
        return torch.load(path, map_location=map_location)
    except (pickle.UnpicklingError, RuntimeError) as exc:
        # RuntimeError as well as UnpicklingError: some torch versions wrap
        # the weights-only failure rather than raising it directly.
        strict_error = exc

    # Step 2: allowlist just the numpy globals, keeping the strict unpickler.
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is not None:
        try:
            with safe_globals(_numpy_safe_globals()):
                return torch.load(path, map_location=map_location,
                                  weights_only=True)
        except Exception:  # noqa: BLE001 - fall through to the last resort
            pass

    # Step 3: full unpickle, announced. The wording stays non-committal about
    # the cause: reaching here usually means a legacy numpy-containing file,
    # but a genuinely corrupt archive lands here too and must not be
    # mis-described as merely out of date.
    print(
        f"WARNING: {path} could not be read by the strict (weights_only) "
        f"loader:\n"
        f"  {type(strict_error).__name__}: "
        f"{str(strict_error).splitlines()[0]}\n"
        f"  Retrying with weights_only=False. If this is an older head file, "
        f"re-save it with save_head() to get a portable one; if the load still "
        f"fails, the file is unreadable rather than merely out of date."
    )
    return torch.load(path, map_location=map_location, weights_only=False)


def load_head(path, map_location="cpu"):
    """Load a head saved by ``save_head``, returning it in eval mode.

    Also accepts a legacy bare state_dict (as written by the notebook's
    ``torch.save(head.state_dict(), ...)``), in which case the architecture is
    recovered from the weight shapes themselves.

    Args:
        path: checkpoint path.
        map_location: where to place the tensors.

    Returns:
        A Head in ``.eval()`` mode.

    Raises:
        ValueError: if the checkpoint has no recoverable input width.
    """
    ckpt = _torch_load(path, map_location)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        in_dim = ckpt.get("in_dim")
        hidden = ckpt.get("hidden", DEFAULT_HIDDEN)
        dropout = ckpt.get("dropout", DEFAULT_DROPOUT)
    else:
        # Legacy path: a raw state_dict. Infer the shape from the first
        # Linear's weight, which is (hidden, in_dim).
        state = ckpt
        in_dim = hidden = None
        dropout = DEFAULT_DROPOUT

    if in_dim is None:
        w = state.get("net.0.weight")
        if w is None:
            raise ValueError(
                f"cannot determine in_dim from checkpoint {path!r}: no 'in_dim' "
                "field and no 'net.0.weight' to infer it from"
            )
        hidden, in_dim = tuple(w.shape)

    model = Head(in_dim=in_dim, hidden=hidden, dropout=dropout)
    model.load_state_dict(state)
    model.eval()

    # Expose the checkpoint's metadata on the returned model. Callers need it
    # to know whether this head was trained on normalized features - see
    # resolve_feature_scaler. Attribute only; no effect on the forward pass.
    model.extra = ckpt.get("extra", {}) if isinstance(ckpt, dict) else {}

    return model


# --------------------------------------------------------------------------
# Optional feature scaler (the --normalize experiment)
#
# The fused vector concatenates two blocks on very different scales: DINOv2
# CLS values are roughly unit-scale, while the FFT block is a log-magnitude
# around 7.8-9.0. Z-scoring each of the 800 dimensions puts them on equal
# footing, which may or may not help - hence an A/B toggle rather than a
# change to the default path.
#
# The scaler is a *fitted* object: it carries the per-dimension mean and
# std learned from the training features. That is exactly why it must be
# persisted next to the head and reused verbatim at eval/inference time -
# re-fitting on val or test would leak their distribution into the transform.
# --------------------------------------------------------------------------

# head_fused.pt -> head_fused.scaler.pkl. Named per-head rather than a single
# shared outputs/scaler.pkl so training a second head (e.g. --augment) cannot
# silently clobber the first one's scaler.
SCALER_SUFFIX = ".scaler.pkl"


def scaler_path_for(head_path):
    """Conventional scaler path beside a head checkpoint."""
    return os.path.splitext(head_path)[0] + SCALER_SUFFIX


def save_scaler(scaler, path):
    """Persist a fitted scaler with joblib."""
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path):
    """Load a fitted scaler written by ``save_scaler``."""
    return joblib.load(path)


def apply_scaler(scaler, features):
    """Transform features with a fitted scaler; a no-op when scaler is None.

    Casts back to float32: StandardScaler.transform promotes to float64, and
    the un-normalized path feeds float32, so this keeps the dtype (and memory
    footprint) identical between the two branches.
    """
    if scaler is None:
        return features
    return scaler.transform(features).astype(np.float32)


def resolve_feature_scaler(head_path, model=None, explicit=None, quiet=False):
    """Decide which scaler (if any) to apply for a given head.

    Auto-detection alone is unsafe: a stale scaler left in outputs/ from an
    earlier normalized run would get silently applied to an un-normalized
    head, quietly corrupting every prediction. So the head's own record of how
    it was trained (``extra["normalized"]``, written by src/train.py) is the
    authority, and a discovered file is only honoured when it agrees.

    Args:
        head_path: path the head was loaded from (used to locate the scaler).
        model: the loaded Head, for its ``extra`` metadata. Optional.
        explicit: an explicitly requested scaler path (--scaler). Wins over
            both auto-detection and the head's metadata.
        quiet: suppress the informational prints.

    Returns:
        (scaler_or_None, path_or_None).

    Raises:
        SystemExit: if the head was trained normalized but no scaler is
            available, or an explicitly named scaler is missing.
    """
    extra = getattr(model, "extra", None) or {}
    was_normalized = extra.get("normalized")

    if explicit:
        if not os.path.exists(explicit):
            raise SystemExit(f"--scaler not found: {explicit}")
        if was_normalized is False:
            print(f"WARNING: {head_path} records normalized=False, but "
                  f"--scaler was given explicitly; applying it anyway.")
        if not quiet:
            print(f"feature scaler: {explicit} (explicit)")
        return load_scaler(explicit), explicit

    # Candidates: the per-head name first, then the generic name in the same
    # directory (so a hand-placed outputs/scaler.pkl is still found).
    candidates = [
        scaler_path_for(head_path),
        os.path.join(os.path.dirname(os.path.abspath(head_path)), "scaler.pkl"),
    ]
    found = next((c for c in candidates if os.path.exists(c)), None)

    if was_normalized is True:
        if found is None:
            raise SystemExit(
                f"{head_path} was trained with --normalize, but no scaler was "
                f"found. Predictions would be wrong without it. Looked for:\n"
                + "\n".join(f"  {c}" for c in candidates)
                + "\nPass --scaler explicitly, or retrain."
            )
        if not quiet:
            print(f"feature scaler: {found} (head was trained normalized)")
        return load_scaler(found), found

    # Head was trained un-normalized, or predates the flag: keep the baseline
    # behaviour and say so if a scaler was lying around.
    if found is not None and not quiet:
        print(f"NOTE: found {found}, but {head_path} was not trained with "
              f"--normalize, so it is being IGNORED. Pass --scaler to force it.")
    return None, None
