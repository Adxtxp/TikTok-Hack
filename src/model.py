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

import numpy as np
import torch
import torch.nn as nn

__all__ = ["Head", "predict_proba", "save_head", "load_head"]

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
    ckpt = {
        "state_dict": model.state_dict(),
        "in_dim": getattr(model, "in_dim", None),
        "hidden": getattr(model, "hidden", DEFAULT_HIDDEN),
        "dropout": getattr(model, "dropout", DEFAULT_DROPOUT),
    }
    if extra:
        ckpt["extra"] = extra
    torch.save(ckpt, path)


def _torch_load(path, map_location):
    """torch.load across versions.

    torch >= 2.6 flipped ``weights_only`` to True by default. Our checkpoint
    holds only tensors, ints and floats, so it loads fine either way; the
    try/except just keeps the explicit flag from breaking older torch that
    doesn't accept the kwarg.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


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
    return model
