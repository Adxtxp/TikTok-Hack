"""Train the classifier head on fused DINOv2 + FFT features.

CLI:
    python -m src.train --manifest manifest.csv --out outputs/head.pt
    python -m src.train --manifest manifest.csv --out outputs/head.pt --augment
    python -m src.train --embeddings outputs/embeddings_fused.npz --epochs 50

WHAT THIS FIXES RELATIVE TO THE NOTEBOOK
========================================
The notebook's training cell has two problems that this script deliberately
does not reproduce.

1. Test-set leakage in model selection.
   The notebook splits the manifest into train/test, trains on train, and then
   prints accuracy/AUC on *test* every 5 epochs. Every decision made by
   looking at that number - when to stop, which hyperparameters to keep - is
   informed by the test set, so the reported test score is optimistically
   biased and is no longer an estimate of generalisation.

   Here: the test rows (``split == "test"``) are filtered out and never
   loaded. The train rows are split again into a real train/validation pair
   using ``val_split``, and only validation metrics are printed. The test set
   stays sealed for ``src/evaluate.py``. See ``_split_train_val`` below - it is
   the only place a split is made, and it operates on train-only data.

2. Full-batch gradient descent mislabelled as an epoch loop.
   The notebook runs one forward/backward on the entire training matrix per
   "epoch" - 30 epochs is 30 gradient steps, which is nowhere near
   convergence for an Adam-trained MLP. Here each epoch iterates shuffled
   mini-batches, giving len(train)/batch_size steps per epoch.

Everything else (architecture, Adam, lr=1e-3, BCEWithLogitsLoss) is kept as
the notebook had it.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# tqdm.auto renders as a widget in Colab and a text bar under `python -m`.
from tqdm.auto import tqdm

# Support both `python -m src.train` (package) and `python src/train.py`.
try:
    from src.features import FUSED_DIM, get_fused_embeddings, load_embeddings, save_embeddings
    from src.model import (Head, apply_scaler, predict_proba, save_head,
                           save_scaler, scaler_path_for)
    from src.transforms import TRANSFORM_NAMES
except ImportError:  # pragma: no cover - sys.path shape, not logic
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.features import FUSED_DIM, get_fused_embeddings, load_embeddings, save_embeddings
    from src.model import (Head, apply_scaler, predict_proba, save_head,
                           save_scaler, scaler_path_for)
    from src.transforms import TRANSFORM_NAMES

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml"
)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

def load_config(path=DEFAULT_CONFIG):
    """Read configs/default.yaml, tolerating a missing file or empty values."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_train_rows(manifest_path):
    """Return the TRAIN-split rows of the manifest only.

    This is the leakage firewall: rows with split == "test" are dropped here
    and never reach any later stage of this script. Nothing downstream has to
    remember to exclude them, because they were never in the dataframe.

    Returns:
        (paths, labels) - a list of image paths and an int array of 0/1 labels.
    """
    df = pd.read_csv(manifest_path)

    missing = {"image_path", "label", "split"} - set(df.columns)
    if missing:
        raise ValueError(
            f"manifest {manifest_path!r} is missing required column(s): {sorted(missing)}"
        )

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    if train_df.empty:
        raise ValueError(
            f"manifest {manifest_path!r} has no rows with split == 'train' "
            f"(found splits: {sorted(df['split'].unique())})"
        )

    n_held_out = len(df) - len(train_df)
    print(f"manifest: {len(df)} rows -> {len(train_df)} train rows used, "
          f"{n_held_out} non-train rows HELD OUT (never loaded)")
    print("  train label balance:", train_df["label"].value_counts().to_dict())

    return train_df["image_path"].tolist(), train_df["label"].to_numpy().astype(np.int64)


def _split_train_val(n, labels, val_split, seed):
    """Split indices [0, n) into train / validation index arrays.

    Stratified on the label so both sides keep the real/fake balance - with a
    small val_split on an imbalanced set, an unstratified split can easily
    produce a validation fold missing a class entirely, which makes AUC
    undefined.

    Returns:
        (train_idx, val_idx) int arrays.
    """
    idx = np.arange(n)
    if val_split <= 0:
        return idx, np.zeros((0,), dtype=np.int64)

    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_split,
        random_state=seed,
        shuffle=True,
        stratify=labels if len(np.unique(labels)) > 1 else None,
    )
    return np.sort(train_idx), np.sort(val_idx)


def build_augmented_embeddings(paths, batch_size, rng, embedder=None):
    """Build fused features with a RANDOM transform per image.

    Training on degraded images is what buys robustness: the head learns a
    boundary that survives JPEG/blur/noise instead of one that depends on
    pristine generator artifacts. Each image gets one randomly chosen
    transform from TRANSFORM_NAMES (including "clean", so undegraded examples
    stay in the mix).

    Implementation note: images are grouped by their assigned transform and
    each group is extracted in one batched call, then scattered back to the
    original row order. Doing it per-image would forfeit batching and make
    extraction dramatically slower.

    Returns:
        (N, FUSED_DIM) float32 array, row i corresponding to paths[i].
    """
    n = len(paths)
    assigned = rng.choice(TRANSFORM_NAMES, size=n)

    out = None
    for name in np.unique(assigned):
        where = np.flatnonzero(assigned == name)
        group_paths = [paths[i] for i in where]

        # ---------------------------------------------------------------
        # TODO(extend me): this is the augmented feature-building hook.
        #
        # Currently: exactly one transform per image, sampled uniformly from
        # TRANSFORM_NAMES. Things worth trying from here:
        #   * Weight the sampling - e.g. draw "clean" more often, or bias
        #     toward the transforms that evaluate.py shows are hurting most.
        #   * Compose transforms (jpeg_50 THEN blur_1.0) to mimic a real
        #     multi-hop upload chain. apply_transform takes a single name, so
        #     this needs a small chain helper in src/transforms.py.
        #   * Emit several augmented copies per image (n_copies > 1) and
        #     repeat the labels to match - grows the effective train set.
        #   * Sample a fresh transform each epoch instead of once up front,
        #     which means moving extraction inside the epoch loop (much
        #     slower, but strictly better regularisation).
        # ---------------------------------------------------------------
        feats = get_fused_embeddings(
            group_paths,
            transform_name=name,
            batch_size=batch_size,
            embedder=embedder,
            show_progress=True,
        )

        if out is None:
            out = np.zeros((n, feats.shape[1]), dtype=np.float32)
        out[where] = feats

    if out is None:
        return np.zeros((0, FUSED_DIM), dtype=np.float32)

    counts = {str(k): int(v) for k, v in zip(*np.unique(assigned, return_counts=True))}
    print("augmentation mix:", counts)
    return out


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def evaluate_split(model, X, y):
    """Accuracy / AUC of the head on one feature matrix.

    AUC is undefined when a fold is single-class, so it degrades to None
    rather than raising - a tiny val_split shouldn't crash a training run.
    """
    if len(X) == 0:
        return None, None
    probs = predict_proba(model, X)
    acc = accuracy_score(y, probs > 0.5)
    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else None
    return acc, auc


def train(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=30,
    batch_size=32,
    learning_rate=1e-3,
    hidden=128,
    dropout=0.2,
    device=None,
    seed=42,
    patience=5,
):
    """Fit a Head with mini-batch Adam, reporting validation metrics per epoch.

    Early stopping on validation AUC: the best-scoring epoch's weights are
    kept, and training halts once ``patience`` consecutive epochs fail to beat
    that best. AUC rather than accuracy because it is threshold-free - it moves
    when the score distributions separate further, while accuracy can sit
    frozen for several epochs at a fixed 0.5 cut and make a still-improving
    model look plateaued.

    ``epochs`` is therefore a ceiling, not a target.

    Args:
        patience: epochs without a val-AUC improvement before stopping.
            0 (or no validation set) disables early stopping, in which case the
            full ``epochs`` are run and the last epoch's weights are kept.

    Returns:
        (model, history). The model carries the weights from the BEST epoch,
        not the last one. Each history entry gains a "best" flag marking which
        epoch that was.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    in_dim = X_train.shape[1]
    model = Head(in_dim=in_dim, hidden=hidden, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    Xt = torch.from_numpy(np.asarray(X_train, dtype=np.float32))
    # (N, 1) to match the head's single-logit output shape.
    yt = torch.from_numpy(np.asarray(y_train, dtype=np.float32)).unsqueeze(1)

    n = len(Xt)
    gen = torch.Generator().manual_seed(seed)

    print(f"\ntraining: {n} train / {len(X_val)} val rows, in_dim={in_dim}, "
          f"device={device}, {epochs} epochs x {max(1, n // batch_size)} steps")

    # Early-stopping state. Needs a validation set to have anything to watch.
    early_stop = bool(patience) and patience > 0 and len(X_val) > 0
    if early_stop:
        print(f"early stopping: on val AUC, patience {patience} "
              f"({epochs} epochs is the maximum)")
    else:
        print(f"early stopping: DISABLED "
              f"({'no validation rows' if len(X_val) == 0 else f'patience={patience}'})"
              f" - running all {epochs} epochs")

    best_auc = float("-inf")
    best_epoch = None
    best_state = None
    stale_epochs = 0
    warned_no_auc = False

    history = []
    for epoch in range(epochs):
        model.train()
        # Fresh shuffle each epoch so batch composition varies - the point of
        # mini-batching that the notebook's full-batch step misses entirely.
        perm = torch.randperm(n, generator=gen)

        epoch_loss, n_batches = 0.0, 0

        # Intra-epoch progress. leave=False erases each epoch's bar once it
        # finishes, so the per-epoch metric lines below stay the permanent
        # record instead of being separated by 30 spent bars.
        batches = tqdm(
            range(0, n, batch_size),
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
            leave=False,
        )

        for start in batches:
            sel = perm[start:start + batch_size]
            xb, yb = Xt[sel].to(device), yt[sel].to(device)

            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            # Running mean loss on the bar, so a diverging run is visible
            # immediately rather than only at the end-of-epoch print.
            batches.set_postfix(loss=f"{epoch_loss / n_batches:.4f}")

        train_loss = epoch_loss / max(1, n_batches)

        # VALIDATION only - the test set is not present in this process.
        val_acc, val_auc = evaluate_split(model, X_val, y_val)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_acc": val_acc, "val_auc": val_auc})

        acc_s = "n/a" if val_acc is None else f"{val_acc:.4f}"
        auc_s = "n/a" if val_auc is None else f"{val_auc:.4f}"
        marker = ""

        # ---- early stopping bookkeeping --------------------------------
        if early_stop and val_auc is None:
            # A single-class validation fold makes AUC undefined; without a
            # signal there is nothing to stop on, so fall back to running out
            # the epochs rather than stopping arbitrarily.
            if not warned_no_auc:
                print("  (val AUC undefined - early stopping inactive this run)")
                warned_no_auc = True
        elif early_stop:
            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                # Detached clones: state_dict() hands back live references, so
                # without copying, the next optimizer.step() would overwrite
                # the very weights we are trying to preserve.
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
                stale_epochs = 0
                marker = "  <- best"
            else:
                stale_epochs += 1
                marker = f"  (no improvement, {stale_epochs}/{patience})"

        print(f"epoch {epoch + 1:>3}/{epochs}  train_loss {train_loss:.4f}  "
              f"val_acc {acc_s}  val_auc {auc_s}{marker}")

        if early_stop and stale_epochs >= patience:
            print(f"\nEARLY STOP at epoch {epoch + 1}/{epochs}: val AUC has not "
                  f"improved for {patience} consecutive epoch(s).")
            print(f"  best epoch: {best_epoch + 1} (val_auc {best_auc:.4f})")
            print(f"  skipped {epochs - (epoch + 1)} remaining epoch(s)")
            break

    # Roll back to the best epoch. Done whether or not early stopping fired,
    # because the last epoch is often slightly worse than the best even in a
    # run that goes the full distance - which is exactly the mild overfitting
    # this is meant to avoid.
    if best_epoch is not None:
        history[best_epoch]["best"] = True
        if best_epoch != len(history) - 1:
            model.load_state_dict(best_state)
            print(f"\nrestored weights from best epoch {best_epoch + 1} "
                  f"(val_auc {best_auc:.4f}); epoch {len(history)}'s weights discarded")
        else:
            print(f"\nbest epoch was the last one ({best_epoch + 1}, "
                  f"val_auc {best_auc:.4f}); no rollback needed")

    return model, history


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train the fused-feature classifier head (test set held out)."
    )
    p.add_argument("--manifest", default=None,
                   help="CSV with image_path,label,split. Only split=='train' rows are used.")
    p.add_argument("--out", default=None,
                   help="Where to write the head checkpoint (.pt).")
    p.add_argument("--epochs", type=int, default=None,
                   help="MAXIMUM training epochs (default: config epochs). Early "
                        "stopping usually ends the run before this.")
    p.add_argument("--patience", type=int, default=5,
                   help="Stop after this many epochs with no val-AUC improvement, "
                        "keeping the best epoch's weights. 0 disables early "
                        "stopping and runs all --epochs. Default 5.")
    p.add_argument("--augment", action="store_true",
                   help="Apply a random transform per training image before feature "
                        "extraction, to train for robustness. Validation stays clean.")
    p.add_argument("--normalize", action="store_true",
                   help="Z-score each feature dimension. The scaler is fit on the "
                        "TRAINING features only and saved beside the head, so "
                        "evaluate/inference reuse the identical transform. Default "
                        "off, leaving the un-normalized baseline unchanged.")
    p.add_argument("--scaler-out", default=None,
                   help="Where to write the fitted scaler (default: alongside "
                        "--out, as <head>.scaler.pkl). Only used with --normalize.")
    # Extras with config-backed defaults.
    p.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config path.")
    p.add_argument("--embeddings", default=None,
                   help="Precomputed fused .npz to train from instead of a manifest. "
                        "Ignored when --augment is set (augmentation needs the images).")
    p.add_argument("--cache-embeddings", default=None,
                   help="Write the freshly built clean embeddings to this .npz.")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--val-split", type=float, default=None)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)

    # CLI beats config beats hardcoded default.
    epochs = args.epochs if args.epochs is not None else int(cfg.get("epochs") or 30)
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.get("batch_size") or 32)
    lr = args.lr if args.lr is not None else float(cfg.get("learning_rate") or 1e-3)
    val_split = args.val_split if args.val_split is not None else float(cfg.get("val_split") or 0.15)
    manifest = args.manifest or cfg.get("manifest_path") or None
    embeddings_path = args.embeddings or (cfg.get("embeddings_path") or None)
    out_path = args.out or cfg.get("head_path") or os.path.join(
        cfg.get("outputs_dir") or "outputs", "head_fused.pt")

    if not (0.0 <= val_split < 1.0):
        raise ValueError(f"--val-split must be in [0, 1); got {val_split}")

    rng = np.random.default_rng(args.seed)

    # ---- assemble TRAIN-ONLY features ---------------------------------
    if args.augment:
        if not manifest:
            raise SystemExit("--augment needs --manifest (it re-reads the image files)")
        paths, labels = load_train_rows(manifest)
        train_idx, val_idx = _split_train_val(len(paths), labels, val_split, args.seed)

        # Split BEFORE extraction so augmentation touches training rows only.
        # Validation must stay clean, or the val metric measures accuracy on
        # degraded images and can't be compared across runs.
        print(f"\nsplit: {len(train_idx)} train / {len(val_idx)} val "
              f"(val_split={val_split}, stratified, seed={args.seed})")
        print("building AUGMENTED training features...")
        X_train = build_augmented_embeddings(
            [paths[i] for i in train_idx], batch_size, rng)
        print("building clean validation features...")
        X_val = get_fused_embeddings(
            [paths[i] for i in val_idx], transform_name="clean", batch_size=batch_size)
        y_train, y_val = labels[train_idx], labels[val_idx]

    elif embeddings_path and os.path.exists(embeddings_path):
        print(f"loading cached embeddings: {embeddings_path}")
        X, y, splits = load_embeddings(embeddings_path)
        if splits is None or y is None:
            raise SystemExit(
                f"{embeddings_path} lacks 'labels'/'splits' arrays, so train rows "
                "cannot be isolated - rebuild it from the manifest instead."
            )
        # Same firewall as load_train_rows, applied to the cached matrix.
        keep = np.asarray(splits) == "train"
        n_held = int((~keep).sum())
        X, y = np.asarray(X)[keep], np.asarray(y)[keep].astype(np.int64)
        print(f"  {len(X)} train rows used, {n_held} non-train rows HELD OUT")

        train_idx, val_idx = _split_train_val(len(X), y, val_split, args.seed)
        print(f"\nsplit: {len(train_idx)} train / {len(val_idx)} val "
              f"(val_split={val_split}, stratified, seed={args.seed})")
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

    else:
        if not manifest:
            raise SystemExit("need --manifest or an existing --embeddings .npz")
        paths, labels = load_train_rows(manifest)
        print("\nbuilding clean features for all train rows...")
        X = get_fused_embeddings(paths, transform_name="clean", batch_size=batch_size)
        y = labels
        if args.cache_embeddings:
            # splits are all "train" here by construction.
            save_embeddings(args.cache_embeddings, X, y,
                            np.array(["train"] * len(y)))
            print("cached embeddings ->", args.cache_embeddings)

        train_idx, val_idx = _split_train_val(len(X), y, val_split, args.seed)
        print(f"\nsplit: {len(train_idx)} train / {len(val_idx)} val "
              f"(val_split={val_split}, stratified, seed={args.seed})")
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

    # ---- optional feature normalization (the --normalize experiment) ----
    # Placed here, after all three feature-assembly branches have converged on
    # (X_train, X_val), so there is exactly one place scaling can happen and
    # no branch can accidentally skip it.
    #
    # THE CORRECTNESS PROPERTY: fit on X_train ONLY, then transform val with
    # that same fitted scaler. Fitting on val (or on the concatenation) would
    # let the validation set's mean/std inform the transform, which inflates
    # the val metric and makes it useless as a held-out estimate. The test set
    # is not even present in this process.
    scaler = None
    if args.normalize:
        scaler = StandardScaler()
        scaler.fit(X_train)                       # <- train only
        X_train = apply_scaler(scaler, X_train)   # <- transform
        X_val = apply_scaler(scaler, X_val)       # <- same fitted scaler
        print(f"\nnormalize: z-scored {X_train.shape[1]} dims "
              f"(scaler fit on {len(X_train)} train rows only)")
        print(f"  pre-fit train mean range:  "
              f"{scaler.mean_.min():.3f} .. {scaler.mean_.max():.3f}")
        print(f"  pre-fit train scale range: "
              f"{scaler.scale_.min():.3f} .. {scaler.scale_.max():.3f}")
    else:
        print("\nnormalize: OFF (baseline, un-normalized features)")

    # ---- train --------------------------------------------------------
    model, history = train(
        X_train, y_train, X_val, y_val,
        epochs=epochs, batch_size=batch_size, learning_rate=lr,
        hidden=args.hidden, dropout=args.dropout, seed=args.seed,
        patience=args.patience,
    )

    # ---- save ---------------------------------------------------------
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # The saved weights come from the best epoch, so the recorded metrics must
    # too - taking history[-1] here would pair best-epoch weights with a
    # different epoch's numbers.
    best = next((h for h in history if h.get("best")), None)
    final = best if best is not None else (history[-1] if history else {})
    save_head(model, out_path, extra={
        "augmented": bool(args.augment),
        "epochs": epochs,
        "epochs_run": len(history),
        "patience": args.patience,
        "best_epoch": (final.get("epoch") + 1) if final.get("epoch") is not None else None,
        "val_split": val_split,
        "seed": args.seed,
        # Recorded so evaluate/inference know whether this head EXPECTS
        # normalized features, instead of guessing from whether a scaler file
        # happens to be sitting in outputs/.
        "normalized": bool(args.normalize),
        "final_val_acc": final.get("val_acc"),
        "final_val_auc": final.get("val_auc"),
    })
    print(f"\nsaved head -> {out_path}")

    if scaler is not None:
        # Saved beside the head so evaluate.py and inference.py find it
        # automatically and apply the IDENTICAL transform.
        scaler_out = args.scaler_out or scaler_path_for(out_path)
        save_scaler(scaler, scaler_out)
        print(f"saved scaler -> {scaler_out}")
    print("NOTE: validation metrics above are NOT a test score. "
          "Run src/evaluate.py on the held-out test split for that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
