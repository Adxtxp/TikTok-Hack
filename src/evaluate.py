"""Robustness evaluation: how much does each degradation hurt the detector?

Ported from the robustness-sweep and diagnosis cells of
``run_on_colab.ipynb``.

CLI:
    python -m src.evaluate --manifest manifest.csv --head outputs/head_fused.pt

What this produces
------------------
``outputs/robustness_table.csv`` - one row per transform in
``src.transforms.TRANSFORM_NAMES``, with accuracy and AUC on a class-balanced
sample of the held-out TEST split. This is the graded deliverable: a single
table showing where the detector survives real-world mangling and where it
collapses.

Followed by ``diagnose_transform`` output for a few interesting transforms - a
confusion matrix plus the mean predicted P(fake) per true class. The table
alone tells you accuracy fell; the diagnosis tells you *how* it fell, which is
the part that actually guides a fix:

  * Both class means drifting toward 0.5 => the degradation destroyed the
    signal and the head is hedging.
  * Both means pushed to one side => a systematic bias; a threshold
    recalibration might recover most of the loss.
  * Only the fake-class mean collapsing => the generator fingerprint was
    erased (the classic JPEG/blur failure) while real images still look real.

This is the ONLY script that touches the test split. ``src/train.py``
deliberately never loads it, so the numbers here are an honest held-out
estimate - provided you do not start tuning against them.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
# tqdm.auto renders as a widget in Colab and a text bar under `python -m`.
from tqdm.auto import tqdm

try:
    from src.features import get_embedder, get_fused_embeddings
    from src.model import (apply_scaler, load_head, predict_proba,
                           resolve_feature_scaler)
    from src.transforms import TRANSFORM_NAMES
except ImportError:  # pragma: no cover - sys.path shape, not logic
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.features import get_embedder, get_fused_embeddings
    from src.model import (apply_scaler, load_head, predict_proba,
                           resolve_feature_scaler)
    from src.transforms import TRANSFORM_NAMES

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml"
)

# The notebook diagnoses these four: the clean baseline plus the three
# degradations that hurt it most.
DEFAULT_DIAGNOSE = ["clean", "blur_2.0", "resize_0.25", "noise_0.10"]


def load_config(path=DEFAULT_CONFIG):
    """Read configs/default.yaml if present."""
    if not path or not os.path.exists(path):
        return {}
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# --------------------------------------------------------------------------
# Eval subset
# --------------------------------------------------------------------------

def build_eval_subset(manifest_path, per_class=500, seed=42):
    """Sample a class-balanced subset of the TEST split.

    Balanced on purpose: with equal class counts, accuracy is directly
    interpretable (0.5 == chance) and cannot be inflated by a majority-class
    guesser, which matters because the whole table is read as a comparison
    across rows.

    Kept small (~1000 images) because the sweep re-extracts DINOv2 features
    once per transform - 15 transforms over the full test set would be a
    needlessly long run for the same conclusion.

    Args:
        manifest_path: CSV with image_path, label, split.
        per_class: images per class; fewer are taken if a class is smaller.
        seed: sampling seed, so the subset is identical across runs and
            different heads stay comparable.

    Returns:
        (paths, labels) - list of paths, int array of 0/1.
    """
    df = pd.read_csv(manifest_path)

    missing = {"image_path", "label", "split"} - set(df.columns)
    if missing:
        raise ValueError(
            f"manifest {manifest_path!r} is missing required column(s): {sorted(missing)}"
        )

    test_df = df[df["split"] == "test"]
    if test_df.empty:
        raise ValueError(
            f"manifest {manifest_path!r} has no rows with split == 'test' "
            f"(found splits: {sorted(df['split'].unique())})"
        )

    # group_keys=False keeps the original columns flat after sampling.
    eval_df = test_df.groupby("label", group_keys=False).apply(
        lambda g: g.sample(n=min(per_class, len(g)), random_state=seed)
    )

    counts = eval_df.groupby("label").size().to_dict()
    print(f"eval subset: {len(eval_df)} images from the TEST split "
          f"(target {per_class}/class) -> per-class counts {counts}")
    if len(counts) < 2:
        raise ValueError("eval subset has only one class; cannot compute AUC")

    return eval_df["image_path"].tolist(), eval_df["label"].to_numpy().astype(np.int64)


# --------------------------------------------------------------------------
# Robustness sweep
# --------------------------------------------------------------------------

def run_robustness_sweep(model, paths, labels, batch_size=32, transform_names=None,
                         embedder=None, scaler=None):
    """Score the head under every transform, returning a table and cached probs.

    Args:
        model: a Head loaded by ``load_head``.
        paths: eval image paths.
        labels: matching 0/1 labels.
        batch_size: DINOv2 batch size.
        transform_names: defaults to every transform in TRANSFORM_NAMES.
        embedder: reuse an existing DinoV2Embedder.
        scaler: optional fitted StandardScaler, applied to every
            transform's features exactly as it was during training.

    Returns:
        (results_df, probs_by_transform) - the table sorted as
        TRANSFORM_NAMES, and a dict name -> predicted P(fake) array so the
        diagnosis step doesn't have to re-extract features.
    """
    transform_names = transform_names or TRANSFORM_NAMES
    # One backbone for the whole sweep; re-loading it per transform would
    # dominate the runtime.
    embedder = embedder or get_embedder()

    rows, probs_by_transform = [], {}

    # Outer bar counts transforms (x/15), so it's obvious how far through the
    # sweep you are; the per-batch bar from get_fused_embeddings nests inside
    # it and shows progress within the current transform.
    sweep = tqdm(transform_names, desc="Evaluating transforms", unit="transform")

    for name in sweep:
        # ---- the eval loop, one iteration per degradation --------------
        # Each iteration is a full re-extraction: every eval image is opened,
        # degraded by `name`, and pushed through DINOv2 + FFT again. The head
        # itself is fixed - only its input changes - so any movement in the
        # metrics is attributable to the degradation alone.
        #
        # Note this re-runs the backbone 15x over the same images. That is
        # inherent: the transform changes the pixels, so the embeddings cannot
        # be cached across transforms. It is why the subset is kept to ~1000.
        embs = get_fused_embeddings(
            paths,
            transform_name=name,
            batch_size=batch_size,
            embedder=embedder,
            show_progress=True,
        )

        # Apply the training-time scaler, if this head was trained with one.
        # Same fitted object for every transform and every image - it is NEVER
        # re-fit here, which would leak the test distribution into the
        # transform and quietly flatter the results.
        embs = apply_scaler(scaler, embs)

        # Sanity-check the head against the feature width before predicting,
        # so a dimension mismatch reports the cause rather than a raw shape
        # error from inside torch.
        expected = getattr(model, "in_dim", None)
        if expected is not None and embs.shape[1] != expected:
            raise ValueError(
                f"feature width {embs.shape[1]} does not match the head's in_dim "
                f"{expected}. The head was trained on different features - "
                "retrain it, or check FFT_N_BINS / the backbone id."
            )

        probs = predict_proba(model, embs)          # P(fake), shape (N,)
        preds = probs > 0.5                         # 0.5 threshold, as notebook

        acc = accuracy_score(labels, preds)
        auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
        # ---------------------------------------------------------------

        rows.append({"transform": name, "accuracy": acc, "auc": auc})
        probs_by_transform[name] = probs
        # tqdm.write instead of print: a bare print() while a bar is live
        # interleaves with the bar's carriage returns and garbles both. Same
        # text, just routed so tqdm can redraw around it.
        tqdm.write(f"  {name:<14} accuracy {acc:.4f}   auc {auc:.4f}")
        # Keeps the latest score visible on the bar itself.
        sweep.set_postfix(acc=f"{acc:.4f}", auc=f"{auc:.4f}")

    results_df = pd.DataFrame(rows).set_index("transform")

    # Degradation relative to the clean baseline - the column you actually
    # read when judging robustness. Absolute accuracy conflates "the model is
    # good" with "the model is stable".
    if "clean" in results_df.index:
        base_acc = results_df.loc["clean", "accuracy"]
        base_auc = results_df.loc["clean", "auc"]
        results_df["acc_drop_vs_clean"] = base_acc - results_df["accuracy"]
        results_df["auc_drop_vs_clean"] = base_auc - results_df["auc"]

    return results_df, probs_by_transform


def diagnose_transform(model, paths, labels, name, batch_size=32, embedder=None,
                       probs=None, scaler=None):
    """Print a confusion matrix and per-true-class mean P(fake) for one transform.

    Args:
        probs: reuse predictions from the sweep. When None, features are
            re-extracted for this transform.
    """
    if probs is None:
        embs = get_fused_embeddings(
            paths, transform_name=name, batch_size=batch_size,
            embedder=embedder, show_progress=True,
        )
        # Same scaler as the sweep, for the same reason.
        probs = predict_proba(model, apply_scaler(scaler, embs))

    preds = probs > 0.5
    labels = np.asarray(labels)

    print(f"\n--- {name} ---")
    # labels=[0, 1] pins the axis order so the matrix is readable even if a
    # class is never predicted (which happens on the worst degradations).
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    print("confusion matrix [rows=true real/fake, cols=pred real/fake]:")
    print(cm)

    tn, fp, fn, tp = cm.ravel()
    print(f"  real images: {tn} correct, {fp} misflagged as fake")
    print(f"  fake images: {tp} caught,  {fn} missed")

    mean_real = probs[labels == 0].mean() if (labels == 0).any() else float("nan")
    mean_fake = probs[labels == 1].mean() if (labels == 1).any() else float("nan")
    print(f"mean predicted P(fake) for TRUE real images: {mean_real:.3f}")
    print(f"mean predicted P(fake) for TRUE fake images: {mean_fake:.3f}")
    # Separation is the quantity that survives a bad threshold: if it stays
    # wide, recalibrating 0.5 recovers accuracy; if it collapses toward 0, the
    # features genuinely stopped carrying the signal.
    print(f"separation (fake - real): {mean_fake - mean_real:+.3f}")

    return {"transform": name, "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "tp": int(tp), "mean_p_real": float(mean_real),
            "mean_p_fake": float(mean_fake)}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Robustness sweep of a trained head over the held-out test split."
    )
    p.add_argument("--manifest", default=None,
                   help="CSV with image_path,label,split. Only split=='test' rows are used.")
    p.add_argument("--head", default=None,
                   help="Trained head checkpoint (.pt) from src/train.py.")
    # Extras.
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--out", default=None,
                   help="Output CSV (default: <outputs_dir>/robustness_table.csv).")
    p.add_argument("--scaler", default=None,
                   help="Fitted feature scaler (.pkl) from a --normalize training "
                        "run. Default: auto-detected beside --head; if the head was "
                        "not trained normalized, no scaling is applied.")
    p.add_argument("--per-class", type=int, default=500,
                   help="Eval images per class from the test split.")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--diagnose", nargs="*", default=None,
                   help=f"Transforms to diagnose (default: {' '.join(DEFAULT_DIAGNOSE)}).")
    p.add_argument("--skip-diagnose", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)

    manifest = args.manifest or cfg.get("manifest_path") or None
    head_path = args.head or cfg.get("head_path") or None
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.get("batch_size") or 32)
    outputs_dir = cfg.get("outputs_dir") or "outputs"
    out_csv = args.out or os.path.join(outputs_dir, "robustness_table.csv")

    if not manifest:
        raise SystemExit("--manifest is required (or set manifest_path in the config)")
    if not head_path:
        raise SystemExit("--head is required (or set head_path in the config)")
    if not os.path.exists(head_path):
        raise SystemExit(f"head checkpoint not found: {head_path}")

    print(f"loading head: {head_path}")
    model = load_head(head_path)
    print(f"  head in_dim={getattr(model, 'in_dim', '?')} "
          f"hidden={getattr(model, 'hidden', '?')}")

    # Decides from the head's own metadata whether normalized features are
    # expected, so a stale scaler cannot be silently applied to a baseline head.
    scaler, _scaler_path = resolve_feature_scaler(head_path, model, args.scaler)

    paths, labels = build_eval_subset(manifest, per_class=args.per_class, seed=args.seed)

    print(f"\nsweeping {len(TRANSFORM_NAMES)} transforms over {len(paths)} images "
          f"({len(TRANSFORM_NAMES) * len(paths)} feature extractions)...")
    results_df, probs_by_transform = run_robustness_sweep(
        model, paths, labels, batch_size=batch_size, scaler=scaler)

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    results_df.to_csv(out_csv)

    print("\n================ ROBUSTNESS TABLE ================")
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(results_df.round(4).to_string())
    print("==================================================")
    print(f"written -> {out_csv}")

    if not args.skip_diagnose:
        diagnose = args.diagnose if args.diagnose is not None else DEFAULT_DIAGNOSE
        unknown = [n for n in diagnose if n not in TRANSFORM_NAMES]
        if unknown:
            raise SystemExit(f"unknown transform(s) to diagnose: {unknown}")

        print("\n================ DIAGNOSIS ================")
        for name in diagnose:
            # Reuse the sweep's predictions - no need to re-extract.
            diagnose_transform(model, paths, labels, name, batch_size=batch_size,
                               probs=probs_by_transform.get(name), scaler=scaler)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
