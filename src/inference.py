"""Batch inference: score a folder of unlabelled images for AI-generation.

This is the deliverable entry point. The notebook's demo cell invokes it as::

    !python inference.py --image_dir /content/my_test_images \
                         --weights /content/head_fused.pt \
                         --output /content/predictions.json

and the following cell consumes the result with::

    pd.DataFrame(json.load(f)).merge(ground_truth, on="image_path")
    merged["predicted_label"] = (merged["pred"] > 0.5).astype(int)

So two things about the output contract are load-bearing and must not drift:

* the JSON document is a **list** of objects (not a dict keyed by path), so
  ``pd.DataFrame`` gets one row per image;
* each object has exactly the keys ``image_path`` (str) and ``pred``
  (float in 0..1, = P(AI-generated)). ``image_path`` is the join key for the
  ground-truth merge and ``pred`` is thresholded at 0.5 downstream.

The flag names use underscores (``--image_dir``, not ``--image-dir``) to match
the notebook cell verbatim.

Pipeline per image: open as RGB -> "clean" transform (identity) -> frozen
DINOv2 CLS + radial FFT profile -> fused 800-d vector -> trained head -> one
logit -> sigmoid -> P(AI-generated).

No labels are read or required. Unreadable files are reported and skipped
rather than guessed at, so a corrupt image cannot silently become a
fabricated prediction.
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

try:
    from src.features import get_fused_embeddings
    from src.model import load_head, predict_proba
except ImportError:  # pragma: no cover - sys.path shape, not logic
    # Lets the script run as `python src/inference.py` or, as the notebook
    # does, as `python inference.py` from inside the src/ directory.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.features import get_fused_embeddings
    from src.model import load_head, predict_proba

# Superset of the manifest builder's extensions (which used jpg/jpeg/png), so
# an unexpected format in a demo folder isn't silently ignored.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def find_images(image_dir, exts=IMAGE_EXTS):
    """Recursively collect image paths under ``image_dir``.

    Sorted so the output JSON has a stable, reproducible row order - handy
    when diffing two runs.

    Returns:
        list of path strings.
    """
    if not os.path.isdir(image_dir):
        raise SystemExit(f"--image_dir is not a directory: {image_dir!r}")

    found = []
    for root, _dirs, files in os.walk(image_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in exts:
                found.append(os.path.join(root, fn))
    return sorted(found)


def filter_readable(paths):
    """Split paths into (readable, unreadable).

    Verified up front because ``get_fused_embeddings`` opens images in
    batches: one corrupt file would otherwise abort the whole run partway
    through, losing every prediction computed so far.

    ``Image.verify()`` consumes the file handle, so the image is reopened
    later for the actual forward pass - this is a cheap header/CRC check, not
    a full decode.
    """
    ok, bad = [], []
    for p in paths:
        try:
            with Image.open(p) as im:
                im.verify()
            ok.append(p)
        except Exception as exc:  # noqa: BLE001 - any decode failure disqualifies
            bad.append((p, f"{type(exc).__name__}: {exc}"))
    return ok, bad


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------

def predict_paths(paths, weights_path, batch_size=32):
    """Score a list of image paths, returning P(AI-generated) per image.

    Args:
        paths: image paths, in the order the output should follow.
        weights_path: head checkpoint (.pt) from src/train.py, or the
            notebook's bare state_dict.
        batch_size: images per DINOv2 forward pass.

    Returns:
        float64 numpy array of shape (len(paths),), values in [0, 1].
    """
    model = load_head(weights_path)
    in_dim = getattr(model, "in_dim", None)
    print(f"loaded head: {weights_path}  (in_dim={in_dim})")

    # transform_name="clean" - the identity. Inference scores images as they
    # arrive; the degradation transforms exist only for the robustness sweep
    # in src/evaluate.py and for --augment training.
    features = get_fused_embeddings(
        paths, transform_name="clean", batch_size=batch_size, show_progress=True
    )

    # Fail with a readable message if the head predates a feature change,
    # rather than surfacing a bare matmul shape error from torch.
    if in_dim is not None and features.shape[1] != in_dim:
        raise SystemExit(
            f"feature width {features.shape[1]} != head in_dim {in_dim}: this head "
            "was trained on different features. Retrain it, or check that "
            "FFT_N_BINS and the DINOv2 model id match the training run."
        )

    probs = predict_proba(model, features)

    # SINGLE-IMAGE SAFETY.
    # The notebook's inference path did `.numpy().squeeze()`, which on a
    # 1-image batch turns (1, 1) into a 0-d scalar - then `len(probs)` raises
    # and `probs[0]` is a TypeError. predict_proba squeezes only the last axis
    # (-1) so N=1 stays (1,), and atleast_1d is a belt-and-braces guard in
    # case a caller passes something that still collapsed.
    probs = np.atleast_1d(np.asarray(probs, dtype=np.float64))

    if probs.shape[0] != len(paths):
        raise RuntimeError(
            f"got {probs.shape[0]} predictions for {len(paths)} images - "
            "row alignment broke, refusing to write a misaligned file"
        )
    return probs


def build_records(paths, probs):
    """Assemble the required JSON payload.

    Exactly two keys per record - ``image_path`` and ``pred`` - because the
    notebook merges on the former and thresholds the latter. float() converts
    numpy float64 to a plain Python float so json.dump accepts it.
    """
    return [
        {"image_path": p, "pred": float(prob)}
        for p, prob in zip(paths, probs)
    ]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Predict P(AI-generated) for every image in a folder."
    )
    # Underscored names, matching the notebook's demo cell exactly.
    p.add_argument("--image_dir", required=True,
                   help="Folder of images to score (searched recursively). No labels needed.")
    p.add_argument("--weights", required=True,
                   help="Trained head checkpoint (.pt).")
    p.add_argument("--output", required=True,
                   help="Destination JSON path.")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Images per DINOv2 forward pass (default: 32).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not os.path.exists(args.weights):
        raise SystemExit(f"--weights not found: {args.weights}")

    paths = find_images(args.image_dir)
    print(f"found {len(paths)} image(s) under {args.image_dir}")

    readable, unreadable = filter_readable(paths)
    if unreadable:
        # Loud, itemised, and non-fatal: the run still produces predictions
        # for everything that could be decoded.
        print(f"WARNING: skipping {len(unreadable)} unreadable file(s):")
        for p, why in unreadable:
            print(f"  - {p}  ({why})")

    if not readable:
        # Still emit a valid empty list so a downstream json.load() succeeds
        # instead of hitting a missing file.
        print("no readable images; writing an empty prediction list")
        records = []
    else:
        probs = predict_paths(readable, args.weights, batch_size=args.batch_size)
        records = build_records(readable, probs)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"\nwrote {len(records)} prediction(s) -> {args.output}")
    if records:
        preds = np.array([r["pred"] for r in records])
        n_ai = int((preds > 0.5).sum())
        print(f"  P(AI-generated): min {preds.min():.4f}  mean {preds.mean():.4f}  "
              f"max {preds.max():.4f}")
        print(f"  flagged AI-generated at the 0.5 threshold: {n_ai}/{len(preds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
