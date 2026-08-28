# Robust AI-Generated Image Detection

Detecting AI-generated images is easy on pristine generator output and hard on
anything that has been through a social platform. Every upload gets
re-encoded, resized, filtered and re-cropped, and those steps destroy exactly
the high-frequency fingerprints that most detectors rely on. A model that
scores 99% on clean data and 60% after a JPEG round-trip is not useful in
production.

This project builds a detector and then **measures how much each real-world
degradation actually costs it**, producing a per-transform robustness table as
the headline deliverable rather than a single clean-data accuracy number.

Dataset: [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
(60k real CIFAR-10 images, 60k Stable-Diffusion-generated counterparts).
Labels: `0` = real, `1` = AI-generated.

## Approach

**Two complementary feature families, fused.**

| feature | dim | catches | weakness |
|---|---|---|---|
| Frozen DINOv2 (`facebook/dinov2-base`) CLS token | 768 | semantic and structural implausibility — anatomy, lighting, material texture | can miss clean-looking synthetic images |
| Radial FFT log-magnitude profile | 32 | the generator fingerprint — periodic upsampling artifacts from diffusion/GAN decoders | destroyed by JPEG and blur |

Concatenated into one **800-d** vector per image. The two halves fail in
different ways: a DINOv2-only detector misses obvious checkerboard artifacts,
an FFT-only detector collapses the moment an image is recompressed. Fusing
them lets the classifier weigh both.

**A small trained head, a frozen backbone.** The DINOv2 backbone is never
fine-tuned — `requires_grad = False`, `.eval()`. Only a
`Linear(800→128) → ReLU → Dropout(0.2) → Linear(128→1)` head is trained, on a
single logit with `BCEWithLogitsLoss`, so `sigmoid(logit)` is P(AI-generated)
and the decision threshold stays explicit and tunable. With ~103k trainable
parameters against the backbone's 86M, this is what keeps the model from
overfitting a hackathon-sized subset — and it means training runs in seconds
once features are cached.

**Augmented training for robustness.** `--augment` applies a random transform
from the 15-transform grid to each training image before feature extraction,
so the head learns a boundary that survives degradation instead of one that
depends on pristine artifacts. Validation deliberately stays clean, so the
metric remains comparable across runs.

**The 15-transform evaluation grid** (`src/transforms.py`), each mapping to a
real pipeline step:

| family | settings | real-world analog |
|---|---|---|
| `clean` | — | control / baseline |
| `jpeg_*` | quality 90, 70, 50, 30 | every social upload re-encodes lossily |
| `blur_*` | sigma 0.5, 1.0, 2.0 | denoising, "beautify" smoothing, upscaler artifacts |
| `resize_*` | scale 0.5, 0.25 round-trip | platform thumbnail served then re-displayed |
| `noise_*` | std 0.02, 0.05, 0.10 | sensor noise, low light, cheap deliberate attack |
| `color_jitter` | ±20% brightness/contrast/saturation | filters, auto-tone, screenshotting |
| `crop_80` | keep 80% per side | reframing to 9:16, watermark removal |

Each is pinned to a single deterministic strength (`p=1.0`, equal range
endpoints) so the robustness table is reproducible.

**Honest evaluation.** The test split is loaded by exactly one script,
`src/evaluate.py`. `src/train.py` filters it out at read time and carves its
validation set from the training rows only, so no model-selection decision is
informed by test data.

## Repository layout

```
src/
  transforms.py   15 degradation transforms + apply_transform / TRANSFORM_NAMES
  features.py     frozen DINOv2 + radial FFT, fused to 800-d, .npz caching
  model.py        the Head, predict_proba, save_head / load_head
  train.py        CLI: train/val split, mini-batch Adam, --augment
  evaluate.py     CLI: robustness sweep -> CSV, confusion-matrix diagnosis
  inference.py    CLI: image folder -> predictions.json
notebooks/
  run_on_colab.ipynb   thin GPU orchestrator; no model logic
configs/default.yaml   model id, batch size, epochs, lr, val_split, paths
outputs/               generated artifacts (gitignored)
run_on_colab.ipynb     the original exploratory notebook, kept for provenance
```

## Setup

Requires Python 3.9+ and, realistically, a CUDA GPU — feature extraction is
one DINOv2 forward pass per image, and the robustness sweep repeats that 15
times.

```bash
git clone https://github.com/Adxtxp/TikTok-Hack.git
cd TikTok-Hack
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`torch` and `torchvision` are intentionally unpinned so pip resolves a build
matching your CUDA driver (and so Colab's preinstalled build is left alone).
**`albumentations` must be >= 2.0** — the transforms use the `quality_range`
and `std_range` argument names introduced there.

For Colab, skip all of the above and open `notebooks/run_on_colab.ipynb`.

## Reproducing the results

### 1. Build the manifest

Every script reads a CSV with columns `image_path`, `label` (0=real, 1=fake),
`split` (`train`/`test`). Cell 3 of `notebooks/run_on_colab.ipynb` downloads
CIFAKE via `kagglehub` and writes it; label and split are inferred from the
dataset's `train/REAL`, `train/FAKE`, `test/REAL`, `test/FAKE` layout. The
dataset's own train/test division is preserved, never re-drawn.

Result: `manifest.csv`.

### 2. Train

```bash
python -m src.train \
  --manifest manifest.csv \
  --out outputs/head_fused.pt \
  --epochs 30 \
  --cache-embeddings outputs/embeddings_fused.npz
```

Trains on `split == "train"` rows only, holding back a stratified 15%
(`val_split` in `configs/default.yaml`) for per-epoch validation accuracy/AUC.
`--cache-embeddings` saves the extracted features so later runs skip the
expensive step.

For the robustness-trained variant:

```bash
python -m src.train --manifest manifest.csv --out outputs/head_augmented.pt --epochs 30 --augment
```

### 3. Evaluate robustness — the headline deliverable

```bash
python -m src.evaluate \
  --manifest manifest.csv \
  --head outputs/head_fused.pt \
  --out outputs/robustness_table.csv
```

Sweeps all 15 transforms over a class-balanced sample of the held-out test
split (500/class), writing accuracy, AUC, and drop-vs-clean per transform to
`outputs/robustness_table.csv`. Then prints a confusion matrix and mean
predicted P(fake) per true class for `clean`, `blur_2.0`, `resize_0.25` and
`noise_0.10` — the diagnosis that distinguishes "threshold drifted"
(recoverable by recalibration) from "signal destroyed" (not).

### 4. Infer on new images

```bash
python -m src.inference \
  --image_dir data/my_test_images \
  --weights outputs/head_fused.pt \
  --output outputs/predictions.json
```

Recursively scores a folder of unlabelled images. Output is a JSON list of
`{"image_path": <str>, "pred": <float 0..1>}`, where `pred` is
P(AI-generated); threshold at 0.5 for a hard label.

## Limitations

**Dataset.** CIFAKE is 32×32 CIFAR-10-derived imagery upscaled to DINOv2's
224×224 input. Results will not transfer directly to high-resolution
photographs, and the FFT profile in particular is sensitive to that upscaling.
The fake class comes from a single generator family (Stable Diffusion), so
generalisation to other generators (Midjourney, GANs, newer diffusion models)
is untested and should be assumed weak — this is the single biggest caveat.

**Feature scaling.** The DINOv2 and FFT blocks are concatenated without
normalisation to a common scale (faithful to the original notebook). DINOv2
CLS values are roughly unit-scale; the FFT block is a log-magnitude around
7–9. The trained head absorbs this, but raw distances are not comparable
across the two blocks, and standardising the fused matrix would change
results.

**Augmentation is single-hop.** `--augment` applies one transform per image.
Real upload chains compose several (resize, then JPEG, then re-encode).
`src/train.py` marks the extension point (`TODO(extend me)`) for composed
chains, weighted sampling, and per-epoch resampling.

**Threshold is fixed at 0.5** everywhere and never calibrated. Under heavy
degradation the score distributions shift, so a tuned per-condition threshold
would recover some of the reported accuracy loss.

**`crop_80` changes resolution.** It is the only transform that does, which is
harmless (the DINOv2 processor resizes anyway) but means feature arrays are not
shape-comparable across transforms.

**Evaluation subset size.** The robustness sweep uses ~1000 test images, not
the full test split, because it re-extracts features 15 times. Per-cell
confidence intervals are correspondingly wide — treat small differences
between adjacent transforms as noise.

**Not validated end-to-end at time of writing.** The modules are ported and
their data-handling logic is unit-tested, but no full GPU training or
evaluation run has been completed against real weights; the numbers the
pipeline produces have yet to be recorded here.

## Results

_To be filled in after the first full GPU run._

| transform | accuracy | AUC | drop vs clean |
|---|---|---|---|
| clean | — | — | — |
| … | | | |

## Team contributions

_Fill in before submission._

| member | contributions |
|---|---|
| _name_ | _e.g. feature fusion, DINOv2 integration_ |
| _name_ | _e.g. robustness transform grid, evaluation harness_ |
| _name_ | _e.g. training pipeline, augmentation strategy_ |
| _name_ | _e.g. inference CLI, Colab orchestration, README_ |

## Acknowledgements

- [CIFAKE dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) — Bird & Lotfi
- [DINOv2](https://github.com/facebookresearch/dinov2) — Meta AI Research
- [albumentations](https://albumentations.ai/) for the degradation transforms
