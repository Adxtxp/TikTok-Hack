"""Feature extraction: frozen DINOv2 embeddings fused with an FFT profile.

Ported from ``run_on_colab.ipynb`` (the DINOv2 loading cell, the
``fft_feature_vector`` cell, and the batched fused-embedding builders).

Why two feature families
------------------------
The two halves fail in different, complementary ways, which is the whole point
of fusing them:

* **DINOv2 CLS embedding (768-d)** - a large self-supervised ViT. Captures
  semantic / structural implausibility: anatomy that doesn't work, physically
  impossible lighting, texture that isn't quite material. Survives mild
  resampling well, because it was trained on augmented crops.
* **Radial FFT log-magnitude profile (32-d)** - a hand-built frequency
  descriptor. Captures the *generator fingerprint*: upsampling ladders in
  diffusion/GAN decoders leave periodic high-frequency structure that no
  amount of semantic realism hides. Cheap, but fragile - it is exactly what
  JPEG recompression and blur destroy (see ``src/transforms.py``).

A detector on DINOv2 alone misses obvious checkerboard artifacts; one on FFT
alone collapses the moment an image is recompressed. Concatenating gives the
linear head both signals and lets it weight them.

Fused layout
------------
Each image becomes one flat vector, DINOv2 first then FFT::

    [ 0 : 768 ]  DINOv2 CLS token
    [ 768: 800 ]  FFT radial profile (n_bins=32)
    -> FUSED_DIM = 800

The order is fixed and load-bearing: a head trained on this layout must be fed
the same layout at inference time.

Note on feature scale
---------------------
The two blocks are NOT normalised to a common scale before concatenation
(faithful to the notebook). DINOv2 CLS values are roughly unit-ish; the FFT
block is a log-magnitude in the ~0-15 range, so it carries visibly larger
raw magnitudes. The trained linear head absorbs this, but it means you cannot
meaningfully compare raw distances across the two blocks, and standardising
the fused matrix would change results.
"""

import numpy as np
import torch
from PIL import Image
# tqdm.auto picks the ipywidgets bar in a notebook and the plain terminal bar
# in a script, so the same call renders correctly in Colab and on the CLI.
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel

# apply_transform lives in the sibling module. The `src.` form is the intended
# import path (run from the repo root); the bare fallback keeps this module
# usable from inside src/ or from a notebook that has added src/ to sys.path.
try:
    from src.transforms import apply_transform
except ImportError:  # pragma: no cover - depends on sys.path, not on logic
    from transforms import apply_transform

__all__ = [
    "DEFAULT_MODEL_ID",
    "DINO_DIM",
    "FFT_N_BINS",
    "FUSED_DIM",
    "DinoV2Embedder",
    "get_embedder",
    "fft_feature_vector",
    "get_fused_embeddings",
    "save_embeddings",
    "load_embeddings",
]

DEFAULT_MODEL_ID = "facebook/dinov2-base"

# dinov2-base has a 768-wide hidden state; the CLS token is one such vector.
DINO_DIM = 768
FFT_N_BINS = 32
FUSED_DIM = DINO_DIM + FFT_N_BINS  # 800


# --------------------------------------------------------------------------
# DINOv2 backbone
# --------------------------------------------------------------------------

class DinoV2Embedder:
    """Frozen DINOv2 backbone, loaded once and reused.

    The backbone is a fixed feature extractor here - we never fine-tune it,
    only the small head in ``src/model.py`` is trained. So it is put in
    ``.eval()`` (disabling dropout and freezing any norm running-stats
    behaviour) and every parameter gets ``requires_grad = False``, which keeps
    autograd from building a graph through 86M parameters we will never update.

    Loading the model and processor is slow (network fetch + weight init), so
    construct this once and pass it around, or use the module-level
    ``get_embedder()`` cache.
    """

    def __init__(self, model_id=DEFAULT_MODEL_ID, device=None):
        """Load the processor and backbone onto the best available device.

        Args:
            model_id: HuggingFace id. Anything whose hidden size is 768 is
                drop-in; a different size changes FUSED_DIM and invalidates
                any saved head.
            device: torch device string. Defaults to CUDA when available.
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

        # Freeze: inference-only backbone.
        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def dim(self):
        """Width of the embedding this backbone produces (768 for base)."""
        return self.model.config.hidden_size

    @torch.no_grad()
    def embed_batch(self, img_list):
        """Embed a list of PIL RGB images.

        The processor handles resize + normalisation to the model's expected
        224x224 input, so images of differing sizes can share a batch (this is
        what makes the size-changing ``crop_80`` transform safe to pass in).

        Args:
            img_list: list of PIL Images.

        Returns:
            float32 numpy array, shape (len(img_list), DINO_DIM).
        """
        inputs = self.processor(images=img_list, return_tensors="pt").to(self.device)
        # [:, 0] is the CLS token of the final layer - the notebook's choice.
        # Note this is NOT `pooler_output`, which would apply an extra
        # layernorm; swapping them shifts the feature distribution and would
        # invalidate a head trained on the other.
        out = self.model(**inputs).last_hidden_state[:, 0]
        return out.cpu().numpy()

    @torch.no_grad()
    def embed_image(self, img_pil):
        """Embed a single PIL image; returns a 1-D (DINO_DIM,) array."""
        return self.embed_batch([img_pil])[0]


# Process-wide cache so repeated calls don't re-download / re-init the weights.
_EMBEDDER = None


def get_embedder(model_id=DEFAULT_MODEL_ID, device=None):
    """Return a cached DinoV2Embedder, constructing it on first use.

    Re-loads if the requested model_id differs from the cached one.
    """
    global _EMBEDDER
    if _EMBEDDER is None or _EMBEDDER.model_id != model_id:
        _EMBEDDER = DinoV2Embedder(model_id=model_id, device=device)
    return _EMBEDDER


# --------------------------------------------------------------------------
# FFT feature
# --------------------------------------------------------------------------

def fft_feature_vector(img_pil, n_bins=FFT_N_BINS):
    """Radially averaged FFT log-magnitude profile of an image.

    Procedure: grayscale -> 2-D FFT -> shift DC to centre -> log-magnitude ->
    average over concentric rings around the centre. Ring 0 is the lowest
    frequency (broad structure), the last ring the highest (fine detail and
    resampling artifacts).

    Because the rings are computed as a fraction of each image's own maximum
    radius, the descriptor is a fixed ``n_bins`` long regardless of input
    resolution - which is what lets differently sized images (e.g. after
    ``crop_80``) share one feature matrix.

    Args:
        img_pil: PIL Image; converted to grayscale internally.
        n_bins: number of radial bins / output length.

    Returns:
        float64 numpy array of shape (n_bins,). Bins containing no pixels stay
        0.0 (only reachable for degenerate, near-1px images).
    """
    gray = np.array(img_pil.convert("L"), dtype=np.float32)

    # Centre the spectrum so radius from the middle == spatial frequency.
    fshift = np.fft.fftshift(np.fft.fft2(gray))
    # log() compresses the enormous dynamic range of an image spectrum; the
    # epsilon guards log(0) at nulls in the spectrum.
    magnitude = np.log(np.abs(fshift) + 1e-8)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    # Integer radius of every pixel from the centre.
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

    max_r = r.max()
    bin_edges = np.linspace(0, max_r, n_bins + 1)

    profile = np.zeros(n_bins)
    for i in range(n_bins):
        # Half-open rings [edge_i, edge_i+1). Note the outermost few corner
        # pixels at exactly r == max_r fall outside the last ring; kept as-is
        # to stay bit-compatible with the notebook's features.
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        if mask.sum() > 0:
            profile[i] = magnitude[mask].mean()
    return profile


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------

def get_fused_embeddings(
    image_paths,
    transform_name="clean",
    batch_size=32,
    n_fft_bins=FFT_N_BINS,
    model_id=DEFAULT_MODEL_ID,
    embedder=None,
    show_progress=True,
):
    """Build the fused feature matrix for a list of image paths.

    For each batch: open -> apply the named degradation -> DINOv2 + FFT ->
    concatenate. Both feature families are computed on the *transformed*
    image, so a robustness sweep measures the effect of the degradation on the
    full pipeline rather than on the backbone alone.

    Args:
        image_paths: sequence of paths readable by PIL.
        transform_name: key from ``src.transforms.TRANSFORM_NAMES``.
            "clean" is the undegraded baseline.
        batch_size: images per DINOv2 forward pass. Lower it if VRAM is tight.
        n_fft_bins: FFT profile length. Changing it changes the fused width.
        model_id: backbone to use, if no explicit embedder is given.
        embedder: an existing DinoV2Embedder to reuse. Defaults to the
            module-level cached one.
        show_progress: draw a tqdm bar.

    Returns:
        float32 numpy array, shape (len(image_paths), DINO_DIM + n_fft_bins).
    """
    embedder = embedder or get_embedder(model_id)

    # The slowest loop in the pipeline: one DINOv2 forward pass per batch, so
    # thousands of them over a full manifest. The bar is what distinguishes
    # "still working" from "hung" on a long Colab run.
    #
    # leave=False so the finished bar is erased - this function is called once
    # per transform by the 15-transform sweep in src/evaluate.py, and 15
    # leftover bars would bury the results table.
    batch_starts = range(0, len(image_paths), batch_size)
    if show_progress:
        batch_starts = tqdm(
            batch_starts,
            desc=f"Extracting features [{transform_name}]",
            unit="batch",
            leave=False,
        )

    all_embeddings = []
    for i in batch_starts:
        batch_paths = image_paths[i:i + batch_size]

        # convert("RGB") before transforming: drops alpha and normalises
        # palette/grayscale sources to 3 channels.
        batch_imgs = [
            apply_transform(Image.open(p).convert("RGB"), transform_name)
            for p in batch_paths
        ]

        # ---- the fusion step -------------------------------------------
        # Two independent views of the same batch, joined along the feature
        # axis (axis=1) so row i of the result still corresponds to
        # batch_paths[i]. DINOv2 block first, FFT block second - this order
        # defines the layout the trained head expects, so it must not change.
        dino_feats = embedder.embed_batch(batch_imgs)                       # (B, 768)
        fft_feats = np.stack(
            [fft_feature_vector(img, n_bins=n_fft_bins) for img in batch_imgs]
        )                                                                   # (B, 32)
        fused = np.concatenate([dino_feats, fft_feats], axis=1)             # (B, 800)
        # ----------------------------------------------------------------

        all_embeddings.append(fused)

    if not all_embeddings:
        # Keep the shape contract even for an empty input list.
        return np.zeros((0, DINO_DIM + n_fft_bins), dtype=np.float32)

    return np.concatenate(all_embeddings, axis=0).astype(np.float32)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def save_embeddings(path, embeddings, labels=None, splits=None):
    """Save a fused feature matrix (and optional metadata) to a .npz.

    Uses the notebook's archive keys - ``embeddings``, ``labels``, ``splits``
    - so files written here stay readable by the original notebook and vice
    versa. Extraction is the expensive step in this pipeline, so this cache is
    what makes training and evaluation cheap to re-run.

    Args:
        path: destination .npz path.
        embeddings: (N, D) array.
        labels: optional (N,) array of 0=real / 1=fake.
        splits: optional (N,) array of "train"/"test" strings.
    """
    arrays = {"embeddings": embeddings}
    if labels is not None:
        arrays["labels"] = np.asarray(labels)
    if splits is not None:
        arrays["splits"] = np.asarray(splits)
    np.savez(path, **arrays)


def load_embeddings(path):
    """Load a .npz written by ``save_embeddings``.

    Returns:
        (embeddings, labels, splits). ``labels`` / ``splits`` are None if the
        archive doesn't carry them.
    """
    # allow_pickle=True mirrors the notebook. Plain numeric/string arrays
    # don't need it, but it keeps older archives loadable.
    data = np.load(path, allow_pickle=True)
    return (
        data["embeddings"],
        data["labels"] if "labels" in data else None,
        data["splits"] if "splits" in data else None,
    )
