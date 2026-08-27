"""Image degradation transforms for AI-image-detector robustness evaluation.

Ported (parameters unchanged) from the ``TRANSFORMS`` cell of
``run_on_colab.ipynb``.

The point of this module is stress testing, not training augmentation: a real
detector sees images that have been through a social media pipeline, not
pristine generator output. Each family below mimics one step of that pipeline,
so a per-transform accuracy table shows which kinds of real-world mangling
actually break the model.

Transform families and their real-world analogs
-----------------------------------------------
clean
    Identity. Baseline / control - no degradation applied.
jpeg_90 / jpeg_70 / jpeg_50 / jpeg_30
    JPEG recompression at decreasing quality. Every upload to TikTok /
    Instagram / WhatsApp re-encodes lossily, which is the most common
    destroyer of generator fingerprints - it attacks exactly the
    high-frequency detail an FFT feature reads.
blur_0.5 / blur_1.0 / blur_2.0
    Gaussian blur, sigma in pixels. Soft focus, denoising filters, "beautify"
    skin smoothing, and upscaler side effects.
resize_0.5 / resize_0.25
    Downscale then upscale back to the original size ("resize round-trip").
    Mimics a platform serving a low-res thumbnail that is then displayed or
    re-saved at full size: fine detail is destroyed while the original
    dimensions are kept.
noise_0.02 / noise_0.05 / noise_0.10
    Additive Gaussian noise, std as a FRACTION of full scale (0.02 is about
    5/255). Sensor noise, low-light capture, and the cheapest deliberate
    attack on a frequency-domain detector.
color_jitter
    Brightness / contrast / saturation +/-20% (hue untouched). Filters,
    auto-tone, screenshotting, HDR display re-grading.
crop_80
    Center crop keeping 80% of each side. Reframing, aspect-ratio cropping to
    9:16, and removal of watermarks or borders. NOTE: this is the only
    transform that changes the output resolution.

Public API
----------
TRANSFORMS
    name -> transform (albumentations object or plain callable)
TRANSFORM_NAMES
    list of every transform name, in display order
apply_transform
    (PIL RGB image, name) -> PIL RGB image

Requires albumentations >= 2.0: the ``quality_range`` and ``std_range``
keyword arguments used below replaced the 1.x ``quality_lower`` / ``var_limit``
spellings.
"""

import albumentations as A
import cv2
import numpy as np
from PIL import Image

__all__ = [
    "TRANSFORMS",
    "TRANSFORM_NAMES",
    "PLAIN_FUNCS",
    "apply_transform",
    "resize_roundtrip",
    "center_crop",
]


# --------------------------------------------------------------------------
# Plain numpy helpers
#
# These two operate directly on HxWx3 uint8 arrays rather than going through
# albumentations, so they are registered in PLAIN_FUNCS below and called with
# a different signature than the albumentations transforms.
#
# Both use only channel-agnostic cv2 ops (resize / slicing), so passing RGB
# instead of cv2's native BGR is safe here - no red/blue swap is possible.
# --------------------------------------------------------------------------

def resize_roundtrip(img_np, scale):
    """Downscale by ``scale``, then upscale back to the original size.

    Detail lost in the downscale does not come back, so the image ends up
    softened while keeping its original dimensions - which is what a platform
    thumbnail pipeline does to an upload.
    """
    h, w = img_np.shape[:2]
    # Clamp to >=1px so extreme scales on tiny images can't produce a
    # zero-size array (cv2.resize raises on a zero dimension).
    small = cv2.resize(
        img_np,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def center_crop(img_np, frac):
    """Keep the central ``frac`` of each side, discarding the border.

    Unlike every other transform here, this returns a SMALLER image
    (frac * H, frac * W). Downstream feature extractors resize to a fixed
    input size anyway, so that is harmless - but don't assume the shape is
    preserved if you compare arrays directly.
    """
    h, w = img_np.shape[:2]
    nh, nw = int(h * frac), int(w * frac)
    top, left = (h - nh) // 2, (w - nw) // 2
    return img_np[top:top + nh, left:left + nw]


# Names whose TRANSFORMS value is a plain callable taking the array
# positionally, i.e. f(img_np), rather than an albumentations transform
# requiring the f(image=img_np)["image"] calling convention.
PLAIN_FUNCS = {"clean", "resize_0.5", "resize_0.25", "crop_80"}


# --------------------------------------------------------------------------
# The transform registry.
#
# Every albumentations transform is pinned to a single deterministic strength
# (both ends of each *_range / *_limit tuple are equal) with p=1.0, so it
# always fires at exactly the documented severity. This is an evaluation grid,
# not random augmentation - a random strength would make the robustness table
# unreproducible.
#
# Insertion order is meaningful: it is the order of TRANSFORM_NAMES and hence
# of report rows / plot panels. Baseline first, then each family from mildest
# to most severe.
# --------------------------------------------------------------------------

TRANSFORMS = {
    # Control: no degradation.
    "clean": lambda img: img,

    # JPEG recompression, quality 90 -> 30 (lower = more artifacts).
    "jpeg_90": A.ImageCompression(quality_range=(90, 90), compression_type="jpeg", p=1.0),
    "jpeg_70": A.ImageCompression(quality_range=(70, 70), compression_type="jpeg", p=1.0),
    "jpeg_50": A.ImageCompression(quality_range=(50, 50), compression_type="jpeg", p=1.0),
    "jpeg_30": A.ImageCompression(quality_range=(30, 30), compression_type="jpeg", p=1.0),

    # Gaussian blur at fixed sigma (pixels). blur_limit=0 lets albumentations
    # derive the kernel size from sigma instead of randomising it.
    "blur_0.5": A.GaussianBlur(sigma_limit=(0.5, 0.5), blur_limit=0, p=1.0),
    "blur_1.0": A.GaussianBlur(sigma_limit=(1.0, 1.0), blur_limit=0, p=1.0),
    "blur_2.0": A.GaussianBlur(sigma_limit=(2.0, 2.0), blur_limit=0, p=1.0),

    # Resize round-trip: 2x and 4x detail loss.
    "resize_0.5": lambda img: resize_roundtrip(img, 0.5),
    "resize_0.25": lambda img: resize_roundtrip(img, 0.25),

    # Additive Gaussian noise; std is a fraction of full scale, not 0-255.
    "noise_0.02": A.GaussNoise(std_range=(0.02, 0.02), p=1.0),
    "noise_0.05": A.GaussNoise(std_range=(0.05, 0.05), p=1.0),
    "noise_0.10": A.GaussNoise(std_range=(0.10, 0.10), p=1.0),

    # +/-20% brightness, contrast and saturation. hue=0.0 on purpose: a hue
    # shift is a far less common real-world edit and would confound the
    # colour-statistics side of the feature vector.
    "color_jitter": A.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=1.0
    ),

    # Center crop keeping 80% of each side.
    "crop_80": lambda img: center_crop(img, 0.8),
}

# Iteration order for callers (evaluate.py, plotting, report tables).
# Derived from TRANSFORMS so the two can never drift apart.
TRANSFORM_NAMES = list(TRANSFORMS.keys())


def apply_transform(img_pil, name):
    """Apply the named transform to a PIL image and return a PIL RGB image.

    Dispatches on ``name`` because the two kinds of registry entry have
    incompatible calling conventions: plain callables take the array
    positionally, while albumentations transforms are keyword-called and
    return a dict.

    Args:
        img_pil: PIL Image. Converted to RGB if it isn't already, so callers
            can hand over palette / grayscale / RGBA images without producing
            a non-3-channel array that ColorJitter would reject.
        name: key of TRANSFORMS (see TRANSFORM_NAMES).

    Returns:
        A new PIL RGB Image. Same size as the input for every transform
        except "crop_80".

    Raises:
        KeyError: if ``name`` is not a known transform.
    """
    if name not in TRANSFORMS:
        raise KeyError(
            "unknown transform {!r}; expected one of {}".format(name, TRANSFORM_NAMES)
        )

    # Guarantee HxWx3 uint8 regardless of what the caller opened.
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
    img_np = np.array(img_pil)

    t = TRANSFORMS[name]
    out = t(img_np) if name in PLAIN_FUNCS else t(image=img_np)["image"]
    return Image.fromarray(out)
