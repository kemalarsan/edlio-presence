"""
renderer/gfpgan_pass.py — Optional GFPGAN post-sharpening for the composite renderer.

Why: MuseTalk V1.5 generates the face patch at 256×256. On a 1024×1024
portrait with a 500×500 face bbox, that patch gets upsampled ~2× before
compositing, which is where the mouth-area softness the user sees comes
from. GFPGAN (a face-restoration GAN trained to take low-quality face
crops and produce 512×512 photorealistic faces) is the canonical post-pass
for exactly this problem.

Where: between MuseTalk's 256×256 output and the `get_image` composite step.
Input  = MuseTalk's 256×256 face patch (already in the face crop colour
         space), containing generated lips + resampled original face.
Output = 256×256 sharper face patch; passed to the existing blend so only
         the mouth portion actually replaces original pixels.

This matches the "restore *then* blend" pattern recommended in the MuseTalk
README (`Restore step` in their post-processing section), but implemented
inline rather than as a separate CLI invocation.

The GFPGAN weights (GFPGANv1.4.pth, ~334 MB) are not in the base pod image.
They live on the network volume next to the other MuseTalk weights:
    {model_dir}/gfpgan/GFPGANv1.4.pth
`fetch_gfpgan_weights()` downloads them on demand.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


_GFPGAN_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
)
_GFPGAN_FNAME = "GFPGANv1.4.pth"


def _download(url: str, dst: Path) -> None:
    import urllib.request

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".partial")
    logger.info("Downloading GFPGAN weights from %s …", url)
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as w:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            w.write(chunk)
    tmp.rename(dst)
    logger.info("GFPGAN weights ready at %s (%.1f MB)", dst, dst.stat().st_size / (1024 * 1024))


def fetch_gfpgan_weights(model_dir: Path) -> Path:
    """
    Ensure GFPGANv1.4.pth exists on the weights volume. Returns its full path.
    """
    weights_path = Path(model_dir) / "gfpgan" / _GFPGAN_FNAME
    if weights_path.exists() and weights_path.stat().st_size > 100 * 1024 * 1024:
        return weights_path
    _download(_GFPGAN_URL, weights_path)
    return weights_path


class GFPGANSharpener:
    """
    Wraps a GFPGANer for post-sharpening MuseTalk's 256×256 output.

    Call:
        s = GFPGANSharpener(model_dir=Path(...), weight=0.5)
        sharpened_256 = s.enhance(muse_talk_output_256)

    `weight` is GFPGAN's adjustable identity/quality trade-off:
        0.0 → pure input (no effect)
        0.5 → balanced (default)
        1.0 → maximum restoration (may change identity slightly)

    For talking-face use: keep `weight` low-ish (0.3–0.5) so the restoration
    doesn't override MuseTalk's lip shape.
    """

    def __init__(
        self,
        *,
        model_dir: Path,
        device: str = "cuda",
        weight: float = 0.5,
    ) -> None:
        self.device = device
        self.weight = float(weight)
        self.model_dir = Path(model_dir)

        # Lazy imports so the module is importable without GFPGAN installed.
        from gfpgan import GFPGANer  # noqa: PLC0415

        weights_path = fetch_gfpgan_weights(self.model_dir)
        logger.info("Initialising GFPGAN (weights=%s, device=%s)", weights_path, device)

        # arch='clean' matches GFPGANv1.4.pth (the "clean" architecture).
        # upscale=1: we don't want super-resolution; we'll paste back at 256.
        self._gfp = GFPGANer(
            model_path=str(weights_path),
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=device,
        )
        logger.info("GFPGAN ready. weight=%.2f", self.weight)

    def enhance(self, face_rgb_256: np.ndarray) -> np.ndarray:
        """
        Sharpen a 256×256 RGB face patch. Returns a 256×256 RGB face patch.

        GFPGANer.enhance(has_aligned=True) resizes inputs to 512 internally,
        runs the restorer, and returns a 512×512 face. We downsample back
        to 256 so the caller's compositing pipeline doesn't change.
        """
        import cv2  # noqa: PLC0415

        # GFPGAN expects BGR uint8.
        face_bgr = cv2.cvtColor(face_rgb_256, cv2.COLOR_RGB2BGR)

        _cropped_faces, restored_faces, _restored_img = self._gfp.enhance(
            face_bgr,
            has_aligned=True,
            only_center_face=False,
            paste_back=False,   # we don't want auto-paste, we'll do it via blending
            weight=self.weight,
        )
        if not restored_faces:
            logger.warning("GFPGAN produced no restored face; returning input unchanged.")
            return face_rgb_256

        restored_bgr_512 = restored_faces[0]
        restored_bgr_256 = cv2.resize(
            restored_bgr_512, (256, 256), interpolation=cv2.INTER_LANCZOS4
        )
        return cv2.cvtColor(restored_bgr_256, cv2.COLOR_BGR2RGB)
