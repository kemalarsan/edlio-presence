"""
renderer/composite.py — Full-portrait MuseTalk renderer with face-region compositing.

This renderer fixes the "blurry 256×256 postage stamp" problem in
renderer/engine.py:_MuseTalkRenderer, which just resized the entire portrait
to 256×256 and returned the raw face patch.

What this does instead (matches the reference flow in
renderer/muse_talk_vendor/app.py):

    1. Keep the original portrait at its native resolution (e.g. 1024×1024)
    2. Crop JUST the face region, resize to 256×256 → run MuseTalk
    3. Resize the predicted 256×256 face back to the original face-region size
    4. Composite it back into the full portrait via blending.get_image()
       (feathered mouth-region mask, BiSeNet face parsing)

Result: the whole portrait is sharp, only the mouth region is re-generated
content, and the seam is soft-blended.

Face bounding-box: we *do not* run DWPose / face_alignment in this path —
those modules have hostile module-level initialisation + heavy deps
(mmpose, face_alignment) that aren't in the pod image. Instead the caller
passes an explicit bbox from the /render request. The server resolves the
bbox with OpenCV's bundled Haar cascade on first sight of a portrait and
caches it alongside the portrait.

This keeps the hot path dep-free: we only need the BiSeNet face-parsing
network for `get_image()`, and that's already on the weights volume.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .engine import AudioFeatures

logger = logging.getLogger(__name__)

_VENDOR_DIR = Path(__file__).parent / "muse_talk_vendor"


class MuseTalkCompositeRenderer:
    """
    MuseTalk renderer that returns the full portrait with just the mouth
    region regenerated and blended back in.

    Parameters
    ----------
    portrait_path : str
        Path to the portrait image on disk. Any cv2-readable format.
    face_box : (x1, y1, x2, y2)
        Face bounding box in the portrait's pixel coordinates. Closed/closed.
    device : str
        "cuda" | "cpu". MuseTalk needs "cuda" in practice.
    model_dir : str | Path
        Path to MuseTalk weights root. Must contain musetalkV15/, sd-vae/,
        whisper/, and face-parse-bisent/.
    use_float16 : bool
        Run UNet + VAE + Whisper in fp16. Always on for A5000/4090.
    extra_margin : int
        Extra pixels added to y2 to include the chin. Matches MuseTalk's
        default of 10 in app.py.
    parsing_mode : str
        Passed through to blending.get_image(). "jaw" is the V1.5 default.
    left_cheek_width : int
    right_cheek_width : int
        FaceParsing cheek-mask widths (protects the cheek area from being
        re-generated). Match app.py defaults of 90/90.
    """

    def __init__(
        self,
        portrait_path: str,
        face_box: Sequence[int],
        *,
        device: str = "cuda",
        model_dir: str | Path,
        use_float16: bool = True,
        extra_margin: int = 10,
        parsing_mode: str = "jaw",
        left_cheek_width: int = 90,
        right_cheek_width: int = 90,
        use_gfpgan: bool = False,
        gfpgan_weight: float = 0.5,
    ) -> None:
        import cv2
        import torch

        self.device = device
        self.use_float16 = use_float16
        self.weight_dtype = torch.float16 if use_float16 else torch.float32
        self.model_dir = Path(model_dir)
        self.extra_margin = int(extra_margin)
        self.parsing_mode = parsing_mode

        # Make MuseTalk's `musetalk.*` package importable.
        vendor_str = str(_VENDOR_DIR)
        if vendor_str not in sys.path:
            sys.path.insert(0, vendor_str)

        # Load portrait (RGB). Keep full resolution.
        img_bgr = cv2.imread(portrait_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Portrait not found: {portrait_path}")
        self.portrait_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.portrait_h, self.portrait_w = self.portrait_rgb.shape[:2]

        # Clamp bbox + apply extra margin (to include chin), bounded by image.
        x1, y1, x2, y2 = (int(v) for v in face_box)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.portrait_w, x2)
        y2 = min(self.portrait_h, y2 + self.extra_margin)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid face_box after clamp: {(x1,y1,x2,y2)}")
        self.bbox = (x1, y1, x2, y2)

        logger.info(
            "CompositeRenderer: portrait=%dx%d bbox=%s (with extra_margin=%d) parsing_mode=%s",
            self.portrait_w, self.portrait_h, self.bbox, self.extra_margin, parsing_mode,
        )

        self._load_models()
        self._init_face_parser(
            left_cheek_width=left_cheek_width,
            right_cheek_width=right_cheek_width,
        )
        self._precompute_portrait_latents()

        # Optional GFPGAN sharpener. Lazy-built so import errors surface early
        # only when the caller actually asks for it.
        self.sharpener = None
        if use_gfpgan:
            from .gfpgan_pass import GFPGANSharpener  # noqa: PLC0415
            self.sharpener = GFPGANSharpener(
                model_dir=self.model_dir,
                device=device,
                weight=gfpgan_weight,
            )
            logger.info("Composite renderer: GFPGAN post-pass ENABLED (weight=%.2f)", gfpgan_weight)
        else:
            logger.info("Composite renderer: GFPGAN post-pass disabled.")

    # ------------------------------------------------------------------ models
    def _load_models(self) -> None:
        import torch  # noqa: PLC0415
        from musetalk.models.unet import UNet  # noqa: PLC0415
        from musetalk.utils.utils import load_all_model  # noqa: PLC0415
        from transformers import WhisperModel  # noqa: PLC0415

        unet_path = str(self.model_dir / "musetalkV15" / "unet.pth")
        unet_config = str(self.model_dir / "musetalkV15" / "musetalk.json")

        logger.info("Loading MuseTalk models from %s …", self.model_dir)
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=unet_path,
            vae_type="sd-vae",
            unet_config=unet_config,
            device=self.device,
        )

        whisper_path = str(self.model_dir / "whisper")
        logger.info("Loading Whisper encoder from %s …", whisper_path)
        self.whisper = WhisperModel.from_pretrained(whisper_path)
        self.whisper.to(self.device).to(self.weight_dtype).eval()

        # Cast sub-models to device+dtype
        self.vae.vae.to(self.device, dtype=self.weight_dtype)
        self.unet.model.to(self.device, dtype=self.weight_dtype)
        self.pe.to(self.device, dtype=self.weight_dtype)

        logger.info("MuseTalk models on %s (fp16=%s)", self.device, self.use_float16)

    # ------------------------------------------------------------------ face parser
    def _init_face_parser(self, *, left_cheek_width: int, right_cheek_width: int) -> None:
        """
        Initialise BiSeNet face-parsing model.

        FaceParsing in MuseTalk hard-codes relative paths
        './models/face-parse-bisent/{resnet18-5c106cde.pth,79999_iter.pth}'.
        Our entrypoint.sh already symlinks /workspace/edlio-presence/models
        → {RENDERER_MODEL_DIR} and chdirs to /workspace/edlio-presence, so
        the default cwd resolves './models/...' correctly.

        If someone runs this module with a different cwd (e.g. local tests),
        we fall back to the workspace dir for the duration of construction.
        """
        from musetalk.utils.face_parsing import FaceParsing  # noqa: PLC0415

        workspace_root = Path(__file__).resolve().parents[1]  # /workspace/edlio-presence
        models_link = workspace_root / "models"

        # Make sure ./models resolves to the actual weights dir.
        if not models_link.exists():
            try:
                os.symlink(str(self.model_dir), str(models_link))
                logger.info("Created symlink %s → %s", models_link, self.model_dir)
            except OSError as e:
                logger.warning("Could not symlink %s: %s", models_link, e)

        old_cwd = os.getcwd()
        try:
            if Path(old_cwd) != workspace_root:
                os.chdir(str(workspace_root))
            logger.info("Initialising FaceParsing (cwd=%s) …", os.getcwd())
            self.face_parser = FaceParsing(
                left_cheek_width=left_cheek_width,
                right_cheek_width=right_cheek_width,
            )
            logger.info("FaceParsing ready.")
        finally:
            if Path(old_cwd) != workspace_root:
                os.chdir(old_cwd)

    # ------------------------------------------------------------------ latents
    def _precompute_portrait_latents(self) -> None:
        """
        Crop the face region (with chin margin), resize to 256×256, encode to
        VAE latents once. Reused for every rendered frame.
        """
        import cv2  # noqa: PLC0415
        import torch  # noqa: PLC0415

        x1, y1, x2, y2 = self.bbox
        face_crop_rgb = self.portrait_rgb[y1:y2, x1:x2]
        # MuseTalk trains on BGR — its vae.get_latents_for_unet expects the
        # native colour space the model was trained with. We pass RGB and let
        # it do its own normalisation; matches app.py.
        face_256 = cv2.resize(face_crop_rgb, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        self._face_crop_shape = (y2 - y1, x2 - x1)  # (h, w) for resize-back

        with torch.no_grad():
            latents = self.vae.get_latents_for_unet(face_256)
            self.face_latents = latents.to(dtype=self.weight_dtype)
        logger.debug("Face latents precomputed: shape %s", tuple(self.face_latents.shape))

    # ------------------------------------------------------------------ warm-up
    def warm_up(self) -> None:
        import torch  # noqa: PLC0415

        logger.info("Warming up Composite GPU kernels …")
        dummy = torch.zeros(1, 50, 384, device=self.device, dtype=self.weight_dtype)
        dummy_features = AudioFeatures(whisper_features=dummy.cpu().numpy())
        self.render_frame(dummy_features, frame_index=0)
        logger.info("Warm-up complete.")

    # ------------------------------------------------------------------ render
    def render_frame(self, audio_features: AudioFeatures, frame_index: int = 0) -> np.ndarray:
        """
        Render one full-portrait frame.

        Returns
        -------
        np.ndarray, shape (portrait_h, portrait_w, 3), dtype uint8, RGB
        """
        import cv2  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from musetalk.utils.blending import get_image  # noqa: PLC0415

        x1, y1, x2, y2 = self.bbox

        # --- MuseTalk inference on the 256×256 face patch -------------------
        whisper_slice = audio_features.whisper_features[frame_index : frame_index + 1]
        audio_tensor = torch.from_numpy(whisper_slice).to(self.device, dtype=self.weight_dtype)
        audio_prompt = self.pe(audio_tensor)
        timesteps = torch.zeros(1, device=self.device, dtype=torch.long)

        with torch.no_grad():
            pred_latents = self.unet.model(
                self.face_latents,
                timesteps,
                encoder_hidden_states=audio_prompt,
            ).sample
            recon = self.vae.decode_latents(pred_latents)

        # recon shape from vae.decode_latents in MuseTalk is (N, 256, 256, 3) float32 [0, 255]
        if hasattr(recon, "__len__") and len(recon) > 0 and recon.ndim == 4:
            res_256 = recon[0]
        else:
            res_256 = recon

        res_uint8 = np.clip(res_256, 0, 255).astype(np.uint8)

        # --- Optional GFPGAN sharpening on the 256×256 patch ---------------
        if self.sharpener is not None:
            res_uint8 = self.sharpener.enhance(res_uint8)

        # --- Resize generated face back to original bbox dims ----------------
        h_crop, w_crop = self._face_crop_shape
        res_orig = cv2.resize(res_uint8, (w_crop, h_crop), interpolation=cv2.INTER_LANCZOS4)

        # --- Composite back into the full portrait --------------------------
        # get_image() expects BGR-like input (see its internal [:,:,::-1]).
        # We have RGB everywhere, so match app.py: pass raw RGB — get_image
        # does its own BGR/RGB dance internally and returns RGB.
        combined = get_image(
            self.portrait_rgb,
            res_orig,
            [x1, y1, x2, y2],
            mode=self.parsing_mode,
            fp=self.face_parser,
        )
        # combined shape == self.portrait_rgb shape, RGB uint8.
        return combined
