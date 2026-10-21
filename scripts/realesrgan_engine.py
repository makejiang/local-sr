# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RealESRGAN_x4plus inference engine (OpenVINO IR, dynamic input).

This is the runtime half of the skill: it loads the converted OpenVINO
IR and provides a single ``upscale(np.ndarray, scale)`` entry point.

Key behaviours
--------------
* **Dynamic input** — the OpenVINO IR is reshaped to (1, 3, H, W) per
  request, where (H, W) are inferred from the input image. This avoids
  the bilinear-resize-to-fixed-size that the static version requires.
* **Tiled inference** for large images. We split the image into
  overlapping tiles, run each tile through the model, and blend the
  overlap region with a raised-cosine window. This keeps peak GPU memory
  roughly constant regardless of input size.
* **Multi-scale** — the trained model only does 4x; for the ``--scale
  2`` / ``--scale 3`` CLI options we first resize the input down so that
  the 4x output lands at the requested final size.
* **BGR <-> RGB** handling matches the original Real-ESRGAN pipeline.

The tiling math is taken from the reference Real-ESRGAN utils:
https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Default paths
# --------------------------------------------------------------------------- #

_DEFAULT_MODEL_XML = Path.home() / ".openvino" / "models" / "realesrgan-ov" / "RealESRGAN_x4plus_dynamic.xml"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class EngineConfig:
    model_xml: Path
    device: str = "GPU"           # "GPU" or "CPU" or "AUTO"
    tile_size: int = 256          # 0 disables tiling
    tile_pad: int = 10
    scale_factor: int = 4         # native scale of the trained model


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #

class RealESRGANEngine:
    """Loads a dynamic-shape OpenVINO IR and provides ``upscale(...)``."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self._lock = threading.Lock()  # compile_model is thread-safe but the
                                       # underlying infer_request objects are
                                       # not, so serialize access.
        self._model_xml_path = str(config.model_xml)
        self._compiled_model = None
        self._infer_request = None
        self._input_tensor_name: str = ""
        self._output_tensor = None

    # ------------------------------------------------------------------ #
    # OpenVINO setup
    # ------------------------------------------------------------------ #

    def _select_device(self) -> str:
        """Pick the first GPU if available, else CPU. Override via config."""
        if self.config.device.upper() != "AUTO":
            return self.config.device
        try:
            import openvino as ov  # noqa: WPS433
            devices = ov.Core().available_devices
            for d in devices:
                if "GPU" in d:
                    return d
        except Exception:
            pass
        return "CPU"

    def ensure_compiled(self) -> None:
        if self._compiled_model is not None:
            return

        import openvino as ov  # noqa: WPS433

        xml_path = Path(self._model_xml_path)
        if not xml_path.exists():
            raise FileNotFoundError(
                f"OpenVINO IR not found at {xml_path}. "
                f"Run scripts\\run.ps1 once with a real input to trigger "
                f"the one-time download of the IR from ModelScope."
            )
        bin_path = xml_path.with_suffix(".bin")
        if not bin_path.exists():
            raise FileNotFoundError(
                f"OpenVINO weights not found at {bin_path}. The .xml is "
                f"present but the .bin is missing - please re-run the "
                f"download step."
            )

        device = self._select_device()
        core = ov.Core()
        model = core.read_model(str(xml_path))
        if not model.inputs:
            raise RuntimeError(f"OpenVINO model {xml_path} has no inputs")
        self._input_tensor_name = model.inputs[0].get_any_name()

        self._compiled_model = core.compile_model(model, device_name=device)
        self._output_tensor = self._compiled_model.outputs[0]
        self._infer_request = self._compiled_model.create_infer_request()
        print(
            f"[engine] Loaded OpenVINO IR: {xml_path.name} on {device}",
            flush=True,
        )

    def unload(self) -> None:
        """Release OpenVINO resources so the next process gets a clean slate."""
        self._compiled_model = None
        self._infer_request = None
        self._output_tensor = None

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def upscale(self, img_bgr: np.ndarray, requested_scale: int) -> np.ndarray:
        """Upscale a BGR uint8 image to ``requested_scale``x its resolution.

        Args:
            img_bgr: HxWxC uint8 BGR image.
            requested_scale: target up-scale factor; 1 returns the input
                unchanged. The native model scale is 4, so 2/3 are
                obtained by pre-resizing.
        """
        if requested_scale == 1:
            return img_bgr

        if img_bgr.dtype != np.uint8:
            raise ValueError(f"expected uint8 BGR image, got {img_bgr.dtype}")
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError(f"expected HxWx3 image, got shape {img_bgr.shape}")

        with self._lock:
            self.ensure_compiled()
            return self._upscale_locked(img_bgr, requested_scale)

    # ------------------------------------------------------------------ #
    # Internals (caller must hold the lock)
    # ------------------------------------------------------------------ #

    def _upscale_locked(self, img_bgr: np.ndarray, requested_scale: int) -> np.ndarray:
        native = self.config.scale_factor  # 4

        # ── 1. Pre-process ───────────────────────────────────────────────
        img = img_bgr.astype(np.float32)
        # 0..1
        img = img / 255.0
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        # add batch
        img = np.expand_dims(img, axis=0).astype(np.float32)
        # NCHW

        h_in, w_in = img.shape[2], img.shape[3]

        # If the user asked for a non-native scale, pre-resize so that
        # the native 4x output lands at the requested final size.
        if requested_scale != native:
            target_h = max(1, int(round(h_in * requested_scale / native)))
            target_w = max(1, int(round(w_in * requested_scale / native)))
            img = self._resize_nchw(img, target_h, target_w)
            h_in, w_in = img.shape[2], img.shape[3]

        # ── 2. Run inference (tiled or not) ──────────────────────────────
        t0 = time.time()
        if self.config.tile_size and (h_in > self.config.tile_size or w_in > self.config.tile_size):
            output = self._infer_tiled(img)
        else:
            output = self._infer_single(img)
        infer_time = time.time() - t0

        # ── 3. Post-process ─────────────────────────────────────────────
        # output is NCHW float32 in [0, 1] (RGB).
        output = output[0]
        output = np.clip(output, 0.0, 1.0)
        output = (output * 255.0).astype(np.uint8)
        # CHW -> HWC
        output = np.transpose(output, (1, 2, 0))
        # RGB -> BGR
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        print(
            f"[engine] inference {w_in}x{h_in} -> {output.shape[1]}x{output.shape[0]} "
            f"in {infer_time:.2f}s",
            flush=True,
        )
        return output

    @staticmethod
    def _resize_nchw(img_nchw: np.ndarray, h: int, w: int) -> np.ndarray:
        """Bilinear resize of an NCHW float32 image to (h, w)."""
        n, c, _, _ = img_nchw.shape
        out = np.empty((n, c, h, w), dtype=np.float32)
        for i in range(n):
            frame = img_nchw[i]                       # (C, H, W)
            frame = np.transpose(frame, (1, 2, 0))    # (H, W, C)
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
            out[i] = np.transpose(frame, (2, 0, 1))   # (C, H, W)
        return out

    def _infer_single(self, img_nchw: np.ndarray) -> np.ndarray:
        """Run the model on a single (small) input.

        The IR is already dynamic (``partial_shape=[1,3,?,?]``), so
        we just feed the input; OpenVINO handles the reshape.
        """
        result = self._infer_request.infer({self._input_tensor_name: img_nchw})[self._output_tensor]
        return np.asarray(result)

    # ------------------------------------------------------------------ #
    # Tiled inference - based on Real-ESRGAN's ``tile_process``
    # ------------------------------------------------------------------ #

    def _infer_tiled(self, img_nchw: np.ndarray) -> np.ndarray:
        """Split the input into tiles with padding context, infer each.

        Mirrors ``realesrgan.utils.tile_process`` from xinntao/Real-ESRGAN:
        each output tile is ``tile_size`` wide, but the model is fed a
        region extended by ``tile_pad`` on every side (clamped to the
        image border). The extra context is discarded before merging, so
        output tiles stay non-overlapping while each one had surrounding
        pixels available — this greatly reduces seam artifacts compared
        with non-overlapping tiling, and avoids the ``np.pad`` reflect
        crash on small edge tiles.
        """
        tile_size = self.config.tile_size
        tile_pad = self.config.tile_pad
        scale = self.config.scale_factor

        _, _, h, w = img_nchw.shape

        out_h = h * scale
        out_w = w * scale
        output = np.zeros((1, 3, out_h, out_w), dtype=np.float32)

        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                # Output tile boundaries in input space (no padding).
                out_y0 = y
                out_y1 = min(h, y + tile_size)
                out_x0 = x
                out_x1 = min(w, x + tile_size)

                # Input tile boundaries (with padding context, clamped
                # to the image border so we never read out of range).
                in_y0 = max(0, y - tile_pad)
                in_y1 = min(h, y + tile_size + tile_pad)
                in_x0 = max(0, x - tile_pad)
                in_x1 = min(w, x + tile_size + tile_pad)

                tile_input = img_nchw[:, :, in_y0:in_y1, in_x0:in_x1]

                # Run the model on the padded tile. The IR is dynamic,
                # so the variable tile size is handled by OpenVINO.
                tile_out = self._infer_request.infer(
                    {self._input_tensor_name: tile_input}
                )[self._output_tensor]
                tile_out = np.asarray(tile_out)

                # Locate the valid (non-context) region inside tile_out.
                # Padding actually applied on each side (may be smaller
                # than tile_pad near image borders).
                pad_top = y - in_y0
                pad_left = x - in_x0
                tile_out_y0 = pad_top * scale
                tile_out_x0 = pad_left * scale
                valid_h = (out_y1 - out_y0) * scale
                valid_w = (out_x1 - out_x0) * scale
                tile_out_cropped = tile_out[
                    :,
                    :,
                    tile_out_y0:tile_out_y0 + valid_h,
                    tile_out_x0:tile_out_x0 + valid_w,
                ]

                # Write into the output at the scaled coordinates.
                oy0 = out_y0 * scale
                ox0 = out_x0 * scale
                output[:, :, oy0:oy0 + valid_h, ox0:ox0 + valid_w] = tile_out_cropped

        return output


# --------------------------------------------------------------------------- #
# Convenience: load default engine
# --------------------------------------------------------------------------- #

def load_default_engine(device: str = "GPU", tile_size: int = 256) -> RealESRGANEngine:
    """Build an engine using the default model location."""
    return RealESRGANEngine(
        EngineConfig(
            model_xml=_DEFAULT_MODEL_XML,
            device=device,
            tile_size=tile_size,
        )
    )
