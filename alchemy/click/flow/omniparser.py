"""OmniParser v2 — fast UI element detection via YOLO + OCR + Florence-2.

Perception layer that sits BEFORE the VLM in the vision pipeline.
Produces a structured element map from a screenshot in ~200-500ms,
enabling:

1. Fast path: if goal matches a detected element, skip VLM entirely
2. Context enrichment: inject element map into VLM prompt for better reasoning
3. Verification: cross-check VLM coordinates against detected elements

v2 pipeline: YOLO icon detect → EasyOCR text → IoU overlap removal →
Florence-2 icon captioning (textless icons get descriptive labels).

Uses Microsoft OmniParser v2 (MIT license).
Model weights loaded from HuggingFace on first use:
  - icon_detect/model.pt  (~50MB YOLO)
  - icon_caption/  (~1GB Florence-2)
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UIElement:
    """A detected UI element with bounding box and label."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in screen pixels
    label: str  # OCR text or Florence-2 icon caption
    element_type: str  # "icon", "text", "button", "input", etc.
    confidence: float  # Detection confidence 0.0-1.0
    center_x: int = 0  # Click target X
    center_y: int = 0  # Click target Y

    def __post_init__(self) -> None:
        if self.center_x == 0 and self.center_y == 0:
            cx = (self.bbox[0] + self.bbox[2]) // 2
            cy = (self.bbox[1] + self.bbox[3]) // 2
            object.__setattr__(self, "center_x", cx)
            object.__setattr__(self, "center_y", cy)


@dataclass
class ParseResult:
    """Result of OmniParser screen analysis."""

    elements: list[UIElement]
    parse_ms: float  # How long detection took
    screen_width: int
    screen_height: int


class OmniParserBackend(Protocol):
    """Backend that performs actual detection. Swappable."""

    async def detect(self, image_bytes: bytes) -> list[dict]: ...


class OmniParser:
    """Fast perception layer: screenshot -> structured UI element map.

    v2 pipeline:
      1. YOLO icon detection (~100ms GPU)
      2. EasyOCR text detection (~200ms GPU)
      3. IoU overlap removal (icons that overlap text get merged)
      4. Florence-2 icon captioning (textless icons get semantic labels)

    Designed to run BEFORE the VLM in the AlchemyFlow pipeline.
    """

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.3,
        ocr_enabled: bool = True,
        caption_enabled: bool = True,
        caption_batch_size: int = 64,
        iou_threshold: float = 0.7,
        screen_width: int = 1920,
        screen_height: int = 1080,
        match_threshold: float = 0.5,
        device: str = "cuda:0",
    ) -> None:
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._ocr_enabled = ocr_enabled
        self._caption_enabled = caption_enabled
        self._caption_batch_size = caption_batch_size
        self._iou_threshold = iou_threshold
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._match_threshold = match_threshold
        self._device = device
        self._loaded = False

        # Instance-level model storage (no globals)
        self._yolo: Any = None
        self._ocr: Any = None
        self._caption_model: Any = None
        self._caption_processor: Any = None

    async def load(self) -> None:
        """Load YOLO + OCR + Florence-2 models. Called once on first use."""
        if self._loaded:
            return

        # --- YOLO icon detection ---
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "OmniParser requires ultralytics: pip install ultralytics"
            )

        model_path = self._model_path
        if model_path is None:
            model_path = _ensure_models_downloaded()

        logger.info("OmniParser: loading YOLO from %s", model_path)
        start = time.monotonic()
        self._yolo = YOLO(str(model_path))
        self._yolo.to(self._device)
        logger.info(
            "OmniParser: YOLO loaded in %.0fms on %s",
            (time.monotonic() - start) * 1000,
            self._device,
        )

        # --- EasyOCR ---
        if self._ocr_enabled:
            try:
                import easyocr

                start = time.monotonic()
                self._ocr = easyocr.Reader(
                    ["en"], gpu=("cuda" in self._device)
                )
                logger.info(
                    "OmniParser: EasyOCR loaded in %.0fms",
                    (time.monotonic() - start) * 1000,
                )
            except ImportError:
                logger.warning(
                    "OmniParser: easyocr not installed, OCR disabled. "
                    "pip install easyocr"
                )

        # --- Florence-2 icon captioning ---
        if self._caption_enabled:
            try:
                self._load_florence2()
            except Exception as exc:
                logger.warning(
                    "OmniParser: Florence-2 captioning disabled: %s", exc
                )

        self._loaded = True

    def _load_florence2(self) -> None:
        """Load Florence-2 model for icon captioning."""
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        caption_dir = (
            Path.home()
            / ".cache"
            / "alchemy"
            / "omniparser"
            / "icon_caption"
        )
        if not caption_dir.exists():
            _ensure_models_downloaded()

        logger.info("OmniParser: loading Florence-2 from %s", caption_dir)
        start = time.monotonic()

        self._caption_model = AutoModelForCausalLM.from_pretrained(
            str(caption_dir),
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(self._device)
        self._caption_model.eval()

        # Processor from base model (local dir has fine-tuned weights only)
        self._caption_processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base-ft", trust_remote_code=True
        )

        logger.info(
            "OmniParser: Florence-2 loaded in %.0fms on %s",
            (time.monotonic() - start) * 1000,
            self._device,
        )

    # ------------------------------------------------------------------
    # Main parse pipeline
    # ------------------------------------------------------------------

    async def parse(self, screenshot: bytes) -> ParseResult:
        """Detect all UI elements in a screenshot.

        v2 pipeline:
          1. YOLO icon detection
          2. EasyOCR text detection
          3. IoU overlap removal (icons overlapping text -> text wins)
          4. Florence-2 captioning for remaining textless icons
        """
        if not self._loaded:
            await self.load()

        from PIL import Image

        img = Image.open(io.BytesIO(screenshot))
        img_w, img_h = img.size
        sx = self._screen_width / img_w
        sy = self._screen_height / img_h

        start = time.monotonic()

        # Stage 1: YOLO icon detection
        icon_boxes, icon_confs = self._run_yolo(img, sx, sy)

        # Stage 2: OCR text detection
        text_elements: list[UIElement] = []
        text_boxes: list[tuple[int, int, int, int]] = []
        if self._ocr_enabled and self._ocr is not None:
            text_elements, text_boxes = self._run_ocr(img, sx, sy)

        # Stage 3: IoU overlap removal — icons that overlap text get removed
        surviving_icon_indices = self._remove_overlap(
            icon_boxes, text_boxes
        )

        # Stage 4: Florence-2 captioning for surviving textless icons
        icon_elements = self._caption_icons(
            img, icon_boxes, icon_confs, surviving_icon_indices
        )

        elements = text_elements + icon_elements
        parse_ms = (time.monotonic() - start) * 1000

        logger.debug(
            "OmniParser v2: %d elements (%d text + %d icons) in %.0fms",
            len(elements),
            len(text_elements),
            len(icon_elements),
            parse_ms,
        )

        return ParseResult(
            elements=elements,
            parse_ms=parse_ms,
            screen_width=self._screen_width,
            screen_height=self._screen_height,
        )

    # ------------------------------------------------------------------
    # Stage 1: YOLO
    # ------------------------------------------------------------------

    def _run_yolo(
        self, img: Any, sx: float, sy: float
    ) -> tuple[list[tuple[int, int, int, int]], list[float]]:
        """Run YOLO icon detection. Returns (boxes, confidences)."""
        if self._yolo is None:
            return [], []

        results = self._yolo.predict(
            img, conf=self._confidence_threshold, verbose=False
        )
        if not results or len(results) == 0:
            return [], []

        boxes: list[tuple[int, int, int, int]] = []
        confs: list[float] = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append((
                int(x1 * sx),
                int(y1 * sy),
                int(x2 * sx),
                int(y2 * sy),
            ))
            confs.append(float(box.conf[0]))

        return boxes, confs

    # ------------------------------------------------------------------
    # Stage 2: OCR
    # ------------------------------------------------------------------

    def _run_ocr(
        self, img: Any, sx: float, sy: float
    ) -> tuple[list[UIElement], list[tuple[int, int, int, int]]]:
        """Run EasyOCR text detection. Returns (elements, boxes)."""
        if self._ocr is None:
            return [], []

        import numpy as np

        img_array = np.array(img)
        results = self._ocr.readtext(img_array)

        elements: list[UIElement] = []
        boxes: list[tuple[int, int, int, int]] = []

        for bbox, text, conf in results:
            if conf < self._confidence_threshold or not text.strip():
                continue

            x1 = int(min(p[0] for p in bbox) * sx)
            y1 = int(min(p[1] for p in bbox) * sy)
            x2 = int(max(p[0] for p in bbox) * sx)
            y2 = int(max(p[1] for p in bbox) * sy)

            box = (x1, y1, x2, y2)
            boxes.append(box)
            elements.append(
                UIElement(
                    bbox=box,
                    label=text.strip(),
                    element_type="text",
                    confidence=conf,
                )
            )

        return elements, boxes

    # ------------------------------------------------------------------
    # Stage 3: IoU overlap removal
    # ------------------------------------------------------------------

    def _remove_overlap(
        self,
        icon_boxes: list[tuple[int, int, int, int]],
        text_boxes: list[tuple[int, int, int, int]],
    ) -> list[int]:
        """Remove icon boxes that overlap significantly with text boxes.

        When an icon and text overlap (IoU > threshold), text wins because
        it already has a label from OCR. Returns indices of surviving icons.

        Ported from Microsoft OmniParser v2 handler.py.
        """
        if not icon_boxes or not text_boxes:
            return list(range(len(icon_boxes)))

        surviving: list[int] = []

        for i, ibox in enumerate(icon_boxes):
            overlaps = False
            for tbox in text_boxes:
                iou = _compute_iou(ibox, tbox)
                if iou > self._iou_threshold:
                    overlaps = True
                    break
            if not overlaps:
                surviving.append(i)

        removed = len(icon_boxes) - len(surviving)
        if removed > 0:
            logger.debug(
                "OmniParser: removed %d icon boxes overlapping with text",
                removed,
            )

        return surviving

    # ------------------------------------------------------------------
    # Stage 4: Florence-2 icon captioning
    # ------------------------------------------------------------------

    def _caption_icons(
        self,
        img: Any,
        icon_boxes: list[tuple[int, int, int, int]],
        icon_confs: list[float],
        surviving_indices: list[int],
    ) -> list[UIElement]:
        """Caption surviving icons using Florence-2.

        Crops each icon to 64x64, runs Florence-2 with <CAPTION> prompt
        in batches. Falls back to "icon" label if captioning is disabled
        or model not loaded.
        """
        if not surviving_indices:
            return []

        from PIL import Image

        has_captioner = (
            self._caption_model is not None
            and self._caption_processor is not None
        )

        if not has_captioner:
            # Fallback: generic "icon" label
            return [
                UIElement(
                    bbox=icon_boxes[i],
                    label="icon",
                    element_type="icon",
                    confidence=icon_confs[i],
                )
                for i in surviving_indices
            ]

        import torch

        # Reverse screen scaling to get image-space coords for cropping
        img_w, img_h = img.size
        inv_sx = img_w / self._screen_width
        inv_sy = img_h / self._screen_height

        crops: list[Any] = []
        valid_indices: list[int] = []

        for i in surviving_indices:
            sx1, sy1, sx2, sy2 = icon_boxes[i]
            ix1 = max(0, int(sx1 * inv_sx))
            iy1 = max(0, int(sy1 * inv_sy))
            ix2 = min(img_w, int(sx2 * inv_sx))
            iy2 = min(img_h, int(sy2 * inv_sy))

            if ix2 <= ix1 or iy2 <= iy1:
                continue

            crop = img.crop((ix1, iy1, ix2, iy2)).resize(
                (64, 64), Image.LANCZOS
            )
            crops.append(crop)
            valid_indices.append(i)

        if not crops:
            return []

        # Batch inference through Florence-2
        captions = self._run_florence2_batch(crops)

        elements: list[UIElement] = []
        for j, i in enumerate(valid_indices):
            caption = captions[j] if j < len(captions) else "icon"
            elements.append(
                UIElement(
                    bbox=icon_boxes[i],
                    label=caption,
                    element_type=_classify_element(caption),
                    confidence=icon_confs[i],
                )
            )

        return elements

    def _run_florence2_batch(self, crops: list[Any]) -> list[str]:
        """Run Florence-2 captioning on a batch of icon crops.

        Each crop is 64x64 PIL Image. Returns one caption string per crop.
        Batched to limit VRAM usage (~2GB for 64 crops).
        """
        import torch

        captions: list[str] = []
        batch_size = self._caption_batch_size

        for start in range(0, len(crops), batch_size):
            batch = crops[start : start + batch_size]

            inputs = self._caption_processor(
                text=["<CAPTION>"] * len(batch),
                images=batch,
                return_tensors="pt",
                padding=True,
            ).to(self._device)
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
            # Processor's attention_mask covers only text tokens (8), but
            # Florence-2 fuses image features into the sequence (585 tokens).
            # Wrong-shaped mask → crash. Removing it lets the model build
            # its own correct mask internally.
            inputs.pop("attention_mask", None)

            with torch.no_grad():
                generated = self._caption_model.generate(
                    **inputs, max_new_tokens=20, num_beams=1
                )

            decoded = self._caption_processor.batch_decode(
                generated, skip_special_tokens=True
            )

            for text in decoded:
                caption = text.strip()
                if caption.startswith("<CAPTION>"):
                    caption = caption[len("<CAPTION>") :]
                caption = caption.strip().lower()
                if not caption:
                    caption = "icon"
                captions.append(caption)

        return captions

    # ------------------------------------------------------------------
    # Goal matching + verification
    # ------------------------------------------------------------------

    def match_goal(
        self,
        elements: list[UIElement],
        goal: str,
    ) -> UIElement | None:
        """Find the best element matching a natural language goal.

        Uses fuzzy text matching between the goal and element labels.
        Returns the highest-confidence match above threshold, or None.
        """
        goal_lower = goal.lower()
        best: UIElement | None = None
        best_score = 0.0

        keywords = _extract_keywords(goal_lower)

        for elem in elements:
            label_lower = elem.label.lower()

            # Direct substring match (highest priority)
            if label_lower in goal_lower or goal_lower in label_lower:
                score = 0.9 * elem.confidence
            else:
                # Fuzzy match
                seq_score = SequenceMatcher(
                    None, label_lower, goal_lower
                ).ratio()
                # Keyword overlap
                kw_score = _keyword_overlap(keywords, label_lower)
                score = max(seq_score, kw_score) * elem.confidence

            if score > best_score and score >= self._match_threshold:
                best_score = score
                best = elem

        if best:
            logger.debug(
                "OmniParser match: %r (score=%.2f, conf=%.2f) at (%d, %d)",
                best.label,
                best_score,
                best.confidence,
                best.center_x,
                best.center_y,
            )

        return best

    def to_prompt_context(self, elements: list[UIElement]) -> str:
        """Format element map as context for VLM prompt enrichment."""
        if not elements:
            return ""

        lines = ["## Detected UI Elements (OmniParser v2)"]
        sorted_elems = sorted(
            elements, key=lambda e: (e.bbox[1], e.bbox[0])
        )

        for i, elem in enumerate(sorted_elems, 1):
            cx, cy = elem.center_x, elem.center_y
            lines.append(
                f'  {i}. [{elem.element_type}] "{elem.label}" '
                f"at ({cx},{cy}) conf={elem.confidence:.2f}"
            )

        return "\n".join(lines)

    def verify_action(
        self,
        elements: list[UIElement],
        x: int,
        y: int,
        tolerance: int = 50,
    ) -> UIElement | None:
        """Check if VLM-proposed coordinates land on a detected element."""
        for elem in elements:
            ex1, ey1, ex2, ey2 = elem.bbox
            if (
                ex1 - tolerance <= x <= ex2 + tolerance
                and ey1 - tolerance <= y <= ey2 + tolerance
            ):
                return elem
        return None

    @property
    def loaded(self) -> bool:
        return self._loaded


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ELEMENT_TYPE_MAP = {
    "button": "button",
    "icon": "icon",
    "checkbox": "input",
    "radio": "input",
    "textbox": "input",
    "input": "input",
    "link": "link",
    "image": "icon",
    "text": "text",
}

_CAPTION_TYPE_HINTS = {
    "button": "button",
    "settings": "icon",
    "gear": "icon",
    "menu": "icon",
    "close": "button",
    "minimize": "button",
    "maximize": "button",
    "search": "input",
    "arrow": "icon",
    "checkbox": "input",
    "toggle": "input",
}


def _classify_element(label: str) -> str:
    """Map YOLO class name or Florence-2 caption to a UI element type."""
    lower = label.lower()
    if lower in _ELEMENT_TYPE_MAP:
        return _ELEMENT_TYPE_MAP[lower]
    for hint, etype in _CAPTION_TYPE_HINTS.items():
        if hint in lower:
            return etype
    return "icon"


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from a goal string."""
    stop_words = {
        "the", "a", "an", "on", "in", "at", "to", "of", "for",
        "click", "press", "tap", "hit", "find", "open", "close",
        "button", "link", "icon", "and", "or", "is", "it",
    }
    words = set(text.split())
    return words - stop_words


def _keyword_overlap(keywords: set[str], label: str) -> float:
    """Calculate what fraction of keywords appear in the label."""
    if not keywords:
        return 0.0
    label_words = set(label.split())
    overlap = keywords & label_words
    return len(overlap) / len(keywords)


def _compute_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    """Compute intersection-over-union between two bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def _ensure_models_downloaded() -> Path:
    """Download OmniParser v2 weights from HuggingFace if not cached.

    Downloads both icon_detect (YOLO) and icon_caption (Florence-2).
    Returns the local path to the YOLO model weights file.
    """
    cache_dir = Path.home() / ".cache" / "alchemy" / "omniparser"
    model_file = cache_dir / "icon_detect" / "model.pt"
    caption_dir = cache_dir / "icon_caption"

    needs_yolo = not model_file.exists()
    needs_caption = not caption_dir.exists()

    if not needs_yolo and not needs_caption:
        return model_file

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "OmniParser model download requires huggingface_hub: "
            "pip install huggingface_hub"
        )

    patterns = []
    if needs_yolo:
        patterns.append("icon_detect/*")
        logger.info("OmniParser: downloading YOLO weights...")
    if needs_caption:
        patterns.append("icon_caption/*")
        logger.info("OmniParser: downloading Florence-2 weights (~1GB)...")

    snapshot_download(
        "microsoft/OmniParser-v2.0",
        local_dir=str(cache_dir),
        allow_patterns=patterns,
    )

    if not model_file.exists():
        raise FileNotFoundError(
            f"YOLO weights not found at {model_file} after download. "
            f"Check https://huggingface.co/microsoft/OmniParser-v2.0"
        )

    return model_file
