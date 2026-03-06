"""OmniParser — fast UI element detection via YOLO + OCR.

Perception layer that sits BEFORE the VLM in the vision pipeline.
Produces a structured element map from a screenshot in ~200-500ms,
enabling:

1. Fast path: if goal matches a detected element, skip VLM entirely
2. Context enrichment: inject element map into VLM prompt for better reasoning
3. Verification: cross-check VLM coordinates against detected elements

Uses Microsoft OmniParser (MIT license) — YOLO icon detector + OCR.
Model weights loaded from HuggingFace on first use (~50MB YOLO + OCR).
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

# Lazy-loaded heavy imports (ultralytics, PIL, easyocr)
_yolo_model: Any = None
_ocr_reader: Any = None


@dataclass(frozen=True)
class UIElement:
    """A detected UI element with bounding box and label."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in screen pixels
    label: str  # OCR text or icon caption
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
    """Fast perception layer: screenshot → structured UI element map.

    Designed to run BEFORE the VLM in the AlchemyFlow pipeline.
    The element map can be used for:
    - Direct element matching (fast path, <500ms)
    - VLM prompt enrichment (inject detected elements as context)
    - Post-VLM verification (check VLM coords against element bboxes)
    """

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.3,
        ocr_enabled: bool = True,
        screen_width: int = 1920,
        screen_height: int = 1080,
        match_threshold: float = 0.5,
        device: str = "cuda:0",
    ) -> None:
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._ocr_enabled = ocr_enabled
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._match_threshold = match_threshold
        self._device = device
        self._loaded = False

    async def load(self) -> None:
        """Load YOLO detection model + OCR. Called once on first use."""
        if self._loaded:
            return

        global _yolo_model, _ocr_reader

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "OmniParser requires ultralytics: pip install ultralytics"
            )

        model_path = self._model_path
        if model_path is None:
            # Default: download from HuggingFace microsoft/OmniParser
            model_path = _ensure_model_downloaded()

        logger.info("OmniParser: loading YOLO model from %s", model_path)
        start = time.monotonic()
        _yolo_model = YOLO(str(model_path))
        _yolo_model.to(self._device)
        load_ms = (time.monotonic() - start) * 1000
        logger.info("OmniParser: YOLO loaded in %.0fms on %s", load_ms, self._device)

        if self._ocr_enabled:
            try:
                import easyocr

                _ocr_reader = easyocr.Reader(["en"], gpu=("cuda" in self._device))
                logger.info("OmniParser: EasyOCR loaded")
            except ImportError:
                logger.warning(
                    "OmniParser: easyocr not installed, OCR disabled. "
                    "pip install easyocr for text detection"
                )
                _ocr_reader = None

        self._loaded = True

    async def parse(self, screenshot: bytes) -> ParseResult:
        """Detect all UI elements in a screenshot.

        Returns a ParseResult with all detected elements and timing info.
        """
        if not self._loaded:
            await self.load()

        start = time.monotonic()
        elements: list[UIElement] = []

        # YOLO detection
        yolo_elements = await self._detect_icons(screenshot)
        elements.extend(yolo_elements)

        # OCR text detection
        if self._ocr_enabled and _ocr_reader is not None:
            ocr_elements = await self._detect_text(screenshot)
            elements.extend(ocr_elements)

        parse_ms = (time.monotonic() - start) * 1000
        logger.debug(
            "OmniParser: %d elements detected in %.0fms", len(elements), parse_ms
        )

        return ParseResult(
            elements=elements,
            parse_ms=parse_ms,
            screen_width=self._screen_width,
            screen_height=self._screen_height,
        )

    async def _detect_icons(self, screenshot: bytes) -> list[UIElement]:
        """Run YOLO icon/element detection."""
        global _yolo_model
        if _yolo_model is None:
            return []

        from PIL import Image

        img = Image.open(io.BytesIO(screenshot))
        img_w, img_h = img.size

        # Scale factors from detection image to screen coords
        sx = self._screen_width / img_w
        sy = self._screen_height / img_h

        results = _yolo_model.predict(
            img,
            conf=self._confidence_threshold,
            verbose=False,
        )

        elements: list[UIElement] = []
        if not results or len(results) == 0:
            return elements

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = results[0].names.get(cls_id, "element")

            # Scale to screen coordinates
            sx1 = int(x1 * sx)
            sy1 = int(y1 * sy)
            sx2 = int(x2 * sx)
            sy2 = int(y2 * sy)

            elements.append(
                UIElement(
                    bbox=(sx1, sy1, sx2, sy2),
                    label=cls_name,
                    element_type=_classify_element(cls_name),
                    confidence=conf,
                )
            )

        return elements

    async def _detect_text(self, screenshot: bytes) -> list[UIElement]:
        """Run OCR text detection."""
        global _ocr_reader
        if _ocr_reader is None:
            return []

        from PIL import Image
        import numpy as np

        img = Image.open(io.BytesIO(screenshot))
        img_w, img_h = img.size
        sx = self._screen_width / img_w
        sy = self._screen_height / img_h

        img_array = np.array(img)
        results = _ocr_reader.readtext(img_array)

        elements: list[UIElement] = []
        for bbox, text, conf in results:
            if conf < self._confidence_threshold:
                continue
            if not text.strip():
                continue

            # EasyOCR returns [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            x1 = int(min(p[0] for p in bbox) * sx)
            y1 = int(min(p[1] for p in bbox) * sy)
            x2 = int(max(p[0] for p in bbox) * sx)
            y2 = int(max(p[1] for p in bbox) * sy)

            elements.append(
                UIElement(
                    bbox=(x1, y1, x2, y2),
                    label=text.strip(),
                    element_type="text",
                    confidence=conf,
                )
            )

        return elements

    def match_goal(
        self, elements: list[UIElement], goal: str,
    ) -> UIElement | None:
        """Find the best element matching a natural language goal.

        Uses fuzzy text matching between the goal and element labels.
        Returns the highest-confidence match above threshold, or None.
        """
        goal_lower = goal.lower()
        best: UIElement | None = None
        best_score = 0.0

        # Extract key terms from goal (simple keyword extraction)
        keywords = _extract_keywords(goal_lower)

        for elem in elements:
            label_lower = elem.label.lower()

            # Direct substring match (highest priority)
            if label_lower in goal_lower or goal_lower in label_lower:
                score = 0.9 * elem.confidence
            else:
                # Fuzzy match
                seq_score = SequenceMatcher(None, label_lower, goal_lower).ratio()
                # Keyword overlap
                kw_score = _keyword_overlap(keywords, label_lower)
                score = max(seq_score, kw_score) * elem.confidence

            if score > best_score and score >= self._match_threshold:
                best_score = score
                best = elem

        if best:
            logger.debug(
                "OmniParser match: %r (score=%.2f, conf=%.2f) at (%d, %d)",
                best.label, best_score, best.confidence, best.center_x, best.center_y,
            )

        return best

    def to_prompt_context(self, elements: list[UIElement]) -> str:
        """Format element map as context for VLM prompt enrichment.

        Produces a compact text representation that can be injected into
        the system prompt, giving the VLM structured awareness of what's
        on screen before it even looks at the screenshot.
        """
        if not elements:
            return ""

        lines = ["## Detected UI Elements (OmniParser)"]
        # Sort by Y position (top to bottom), then X (left to right)
        sorted_elems = sorted(elements, key=lambda e: (e.bbox[1], e.bbox[0]))

        for i, elem in enumerate(sorted_elems, 1):
            cx, cy = elem.center_x, elem.center_y
            lines.append(
                f"  {i}. [{elem.element_type}] \"{elem.label}\" "
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
        """Check if VLM-proposed coordinates land on a detected element.

        Returns the matching element if found within tolerance pixels,
        or None if the VLM is clicking empty space.
        """
        for elem in elements:
            # Check if (x, y) is within the element bbox (with tolerance)
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


def _classify_element(cls_name: str) -> str:
    """Map YOLO class name to a UI element type."""
    return _ELEMENT_TYPE_MAP.get(cls_name.lower(), "element")


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


def _ensure_model_downloaded() -> Path:
    """Download OmniParser YOLO weights from HuggingFace if not cached.

    Returns the local path to the model weights file.
    """
    cache_dir = Path.home() / ".cache" / "alchemy" / "omniparser"
    model_file = cache_dir / "icon_detect" / "best.pt"

    if model_file.exists():
        return model_file

    logger.info("OmniParser: downloading model weights (first time only)...")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            "microsoft/OmniParser-v2.0",
            local_dir=str(cache_dir),
            allow_patterns=["icon_detect/*"],
        )
    except ImportError:
        raise ImportError(
            "OmniParser model download requires huggingface_hub: "
            "pip install huggingface_hub"
        )

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model weights not found at {model_file} after download. "
            f"Check https://huggingface.co/microsoft/OmniParser-v2.0"
        )

    return model_file
