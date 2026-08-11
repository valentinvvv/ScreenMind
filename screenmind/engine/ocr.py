"""
OCR Engine — Text Extraction from Screenshots
Extracts visible text to enhance Gemma 4's analysis context.
Uses easyocr with dark-theme preprocessing for robust recognition.
"""

import logging
import time
import warnings
from typing import Optional

from PIL import Image, ImageOps, ImageEnhance

logger = logging.getLogger("screenmind.engine.ocr")


class OCRExtractor:
    """
    Lightweight OCR that extracts screen text to feed into Gemma 4
    as additional context, improving analysis accuracy.
    """

    def __init__(self):
        self._reader = None
        self._available = True

    def _ensure_reader(self):
        """Lazy-load easyocr (downloads ~100MB model on first use)."""
        if self._reader is None and self._available:
            try:
                import easyocr
                self._reader = easyocr.Reader(
                    ["en"],
                    gpu=True,       # Use GPU if available, falls back to CPU
                    verbose=False,
                )
                logger.info("EasyOCR initialized")
            except Exception as e:
                logger.warning(f"EasyOCR unavailable, skipping text extraction: {e}")
                self._available = False

    def _preprocess(self, image: Image.Image) -> Image.Image:
        """
        Preprocess screenshot for maximum OCR accuracy.
        
        Optimized via A/B testing (5 rounds, 7 strategies):
        - Ensure minimum 1920px width (upscale if needed)
        - Grayscale + sharpen + contrast 1.5
        - Only invert for very dark screens (<100 brightness)
        - Result: eCAS detection 0.04 → 0.54 confidence
        """
        import numpy as np
        from PIL import ImageFilter

        # Ensure minimum 1920px width for reliable text detection
        min_width = 1920
        if image.size[0] < min_width:
            scale = min_width / image.size[0]
            image = image.resize(
                (int(image.size[0] * scale), int(image.size[1] * scale)),
                Image.Resampling.LANCZOS,
            )

        # Convert to grayscale
        gray = image.convert("L")

        # Check brightness for dark theme handling
        avg_brightness = np.mean(np.array(gray))

        if avg_brightness < 100:
            # Very dark theme — invert first
            gray = ImageOps.invert(gray)

        # Sharpen text edges
        gray = gray.filter(ImageFilter.SHARPEN)

        # Mild contrast boost (1.5 is the sweet spot — 2.2 destroyed text)
        result = ImageEnhance.Contrast(gray).enhance(1.5)

        return result

    def extract_text(self, image: Image.Image) -> Optional[str]:
        """Extract text only (backward compatible)."""
        text, _ = self.extract_text_with_boxes(image)
        return text

    def extract_text_with_boxes(self, image: Image.Image):
        """
        Extract text AND bounding boxes from a screenshot.

        Returns:
            Tuple of (text_string, boxes_list) where boxes_list is a list of
            {"box": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], "text": str, "conf": float}
            Coordinates are relative to the ORIGINAL image size.
        """
        if not self._available:
            return None, []

        self._ensure_reader()
        if self._reader is None:
            return None, []

        try:
            import numpy as np
            start = time.time()

            # Track scale for coordinate mapping back to original
            orig_w, orig_h = image.size

            # Preprocess: resize, dark-theme inversion, contrast boost
            processed = self._preprocess(image)
            proc_w, proc_h = processed.size

            # Scale factors to map preprocessed coords back to original image
            scale_x = orig_w / proc_w
            scale_y = orig_h / proc_h

            img_array = np.array(processed)

            # Run OCR with word-level detail (paragraph=False for per-word boxes)
            # easyocr hardcodes pin_memory=True in its DataLoader; on machines
            # without a CUDA accelerator torch emits a UserWarning on every call.
            # Harmless — silence it so it doesn't spam the capture log.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*'pin_memory' argument is set as true but no accelerator is found.*",
                    category=UserWarning,
                )
                results = self._reader.readtext(
                    img_array, detail=1, paragraph=False,
                    batch_size=4,
                )

            texts = []
            boxes = []
            for result in results:
                if len(result) == 3:
                    bbox, text, conf = result
                elif len(result) == 2:
                    bbox, text = result
                    conf = 0.5
                else:
                    continue

                text = text.strip()
                if conf > 0.2 and len(text) > 1:
                    texts.append(text)
                    # Scale bbox coords back to original image coordinates
                    scaled_box = [[int(pt[0] * scale_x), int(pt[1] * scale_y)] for pt in bbox]
                    boxes.append({"box": scaled_box, "text": text, "conf": round(conf, 2)})

            elapsed = time.time() - start
            full_text = "\n".join(texts)

            if texts:
                logger.debug(f"Extracted {len(texts)} text blocks in {elapsed:.1f}s")

            return (full_text if full_text else None), boxes

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return None, []

    @property
    def is_available(self) -> bool:
        return self._available
