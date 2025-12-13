"""Configuration constants for medical certificate OCR processing."""

from pathlib import Path

# Normalised portrait size used after perspective correction
TARGET_WIDTH = 1400
TARGET_HEIGHT = 1980

# ROI map file path
ROI_MAP_PATH = Path(__file__).parent / "roi_map.json"

