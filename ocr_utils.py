from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import unicodedata
from typing import Any, Dict, List, Tuple

import cv2
import easyocr
import numpy as np
from PIL import Image

from config import ROI_MAP_PATH, TARGET_HEIGHT, TARGET_WIDTH


_reader_ref: Dict[str, easyocr.Reader] = {}

def _reader() -> easyocr.Reader:
    r = _reader_ref.get("r")
    if r is None:
        # GPU retained for parity. Change to gpu=False if needed.
        r = easyocr.Reader(["en"], gpu=True)
        _reader_ref["r"] = r
    return r

def _normalize_text(text: str) -> str:
    """Lowercase, strip accents, remove punctuation, collapse spaces."""
    if text is None:
        return ""
    text = str(text).lower()
    # strip Romanian diacritics
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    # keep only letters, digits and spaces
    text = re.sub(r"[^0-9a-z]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



def _build_ocr_index(ocr_raw: List[Tuple]) -> List[Dict[str, Any]]:
    """
    Turn EasyOCR output into a convenient list of dicts:
    {x1, y1, x2, y2, cx, cy, w, h, text, norm, conf}.
    """
    entries: List[Dict[str, Any]] = []
    for box, text, conf in ocr_raw:
        pts = np.array(box, dtype=float)  # shape (4, 2)
        xs = pts[:, 0]
        ys = pts[:, 1]
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        norm = _normalize_text(text)
        entries.append({
            "box": pts,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cx": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
            "w": (x2 - x1),
            "h": (y2 - y1),
            "text": text,
            "norm": norm,
            "norm_ns": norm.replace(" ", ""),  # no-space variant
            "conf": float(conf),
        })
    return entries


def _find_anchor(entries: List[Dict[str, Any]], pattern: str) -> Dict[str, Any] | None:
    """
    Find the OCR box that matches the given pattern.

    Important:
    - We only accept boxes where the *pattern* (with or without spaces)
      is contained in the OCR text.
    - We DO NOT accept boxes where the OCR text is just a substring of
      the pattern (this was causing random matches like 'anui' for 'anui 20').
    - If multiple matches, we pick the *topmost* one on the page.
    """
    target = _normalize_text(pattern)
    if not target:
        return None
    target_ns = target.replace(" ", "")

    candidates: List[Dict[str, Any]] = []

    for e in entries:
        norm = e.get("norm") or ""
        if not norm:
            continue
        norm_ns = e.get("norm_ns") or norm.replace(" ", "")

        # Only allow: pattern ⊆ OCR text (not the other way around)
        if target in norm or target_ns in norm_ns:
            candidates.append(e)

    if not candidates:
        return None

    # Choose the *topmost* candidate (smallest y1).
    # If two are on the same line, prefer the one with higher conf * length.
    def sort_key(e: Dict[str, Any]):
        norm = e.get("norm_ns") or e.get("norm") or ""
        score = -(e.get("conf", 0.0) * max(1, len(norm)))
        return (e["y1"], score)

    candidates.sort(key=sort_key)
    return candidates[0]





def _iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0: return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)

def _digitish(s: str) -> bool:
    return bool(re.fullmatch(r"[0-9]+", s))

def avg_conf(tokens):
    return (sum(t["conf"] for t in tokens) / len(tokens)) if tokens else 0.0

def _collect_digit_tokens(ocr_entries, x1, x2, y1, y2):
    # 1st pass: overlap ≥ 0.10 OR center-in-box
    B = (x1, y1, x2, y2)
    picked = []
    for t in ocr_entries:
        if not _digitish(t["text"].strip()):
            continue
        tx1, ty1, tx2, ty2 = t["x1"], t["y1"], t["x2"], t["y2"]
        cx, cy = (tx1 + tx2)/2.0, (ty1 + ty2)/2.0
        center_in = (x1 <= cx <= x2 and y1 <= cy <= y2)
        if center_in or _iou(B, (tx1, ty1, tx2, ty2)) >= 0.10:
            picked.append(t)

    # 2nd pass: if empty, gently expand by 20% and try again
    if not picked:
        mx, my = 0.20*(x2-x1), 0.20*(y2-y1)
        ex1, ex2 = max(0, x1 - mx), x2 + mx
        ey1, ey2 = max(0, y1 - my), y2 + my
        B2 = (ex1, ey1, ex2, ey2)
        for t in ocr_entries:
            if not _digitish(t["text"].strip()):
                continue
            tx1, ty1, tx2, ty2 = t["x1"], t["y1"], t["x2"], t["y2"]
            if _iou(B2, (tx1, ty1, tx2, ty2)) >= 0.10:
                picked.append(t)

    # sort left-to-right
    picked.sort(key=lambda t: (t["y1"], t["x1"]))
    return picked



def _join_digits(tokens: List[Dict[str, Any]], expected: int | None = None) -> str:
    """
    Concatenate digit tokens in left-to-right order. Optionally trim to expected length.
    """
    if not tokens:
        return ""
    s = "".join(re.sub(r"\D", "", str(t["text"])) for t in tokens)
    if expected is not None and len(s) > expected:
        s = s[:expected]
    return s


def _extract_checkbox_near_anchor(img_bgr, anchor, mode, page_w, page_h):
    ax1, ay1, ax2, ay2 = anchor["x1"], anchor["y1"], anchor["x2"], anchor["y2"]
    h = ay2 - ay1

    if mode == "left":
        # move right: reduce how far left from the text we probe
        side = 1.00 * h
        cx = ax1 + 9.00 * h   # was ~ -0.55*h -> now closer to the text (RIGHT)
        cy = (ay1 + ay2) / 2.0
        x1, x2 = cx - side/2, cx + side/2
        y1, y2 = cy - side/2, cy + side/2

    elif mode == "below":
        cx = (ax1 + ax2) / 2.0
        y1 = ay2 + 0.10 * h
        y2 = y1 + 1.30 * h
        x1 = cx - 0.85 * h
        x2 = cx + 0.85 * h
    else:
        return "false", 0.0, None

    x1 = max(0, int(round(x1))); y1 = max(0, int(round(y1)))
    x2 = min(page_w, int(round(x2))); y2 = min(page_h, int(round(y2)))
    if x2 <= x1 or y2 <= y1: return "false", 0.0, None

    patch = img_bgr[y1:y2, x1:x2]
    val, conf = detect_checkbox(patch, return_confidence=True)
    return val, float(conf), (x1, y1, x2, y2)


@dataclass(frozen=True)
class RoiSpec:
    name: str
    top: float
    left: float
    bottom: float
    right: float
    kind: str = "text"
    description: Optional[str] = None
    margin: Optional[float] = None
    auto_trim: Optional[float] = None
    expected_length: Optional[int] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    allowlist: Optional[str] = None
    preferred_ink: Optional[str] = None


def _safe_float(val, default=None, ctx=""):
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        # You can replace this with logging.warning if you prefer
        print(f"[ROI WARNING] Cannot parse float from {val!r} for {ctx}; ignoring.")
        return default


def _safe_int(val, default=None, ctx=""):
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        print(f"[ROI WARNING] Cannot parse int from {val!r} for {ctx}; ignoring.")
        return default


def _load_roi(path: Path) -> List[RoiSpec]:
    try:
        # note the utf-8-sig – important because roi_map.json has a BOM
        with path.open("r", encoding="utf-8-sig") as fh:
            raw = json.load(fh)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"ROI map file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in ROI map file: {path}") from e

    out: List[RoiSpec] = []
    for k, v in raw.items():
        if not isinstance(v, dict):
            raise ValueError(f"ROI '{k}' must be an object.")
        r = v.get("roi")
        if not (isinstance(r, list) and len(r) == 4):
            raise ValueError(f"ROI '{k}' must define 'roi' as a list of four numbers.")
        t, l, b, rr = r

        out.append(
            RoiSpec(
                name=k,
                top=float(t),
                left=float(l),
                bottom=float(b),
                right=float(rr),
                kind=v.get("kind", "text"),
                description=v.get("description"),
                margin=_safe_float(v.get("margin"), None, f"{k}.margin"),
                auto_trim=_safe_float(v.get("auto_trim"), None, f"{k}.auto_trim"),
                expected_length=_safe_int(v.get("expected_length"), None, f"{k}.expected_length"),
                min_length=_safe_int(v.get("min_length"), None, f"{k}.min_length"),
                max_length=_safe_int(v.get("max_length"), None, f"{k}.max_length"),
            )
        )
    return out


try:
    ROI_SPECS: List[RoiSpec] = _load_roi(ROI_MAP_PATH)
except Exception as exc:
    raise RuntimeError(f"Failed to load ROI specifications from '{ROI_MAP_PATH}': {exc}") from exc


def _bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()

def _rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

def _split(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return (img, _gray(img)) if img.ndim == 3 else (cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), img.copy())

def _scale(img: np.ndarray, s: float) -> np.ndarray:
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)

def _tidy(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _crop_margin(img: np.ndarray, frac: float) -> np.ndarray:
    if img.size == 0 or frac <= 0:
        return img
    h, w = img.shape[:2]
    dy = min(h // 2, max(0, int(round(h * frac))))
    dx = min(w // 2, max(0, int(round(w * frac))))
    y1, y2 = dy, h - dy
    x1, x2 = dx, w - dx
    return img if (y2 <= y1 or x2 <= x1) else img[y1:y2, x1:x2]

def _rect_from_roi(roi: RoiSpec, h: int, w: int) -> Tuple[int, int, int, int]:
    y1 = max(0, int(roi.top * h))
    y2 = min(h, int(roi.bottom * h))
    x1 = max(0, int(roi.left * w))
    x2 = min(w, int(roi.right * w))
    return y1, y2, x1, x2


def _load_image(uploaded_file) -> np.ndarray:
    if hasattr(uploaded_file, "read"):
        uploaded_file.seek(0)
        b = uploaded_file.read()
        im = Image.open(io.BytesIO(b)).convert("RGB")
    else:
        im = uploaded_file.convert("RGB") if isinstance(uploaded_file, Image.Image) else uploaded_file
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

def _prep_for_ocr(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return cv2.bilateralFilter(_gray(out), 5, 35, 35)


def _order_pts(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(d)],
        pts[np.argmax(s)],
        pts[np.argmax(d)],
    ], dtype="float32")

def _deskew(img: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Geometry normalization that:
    - keeps the entire page (no cropping),
    - preserves aspect ratio,
    - scales to fit inside TARGET_WIDTH x TARGET_HEIGHT and pads with white.
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img, False

    # scale so that the page fits inside the target box
    scale = min(TARGET_WIDTH / float(w), TARGET_HEIGHT / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # white canvas, center the resized page
    canvas = np.full((TARGET_HEIGHT, TARGET_WIDTH, 3), 255, dtype=resized.dtype)
    y_off = (TARGET_HEIGHT - new_h) // 2
    x_off = (TARGET_WIDTH - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    # we return False for "was_warped" because we no longer do perspective warping
    return canvas, False



def _suppress_guides(img: np.ndarray) -> np.ndarray:
    if img.size == 0:
        return img
    to_color = (img.ndim == 3)
    color = img if to_color else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    g = _gray(color)
    inv = cv2.adaptiveThreshold(cv2.GaussianBlur(g, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 7)
    h, w = g.shape[:2]
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, w // 6), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, h // 6)))
    dh = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    dv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    horiz = cv2.morphologyEx(cv2.morphologyEx(inv, cv2.MORPH_CLOSE, dh, 1), cv2.MORPH_OPEN, hk, 1)
    vert  = cv2.morphologyEx(cv2.morphologyEx(inv, cv2.MORPH_CLOSE, dv, 1), cv2.MORPH_OPEN, vk, 1)
    mask = cv2.bitwise_or(horiz, vert)
    if not np.any(mask):
        return g if not to_color else color
    dil = cv2.dilate(mask, np.ones((3, 3), np.uint8), 1)
    repaired = cv2.inpaint(color, dil, 3, cv2.INPAINT_TELEA)
    return repaired if to_color else _gray(repaired)

def _strip_rulings(graylike: np.ndarray) -> np.ndarray:
    if graylike.size == 0:
        return graylike
    g = _gray(graylike)
    h, w = g.shape[:2]
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min(h, max(3, int(round(h * 0.9))))))
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (min(w, max(3, int(round(w * 0.9)))), 1))
    v = cv2.morphologyEx(g, cv2.MORPH_OPEN, vk, 1)
    hline = cv2.morphologyEx(g, cv2.MORPH_OPEN, hk, 1)
    cleaned = cv2.subtract(g, cv2.bitwise_or(v, hline))
    cleaned = cv2.medianBlur(cleaned, 3)
    _, cleaned = cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cleaned

def _auto_trim(img: np.ndarray, max_frac: Optional[float], int_thr: float, var_thr: float) -> np.ndarray:
    if img.size == 0 or not max_frac or max_frac <= 0:
        return img
    max_frac = max(0.0, min(max_frac, 0.45))
    g = _gray(img)
    h, w = g.shape[:2]
    if h < 8 or w < 8:
        return img
    rmean, rstd = g.mean(1), g.std(1)
    cmean, cstd = g.mean(0), g.std(0)
    rlim = max(1, int(round(h * max_frac)))
    clim = max(1, int(round(w * max_frac)))

    def _front(mean, std, lim):
        k = 0
        for m, s in zip(mean, std):
            if k >= lim or not (m < int_thr and s < var_thr):
                break
            k += 1
        return k

    def _back(mean, std, lim):
        k = 0
        for m, s in zip(reversed(mean), reversed(std)):
            if k >= lim or not (m < int_thr and s < var_thr):
                break
            k += 1
        return k

    top = _front(rmean, rstd, rlim)
    bot = _back(rmean, rstd, rlim)
    left = _front(cmean, cstd, clim)
    right = _back(cmean, cstd, clim)
    y1, y2 = top, h - bot
    x1, x2 = left, w - right
    return img if (y2 <= y1 or x2 <= x1) else img[y1:y2, x1:x2]


def _pref_blue(img: np.ndarray) -> np.ndarray:
    if img.size == 0:
        return img
    bgr = _bgr(img)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([85, 40, 50], np.uint8), np.array([150, 255, 255], np.uint8))
    b, g, r = cv2.split(bgr)
    dom = cv2.normalize(cv2.subtract(b, cv2.max(g, r)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mix = cv2.GaussianBlur(cv2.max(mask, dom), (5, 5), 0)
    mix = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(mix)
    return cv2.morphologyEx(mix, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)

def _pref_dark(img: np.ndarray) -> np.ndarray:
    if img.size == 0:
        return img
    g = _gray(img)
    g = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(g)
    return cv2.medianBlur(g, 3)

def run_easyocr(image: np.ndarray, allowlist: Optional[str] = None) -> str:
    if image.size == 0:
        return ""
    return " ".join(_reader().readtext(_rgb(image), detail=0, paragraph=True, allowlist=allowlist)).strip()

def _ocr_text(image: np.ndarray, allowlist: Optional[str], minc: float) -> Tuple[str, float]:
    if image.size == 0:
        return "", 0.0
    out = _reader().readtext(_rgb(image), detail=1, paragraph=False, allowlist=allowlist)
    picked: List[str] = []
    backup: List[str] = []
    best = 0.0
    for _b, t, c in out:
        if not t:
            continue
        cleaned = _tidy(t)
        if not cleaned:
            continue
        c = float(c)
        best = max(best, c)
        backup.append(cleaned)
        if c >= minc:
            picked.append(cleaned)
    if picked:
        return " ".join(picked), best
    if backup and best >= max(0.01, minc * 0.7):
        return " ".join(backup), best
    return "", best

def _ocr_digits(image: np.ndarray, allowlist: str, minc: float) -> Tuple[str, float]:
    if image.size == 0:
        return "", 0.0
    out = _reader().readtext(_rgb(image), detail=1, paragraph=False, allowlist=allowlist)
    hi, lo, best = [], [], 0.0
    for _b, t, c in out:
        if not t:
            continue
        d = re.sub(r"[^0-9]", "", t)
        if not d:
            continue
        c = float(c)
        best = max(best, c)
        lo.append(d)
        if c >= minc:
            hi.append(d)
    if hi:
        return "".join(hi), best
    if lo and best >= max(0.01, minc * 0.8):
        return "".join(lo), best
    return "", best

def _variants(img: np.ndarray) -> Dict[str, np.ndarray]:
    color, g = _split(img)
    no_guides = _suppress_guides(color)
    g2 = _gray(no_guides)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return {
        "color": color,
        "gray": g,
        "gray_c": clahe.apply(g),
        "guide_gray": g2,
        "guide_c": clahe.apply(g2),
        "blue": _pref_blue(color),
        "blue_guide": _pref_blue(no_guides),
        "guide_color": _bgr(g2),
    }

def _binaries(v: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    g = v["gray"]
    gg = v["guide_gray"]
    blue_g = _gray(v["blue"])
    blue_bin = cv2.adaptiveThreshold(cv2.GaussianBlur(blue_g, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 9)
    g_bin = cv2.adaptiveThreshold(cv2.GaussianBlur(gg, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 7)

    # a bit different: build all main gates, then cleaning
    blur = cv2.GaussianBlur(g, (3, 3), 0)
    adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 9)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, k, 1)
    g_ad = cv2.adaptiveThreshold(cv2.GaussianBlur(gg, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 9)
    g_closed = cv2.morphologyEx(g_ad, cv2.MORPH_CLOSE, k, 1)

    return {
        "blue_bin": blue_bin,
        "blue_bin_clean": _strip_rulings(blue_bin),
        "g_bin": g_bin,
        "g_bin_clean": _strip_rulings(g_bin),
        "adapt": adapt,
        "closed": closed,
        "g_ad": g_ad,
        "g_closed": g_closed,
        "g_ad_clean": _strip_rulings(g_ad),
        "adapt_clean": _strip_rulings(adapt),
    }

def choose_best_digit_candidate(
    cands: List[Tuple[str, float]],
    expected_length: Optional[int] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
) -> Tuple[str, float]:
    seen = set()
    best = ("", 0.0, float("-inf"))  # digits, conf, score
    for raw, conf in cands:
        d = re.sub(r"[^0-9]", "", raw)
        if max_length and len(d) > max_length:
            d = d[:max_length]
        if not d:
            continue
        key = (d, int(round(conf * 1000)))
        if key in seen:
            continue
        seen.add(key)
        L = len(d)
        score = float(conf)
        if expected_length is not None:
            score += (0.35 if L == expected_length else -abs(L - expected_length) * 0.28)
        else:
            score += min(L * 0.05, 0.3)
        if min_length and L < min_length:
            score -= (min_length - L) * 0.25
        if max_length and L > max_length:
            score -= (L - max_length) * 0.2
        if score > best[2]:
            best = (d, conf, score)
    return best[0], best[1]


def extract_digits(
    image: np.ndarray,
    expected_length: Optional[int] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    return_confidence: bool = False,
) -> Union[str, Tuple[str, float]]:
    if image.size == 0:
        return ("", 0.0) if return_confidence else ""

    v = _variants(image)
    b = _binaries(v)

    # heuristic ink gating — different vars naming
    ink_ratio = float(cv2.countNonZero(b["blue_bin"])) / float(b["blue_bin"].size)
    max_blue = cv2.minMaxLoc(v["blue"])[1] if v["blue"].size else 0
    if ink_ratio < 0.0008 and max_blue < 80:
        r2 = float(cv2.countNonZero(b["g_bin"])) / float(b["g_bin"].size)
        max_guide_blue = cv2.minMaxLoc(v["blue_guide"])[1] if v["blue_guide"].size else 0
        if r2 < 0.0008 and max_guide_blue < 80:
            return ("", 0.0) if return_confidence else ""

    allow = "0123456789"
    cand_imgs = [
        ("gray", 2.2), ("gray_c", 2.4), ("color", 2.0),
        ("blue", 2.8), ("blue_bin", 3.0), ("blue_bin_clean", 3.0),
        ("guide_gray", 2.4), ("guide_c", 2.6), ("guide_color", 2.2),
        ("blue_guide", 3.0),
        ("g_bin", 3.0), ("g_bin_clean", 3.2),
        ("adapt", 3.0), ("adapt_clean", 3.0),
        ("g_ad", 3.0), ("g_ad_clean", 3.2),
    ]

    def _passes(images: Iterable[Tuple[str, float]], minc: float) -> List[Tuple[str, float]]:
        acc: List[Tuple[str, float]] = []
        for key, sc in images:
            img = v.get(key, b.get(key))
            if img is None:
                continue
            txt, conf = _ocr_digits(_scale(img, sc), allow, minc)
            if txt:
                if expected_length:
                    d = re.sub(r"[^0-9]", "", txt)
                    d = d[:max_length] if (max_length and len(d) > max_length) else d
                    if len(d) == expected_length and conf >= (minc + 0.03):
                        return [(d, conf)]
                acc.append((txt, conf))
        return acc

    # pass A: stricter
    got = _passes(cand_imgs, 0.55)
    if got and len(got) == 1 and expected_length:
        return got[0] if return_confidence else got[0][0]

    # pass B: looser
    got.extend(_passes(cand_imgs, 0.40))

    # fallback: OCR on inverted cleaned adapt
    if ink_ratio >= 0.0015:
        inv = cv2.bitwise_not(b["adapt_clean"])
        fb = run_easyocr(_scale(inv, 3.0), allowlist=allow)
        if fb:
            got.append((re.sub(r"[^0-9]", "", fb), 0.38))

    if not got:
        return ("", 0.0) if return_confidence else ""

    best, conf = choose_best_digit_candidate(got, expected_length, min_length, max_length)
    return (best, conf) if return_confidence else best


def _cnp_ok(d: str) -> bool:
    if len(d) != 13 or not d.isdigit():
        return False
    weights = [2, 7, 9, 1, 4, 6, 3, 5, 8, 2, 7, 9]
    total = sum(int(d[i]) * weights[i] for i in range(12))
    ctrl = total % 11
    if ctrl == 10:
        ctrl = 1
    return int(d[12]) == ctrl

def _cnp_best13(raw: str) -> str:
    best = ""
    for i in range(0, max(0, len(raw) - 12)):
        seg = raw[i:i + 13]
        if not seg.isdigit():
            continue
        if len(seg) > len(best):
            best = seg
        if _cnp_ok(seg):
            return seg
    return best

def _trim_to_ink(g: np.ndarray) -> np.ndarray:
    if g.size == 0:
        return g
    blur = cv2.GaussianBlur(_gray(g), (3, 3), 0)
    _, inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cols, rows = (inv > 0).sum(0), (inv > 0).sum(1)
    if not np.any(cols) or not np.any(rows):
        return _gray(g)
    ci, ri = np.where(cols > 0)[0], np.where(rows > 0)[0]
    x1, x2 = int(ci[0]), int(ci[-1])
    y1, y2 = int(ri[0]), int(ri[-1])
    px, py = max(1, (x2 - x1) // 40), max(1, (y2 - y1) // 10)
    x1, x2 = max(0, x1 - px), min(inv.shape[1], x2 + px)
    y1, y2 = max(0, y1 - py), min(inv.shape[0], y2 + py)
    return _gray(g) if (x2 <= x1 or y2 <= y1) else _gray(g)[y1:y2, x1:x2]

def _cnp_try_fix(d: str, confs: List[float]) -> str:
    if len(d) != 13 or not d.isdigit():
        return ""
    order = sorted(range(13), key=lambda i: confs[i] if i < len(confs) else 0.0)
    for i in order:
        orig = d[i]
        for ch in "0123456789":
            if ch == orig:
                continue
            t = d[:i] + ch + d[i + 1:]
            if _cnp_ok(t):
                return t
    return ""


def extract_cnp(image: np.ndarray) -> str:
    if image.size == 0:
        return ""

    # Attempt #1: equal-width slicing
    try:
        _, g0 = _split(image)
        g = _trim_to_ink(g0)
        h, w = g.shape[:2]
        box_fb = ""
        if w >= 130 and h >= 16:
            step = max(1, w // 13)
            parts, confs = [], []
            for i in range(13):
                a, b = i * step, (i + 1) * step if i < 12 else w
                seg = g[:, a:b]
                inner = _crop_margin(seg, 0.08)
                if inner.size == 0:
                    inner = seg
                d, c = _ocr_digits(_scale(inner, 2.8), "0123456789", 0.28)
                d = re.sub(r"[^0-9]", "", d)[:1]
                parts.append(d if d else "")
                confs.append(float(c) if d else 0.0)
            cand = "".join(parts)
            if len(cand) == 13 and _cnp_ok(cand):
                return cand
            if len(cand) == 13 and cand.isdigit():
                fixed = _cnp_try_fix(cand, confs)
                if fixed:
                    return fixed
                box_fb = cand
        else:
            box_fb = ""
    except Exception:
        box_fb = ""

    # Attempt #2: general digits
    d, conf = extract_digits(image, expected_length=13, min_length=11, max_length=13, return_confidence=True)
    if len(d) == 13 and _cnp_ok(d):
        return d
    if len(d) == 13 and conf >= 0.45:
        return d

    # Attempt #3: sweep several transforms
    v = _variants(image)
    cla = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_c = cla.apply(v["gray"])
    gg_c = cla.apply(v["guide_gray"])
    blur = cv2.GaussianBlur(g_c, (3, 3), 0)
    ad = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 9)
    joined = cv2.morphologyEx(ad, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)

    probes = [
        ("gray", v["gray"], 2.4),
        ("gray_c", g_c, 2.6),
        ("guide_gray", v["guide_gray"], 2.6),
        ("guide_c", gg_c, 2.8),
        ("ad", ad, 3.0),
        ("ad_inv", cv2.bitwise_not(ad), 3.0),
        ("joined", joined, 3.0),
        ("joined_inv", cv2.bitwise_not(joined), 3.0),
    ]
    best = ""
    for _, imgp, sc in probes:
        txt, c = _ocr_digits(_scale(imgp, sc), "0123456789", 0.35)
        digits = re.sub(r"[^0-9]", "", txt)
        if len(digits) >= 13:
            pick = _cnp_best13(digits)
            if _cnp_ok(pick):
                return pick
            if len(pick) == 13 and len(best) != 13:
                best = pick
            elif len(pick) > len(best):
                best = pick
        elif len(digits) > len(best):
            best = digits

    # Attempt #4: final fallback via generic
    fb = extract_digits(image, expected_length=13, min_length=11, max_length=13)
    chosen = _cnp_best13(fb)
    if _cnp_ok(chosen):
        return chosen
    return chosen or box_fb


def extract_text(
    image: np.ndarray,
    allowlist: Optional[str] = None,
    return_confidence: bool = False,
) -> Union[str, Tuple[str, float]]:
    if image.size == 0:
        return ("", 0.0) if return_confidence else ""

    v = _variants(image)
    b = _binaries(v)

    dark = _pref_dark(v["color"])
    guide_dark = _pref_dark(v["guide_color"])

    base_allow = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_/.,"  # slightly reordered
    allow = allowlist if allowlist is not None else base_allow

    plan = [
        (v["gray"], 2.0, 0.38, None),
        (v["gray_c"], 2.2, 0.40, None),
        (dark, 2.3, 0.40, None),
        (cv2.bitwise_not(b["adapt"]), 2.6, 0.45, allow),
        (b["adapt"], 2.6, 0.45, allow),
        (cv2.bitwise_not(b["closed"]), 2.8, 0.45, allow),
        (b["closed"], 2.8, 0.45, allow),
        (cv2.bitwise_not(b["adapt_clean"]), 2.8, 0.42, allow),
        (b["adapt_clean"], 2.8, 0.42, allow),
        (v["guide_gray"], 2.2, 0.38, None),
        (v["guide_c"], 2.4, 0.40, None),
        (guide_dark, 2.5, 0.40, None),
        (v["blue_guide"], 2.7, 0.45, allow),
        (b["g_ad"], 2.7, 0.45, allow),
        (cv2.bitwise_not(b["g_ad"]), 2.7, 0.45, allow),
        (b["g_closed"], 2.9, 0.46, allow),
        (cv2.bitwise_not(b["g_closed"]), 2.9, 0.46, allow),
        (b["g_ad_clean"], 2.9, 0.44, allow),
        (cv2.bitwise_not(b["g_ad_clean"]), 2.9, 0.44, allow),
    ]

    best_txt, best_conf = "", 0.0
    for imgp, sc, minc, alw in plan:
        txt, conf = _ocr_text(_scale(imgp, sc), alw, minc)
        txt = _tidy(txt)
        if txt and conf >= (minc + 0.05):
            return (txt, conf) if return_confidence else txt
        if txt:
            if conf > best_conf + 0.02:
                best_txt, best_conf = txt, conf
            elif abs(conf - best_conf) <= 0.02 and len(txt) > len(best_txt):
                best_txt, best_conf = txt, conf

    if best_txt:
        return (best_txt, best_conf) if return_confidence else best_txt

    fb = _tidy(run_easyocr(_scale(v["gray"], 2.4), allowlist=allowlist))
    return (fb, 0.25 if fb else 0.0) if return_confidence else fb

def format_date_from_digits(d: str) -> Optional[str]:
    if not d or len(d) < 6:
        return None
    dd, mm, yy = int(d[:2]), int(d[2:4]), int(d[4:6])
    year = yy + (2000 if yy < 50 else 1900)
    if 1 <= dd <= 31 and 1 <= mm <= 12 and 1900 <= year <= 2100:
        return f"{dd:02d}.{mm:02d}.{year:04d}"
    return None

def extract_date(image: np.ndarray, expected_length: int = 6, return_confidence: bool = False) -> Union[str, Tuple[str, float]]:
    ds, conf = extract_digits(image, expected_length=expected_length, min_length=expected_length, max_length=expected_length, return_confidence=True)
    pretty = format_date_from_digits(ds)
    if pretty:
        return (pretty, conf) if return_confidence else pretty
    digits = re.sub(r"[^0-9]", "", ds)[:expected_length]
    return (digits, conf) if return_confidence else digits

def parse_date(raw: str) -> str:
    digits = re.sub(r"[^0-9]", "", raw)
    pretty = format_date_from_digits(digits)
    return pretty if pretty else digits[:6].strip()


def detect_checkbox(image: np.ndarray, return_confidence: bool = False) -> Union[str, Tuple[str, float]]:
    if image.size == 0:
        return ("false", 0.0) if return_confidence else "false"

    color, _ = _split(image)
    inner = _auto_trim(_crop_margin(color, 0.12), max_frac=0.2, int_thr=210.0, var_thr=22.0)
    cleaned = _suppress_guides(inner)
    g = _gray(cleaned)

    cla = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enh = cla.apply(g)
    _, inv = cv2.threshold(cv2.GaussianBlur(enh, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    strokes = cv2.morphologyEx(cv2.morphologyEx(inv, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), 1),
                               cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), 1)

    h, w = strokes.shape[:2]
    area = float(h * w)
    if area == 0:
        return ("false", 0.0) if return_confidence else "false"

    num, _, stats, _ = cv2.connectedComponentsWithStats(strokes, connectivity=8)
    ink = 0.0
    elongated = False
    max_comp = 0.0
    for i in range(1, num):
        a = float(stats[i, cv2.CC_STAT_AREA])
        if a < 12:
            continue
        ink += a
        max_comp = max(max_comp, a)
        ww, hh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        elongated |= (ww >= 0.35 * w or hh >= 0.35 * h)

    if ink <= 0:
        return ("false", 0.0) if return_confidence else "false"

    ink_ratio = ink / area
    pad = max(1, int(round(min(h, w) * 0.18)))
    if pad * 2 >= min(h, w):
        pad = max(1, min(h, w) // 3)
    core = strokes[pad:h - pad, pad:w - pad] if (h - 2 * pad > 0 and w - 2 * pad > 0) else strokes
    core_ratio = float(cv2.countNonZero(core)) / area
    edge_ratio = max(0.0, ink_ratio - core_ratio)

    central = (core_ratio >= 0.0028 and cv2.countNonZero(core) >= 14)
    edgedom = edge_ratio > core_ratio * 1.2

    diag = False
    if max_comp >= min(24.0, area * 0.02):
        dil = cv2.dilate(strokes, np.ones((3, 3), np.uint8), 1)
        lines = cv2.HoughLinesP(cv2.Canny(dil, 50, 150, apertureSize=3), 1, np.pi / 180, 18,
                                minLineLength=min(w, h) * 0.4, maxLineGap=4)
        if lines is not None:
            for L in lines:
                x1, y1, x2, y2 = L[0]
                dx, dy = x2 - x1, y2 - y1
                if dx == 0 and dy == 0:
                    continue
                ang = min(abs(np.degrees(np.arctan2(dy, dx))), 180 - abs(np.degrees(np.arctan2(dy, dx))))
                if 30 <= ang <= 60:
                    diag = True
                    break

    if elongated and central and not edgedom and ink_ratio >= 0.006:
        c = min(1.0, max(0.45, ink_ratio * 90) + (0.1 if diag else 0.0))
        return ("true", c) if return_confidence else "true"
    if central and not edgedom and (ink_ratio >= 0.007 or diag):
        c = min(1.0, max(0.4, ink_ratio * 80 + (0.15 if diag else 0.0)))
        return ("true", c) if return_confidence else "true"

    c = max(0.0, min(0.4, core_ratio * 40))
    return ("false", c) if return_confidence else "false"



def extract_roi_value_with_conf(
    roi: RoiSpec,
    color_img: np.ndarray,
    gray_img: np.ndarray,
) -> Tuple[str, float]:
    """
    Extract a single ROI and return (value, confidence in [0,1]).
    This is the main function used by extract_fields.
    """
    h, w = gray_img.shape[:2]
    y1, y2, x1, x2 = _rect_from_roi(roi, h, w)
    patch = color_img[y1:y2, x1:x2]

    # Default margins / auto-trim per kind
    defaults_margin = {"digits": 0.06, "date": 0.06, "checkbox": 0.08, "text": 0.05}
    defaults_trim   = {"digits": None,  "date": None,  "checkbox": 0.18, "text": None}

    mg = roi.margin if roi.margin is not None else defaults_margin.get(roi.kind, 0.05)
    if roi.name in {"cnp", "cnp_copil"} and roi.margin is None:
        mg = 0.02
    if mg and mg > 0:
        patch = _crop_margin(patch, mg)

    tr = roi.auto_trim if roi.auto_trim is not None else defaults_trim.get(roi.kind)
    if tr and tr > 0:
        patch = _auto_trim(patch, tr, 200.0, 28.0)

    val: str
    conf: float

    match roi.kind:
        case "digits":
            if roi.name in {"cnp", "cnp_copil"}:
                # extract_cnp currently has no explicit confidence;
                # approximate based on checksum validity.
                val = extract_cnp(patch)
                conf = 1.0 if val and _cnp_ok(val) else (0.5 if val else 0.0)
            else:
                val, conf = extract_digits(
                    patch,
                    expected_length=roi.expected_length,
                    min_length=roi.min_length,
                    max_length=roi.max_length,
                    return_confidence=True,
                )
        case "date":
            val, conf = extract_date(
                patch,
                expected_length=roi.expected_length or 6,
                return_confidence=True,
            )
        case "checkbox":
            val, conf = detect_checkbox(
                patch,
                return_confidence=True,
            )
        case _:
            val, conf = extract_text(
                patch,
                allowlist=roi.allowlist,
                return_confidence=True,
            )

    return val.strip(), float(conf)


def extract_roi_value(roi: RoiSpec, color_img: np.ndarray, gray_img: np.ndarray) -> str:
    """
    Backwards-compatible wrapper: keep old signature
    that returns only the string value.
    """
    val, _ = extract_roi_value_with_conf(roi, color_img, gray_img)
    return val

def _extract_fields_by_layout(
    canonical_bgr: np.ndarray,
    ocr_entries: List[Dict[str, Any]],
) -> Tuple[np.ndarray, Dict[str, str], Dict[str, float]]:
    h, w = canonical_bgr.shape[:2]
    overlay = canonical_bgr.copy()
    fields: Dict[str, str] = {}
    confs: Dict[str, float] = {}

    # tipuri de câmpuri – ca să știm ce extractor să folosim pe patch
    digit_fields = {
        "valabil_pentru_luna_digits",
        "valabil_pentru_anul",
        "cod_indemnizatie",
        "cnp",
        "cnp_copil",
        "nr_inregistrare",
        "nr_zile",
        "cod_diagnostic",
    }

    date_fields = {"data_acordarii", "de_la", "pana_la"}

    checkbox_fields = {
        "initial_checkbox",
        "in_continuare_checkbox",
        "adult_checkbox",
    }

    def _expected_len_for(name: str) -> Optional[int]:
        """Lungimi tipice pentru câmpurile stricte numerice."""
        mapping = {
            "valabil_pentru_luna_digits": 2,
            "valabil_pentru_anul": 2,
            "cnp": 13,
            "cnp_copil": 13,
            "nr_zile": 2,
            "cod_diagnostic": 5,
        }
        return mapping.get(name)

    def record(
            name: str,
            value: str,
            region=None,
            color=(0, 200, 0),
            conf: float = 0.0,
    ):
        """
        - Dacă avem un dreptunghi verde (region), mai încercăm o dată OCR direct pe patch:
            * croppat ușor în interior (să scăpăm de linii de tabel)
            * upscalat (cifre mai groase)
            * extractor specializat: digits / date / checkbox / text
        """

        if region is not None:
            x1, y1, x2, y2 = region
            x1_i = max(0, int(round(x1)))
            x2_i = min(w, int(round(x2)))
            y1_i = max(0, int(round(y1)))
            y2_i = min(h, int(round(y2)))

            if x2_i > x1_i and y2_i > y1_i:
                # re-OCR pe patch dacă nu avem nimic sau conf este mic
                if (not value) or conf < 0.7:
                    rw = x2_i - x1_i
                    rh = y2_i - y1_i

                    # mic offset spre interior, ca să tăiem marginile de tabel
                    mx = int(0.06 * rw)
                    my = int(0.18 * rh)
                    cx1 = max(x1_i + mx, x1_i)
                    cx2 = min(x2_i - mx, x2_i)
                    cy1 = max(y1_i + my, y1_i)
                    cy2 = min(y2_i - my, y2_i)

                    if cx2 > cx1 and cy2 > cy1:
                        patch = canonical_bgr[cy1:cy2, cx1:cx2]

                        # upscaling pentru cifre mai clare
                        patch = cv2.resize(
                            patch,
                            None,
                            fx=2.5,
                            fy=2.5,
                            interpolation=cv2.INTER_CUBIC,
                        )

                        new_val = ""
                        new_conf = 0.0

                        if name in digit_fields:
                            exp = _expected_len_for(name)
                            new_val, new_conf = extract_digits(
                                patch,
                                expected_length=exp,
                                min_length=exp,
                                max_length=exp,
                                return_confidence=True,
                            )
                        elif name in date_fields:
                            new_val, new_conf = extract_date(
                                patch,
                                expected_length=6,
                                return_confidence=True,
                            )
                        elif name in checkbox_fields:
                            new_val, new_conf = detect_checkbox(
                                patch,
                                return_confidence=True,
                            )
                        else:
                            new_val, new_conf = extract_text(
                                patch,
                                allowlist=None,
                                return_confidence=True,
                            )

                        # dacă noul rezultat e mai bun, îl folosim
                        if new_val and (not value or new_conf > conf + 0.02):
                            value = new_val
                            conf = float(new_conf)

                # desenăm dreptunghiul + eticheta pe overlay
                cv2.rectangle(overlay, (x1_i, y1_i), (x2_i, y2_i), color, 2)
                cv2.putText(
                    overlay,
                    name,
                    (x1_i, max(10, y1_i - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        fields[name] = value or ""
        confs[name] = float(conf) if conf is not None else 0.0

    def region_from_tokens(tokens, fallback_region):
        if tokens:
            xs1 = [t["x1"] for t in tokens]
            xs2 = [t["x2"] for t in tokens]
            ys1 = [t["y1"] for t in tokens]
            ys2 = [t["y2"] for t in tokens]
            return (min(xs1), min(ys1), max(xs2), max(ys2))
        return fallback_region

    def find_anchor_any(candidates: List[str]) -> Optional[Dict[str, Any]]:
        """Try several patterns until one matches."""
        for pat in candidates:
            a = _find_anchor(ocr_entries, pat)
            if a is not None:
                return a
        return None

    # -----------------------------
    # 1) valabil_pentru_luna_digits
    # -----------------------------
    a = find_anchor_any([
        "valabil pentru luna",
        "valabll pentau iuna",   # from your OCR
    ])
    if a:
        ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
        ah = ay2 - ay1
        sx1 = ax2 + 0.3 * ah
        sx2 = min(w, ax2 + 3.0 * ah)   # two small boxes
        sy1 = max(0.0, ay1 - 1.0 * ah)
        sy2 = min(float(h), ay2 + 1.0 * ah)

        tokens = _collect_digit_tokens(ocr_entries, sx1, sx2, sy1, sy2)
        val = _join_digits(tokens, expected=2)
        field_conf = sum(t["conf"] for t in tokens) / len(tokens) if tokens else 0.0
        record(
            "valabil_pentru_luna_digits",
            val,
            region_from_tokens(tokens, (sx1, sy1, sx2, sy2)),
            conf=field_conf,
        )
    else:
        record("valabil_pentru_luna_digits", "", conf=0.0)

    # -----------------------------
    # 2) valabil_pentru_anul
    # -----------------------------
    a = find_anchor_any([
        "anul 20",
        "anui 20",               # from your OCR
    ])
    if a:
        ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
        ah = ay2 - ay1
        sx1 = ax2 + 0.3 * ah
        sx2 = min(w, ax2 + 3.0 * ah)   # two small boxes
        sy1 = max(0.0, ay1 - 1.0 * ah)
        sy2 = min(float(h), ay2 + 1.0 * ah)

        tokens = _collect_digit_tokens(ocr_entries, sx1, sx2, sy1, sy2)
        val = _join_digits(tokens, expected=2)
        field_conf = sum(t["conf"] for t in tokens) / len(tokens) if tokens else 0.0
        record(
            "valabil_pentru_anul",
            val,
            region_from_tokens(tokens, (sx1, sy1, sx2, sy2)),
            conf=field_conf,
        )
    else:
        record("valabil_pentru_anul", "", conf=0.0)

    # -----------------------------
    # 3) cod_indemnizatie
    # -----------------------------
    a = find_anchor_any([
        "cod indemnizatie",
        "cod indemnizație",
        "cod indemnizalie",
        "pt cod indemnizatie",
        "pt cod indemnizalie 1 17",  # your OCR
    ])
    if a:
        ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
        ah = ay2 - ay1
        sx1 = ax2 + 0.3 * ah
        sx2 = min(w, ax2 + 3.0 * ah)   # two small boxes
        sy1 = max(0.0, ay1 - 1.0 * ah)
        sy2 = min(float(h), ay2 + 1.0 * ah)

        tokens = _collect_digit_tokens(ocr_entries, sx1, sx2, sy1, sy2)
        val = _join_digits(tokens, expected=2)
        field_conf = sum(t["conf"] for t in tokens) / len(tokens) if tokens else 0.0
        record(
            "cod_indemnizatie",
            val,
            region_from_tokens(tokens, (sx1, sy1, sx2, sy2)),
            conf=field_conf,
        )
    else:
        record("cod_indemnizatie", "", conf=0.0)

    # -----------------------------
    # 4) cnp – 13 boxed digits
    # -----------------------------
    a = find_anchor_any(["cod numeric personal", "ccd numenc personal", "cod numenc personal"])
    if a:
        ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
        ah = ay2 - ay1
        sx1 = ax2 + 1.50 * ah  # slightly more RIGHT than before
        sx2 = ax2 + 16.00 * ah  # up to 13 boxes
        sy1 = ay1 - 1.00 * ah
        sy2 = ay2 + 1.00 * ah
        tokens = _collect_digit_tokens(ocr_entries, sx1, sx2, sy1, sy2)
        val = _join_digits(tokens, expected=13)
        field_conf = avg_conf(tokens)
        record("cnp", val, region_from_tokens(tokens, (sx1, sy1, sx2, sy2)), conf=field_conf)
    else:
        record("cnp", "", conf=0.0)

    # -----------------------------
    # 5) initial_checkbox
    # -----------------------------
    a = find_anchor_any([
        "iniţial",
        "initial",
        "inilia"
    ])
    if a:
        val, cb_conf, region = _extract_checkbox_near_anchor(
            canonical_bgr, a, "left", w, h
        )
        record("initial_checkbox", val, region, conf=cb_conf)
    else:
        record("initial_checkbox", "", conf=0.0)

    # -----------------------------
    # 6) in_continuare_checkbox
    # -----------------------------
    a = find_anchor_any([
        "în continuare",
        "in continuare",
        "in coniinuer8",          # your OCR
    ])
    if a:
        # Slightly shift the anchor to the LEFT so the checkbox moves left
        a2 = dict(a)
        ah = a2["y2"] - a2["y1"]
        a2["x1"] = a2["x1"] - 1.4 * ah  # small left correction

        val, cb_conf, region = _extract_checkbox_near_anchor(
            canonical_bgr, a2, "left", w, h
        )
        record("in_continuare_checkbox", val, region, conf=cb_conf)

    # -----------------------------
    # 7) nr_inregistrare – 5 digits to the right
    # -----------------------------
    a = find_anchor_any([
        "nr inreg",
        "nr. inreg",
        "nr inreg rc/fo",
        "inc inreg",              # your OCR
    ])
    if a:
        ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
        ah = ay2 - ay1
        sx1 = ax2 + 0.3 * ah
        sx2 = min(w, ax2 + 8.0 * ah)
        sy1 = max(0.0, ay1 - 1.0 * ah)
        sy2 = min(float(h), ay2 + 1.0 * ah)

        tokens = _collect_digit_tokens(ocr_entries, sx1, sx2, sy1, sy2)
        val = _join_digits(tokens, expected=5)
        field_conf = sum(t["conf"] for t in tokens) / len(tokens) if tokens else 0.0
        record(
            "nr_inregistrare",
            val,
            region_from_tokens(tokens, (sx1, sy1, sx2, sy2)),
            conf=field_conf,
        )
    else:
        record("nr_inregistrare", "", conf=0.0)

    # -----------------------------
    # 8) data_acordarii – 6 digits BELOW
    # -----------------------------
    a = find_anchor_any([
        "data acordarii",
        "dala acordarii",         # your OCR
    ])
    if a:
        ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
        ah = ay2 - ay1
        sx1 = ax1 - 1.0 * ah
        sx2 = ax2 + 1.55 * ah  # tight width to avoid "Data acordarii"
        sy1 = ay2 + 0.5 * ah
        sy2 = min(float(h), sy1 + 3.0 * ah)

        tokens = _collect_digit_tokens(ocr_entries, sx1, sx2, sy1, sy2)
        val = _join_digits(tokens, expected=6)
        field_conf = sum(t["conf"] for t in tokens) / len(tokens) if tokens else 0.0
        record(
            "data_acordarii",
            val,
            region_from_tokens(tokens, (sx1, sy1, sx2, sy2)),
            conf=field_conf,
        )
    else:
        record("data_acordarii", "", conf=0.0)

    # -----------------------------
    # 9) nr_zile – 2 digits BELOW
    # -----------------------------
    a = find_anchor_any([
        "nr zile",
        "nr. zile",
        # you can keep or drop "zile" – I'd keep just the two above
    ])
    if a:
        ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
        ah = ay2 - ay1
        sx1 = ax1 - 1.0 * ah
        sx2 = ax2 + 6.0 * ah
        sy1 = ay2 + 0.5 * ah
        sy2 = min(float(h), sy1 + 3.0 * ah)

        tokens = _collect_digit_tokens(ocr_entries, sx1, sx2, sy1, sy2)
        if tokens:
            # keep only the two boxes right under "Nr. zile"
            tokens = tokens[:2]

        val = _join_digits(tokens, expected=2)
        field_conf = sum(t["conf"] for t in tokens) / len(tokens) if tokens else 0.0
        record(
            "nr_zile",
            val,
            region_from_tokens(tokens, (sx1, sy1, sx2, sy2)),
            conf=field_conf,
        )
    else:
        record("nr_zile", "", conf=0.0)

    # -----------------------------
    # 10) de_la – 6 digits BELOW
    # -----------------------------
    a = find_anchor_any(["de la zi/luna/an", "de la", "de ia"])
    if a:
        ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
        ah = ay2 - ay1
        sx1 = ax2 - 5 * ah  # a bit more to the LEFT (was 0.20 * ah)
        sx2 = ax2 + 3.00 * ah  # tight width to avoid "Data acordarii"
        sy1 = ay2 + 0.50 * ah
        sy2 = sy1 + 3.00 * ah
        tokens = _collect_digit_tokens(ocr_entries, sx1, sx2, sy1, sy2)
        val = _join_digits(tokens, expected=6)
        field_conf = avg_conf(tokens)
        record("de_la", val, region_from_tokens(tokens, (sx1, sy1, sx2, sy2)), conf=field_conf)
    else:
        record("de_la", "", conf=0.0)

    # -----------------------------
    # 11) pana_la – 6 digits to the RIGHT
    # -----------------------------
    a = find_anchor_any(["pana la zi/luna/an", "pana la", "pana ia"])
    if a:
        ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
        ah = ay2 - ay1
        sx1 = ax2 - 5 * ah  # a bit more to the LEFT (was 0.20 * ah)
        sx2 = ax2 + 3.00 * ah  # tight width to avoid "Data acordarii"
        sy1 = ay2 + 0.50 * ah  # same good height
        sy2 = sy1 + 3.00 * ah
        tokens = _collect_digit_tokens(ocr_entries, sx1, sx2, sy1, sy2)
        val = _join_digits(tokens, expected=6)
        field_conf = avg_conf(tokens)
        record("pana_la", val, region_from_tokens(tokens, (sx1, sy1, sx2, sy2)), conf=field_conf)
    else:
        record("pana_la", "", conf=0.0)

    # -----------------------------
    # 12) cod_diagnostic – 5 digits to the RIGHT
    # -----------------------------
    a = find_anchor_any(["cod diagnostic", "dngncenc"])
    if a:
        ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
        ah = ay2 - ay1
        sx1 = ax2 - 5 * ah  # a bit more to the LEFT (was 0.20 * ah)
        sx2 = ax2 + 1.00 * ah  # tight width to avoid "Data acordarii"
        sy1 = ay2 + 0.50 * ah
        sy2 = sy1 + 3.00 * ah
        tokens = _collect_digit_tokens(ocr_entries, sx1, sx2, sy1, sy2)
        val = _join_digits(tokens, expected=5)
        field_conf = avg_conf(tokens)
        record("cod_diagnostic", val, region_from_tokens(tokens, (sx1, sy1, sx2, sy2)), conf=field_conf)
    else:
        record("cod_diagnostic", "", conf=0.0)

    # -----------------------------
    # 13) adult_checkbox
    # -----------------------------
    a = find_anchor_any([
        "adult",
    ])
    if a:
        val, cb_conf, region = _extract_checkbox_near_anchor(
            canonical_bgr, a, "below", w, h
        )
        record("adult_checkbox", val, region, conf=cb_conf)
    else:
        # EasyOCR often misses the vertical 'Adult' text; this may stay empty
        record("adult_checkbox", "", conf=0.0)

    return overlay, fields, confs



def extract_fields(uploaded_file, preview: bool = False, progress_callback=None, with_confidence: bool = False):
    """
    High-level pipeline:
    1) load image
    2) normalize geometry (scale+pad, no cropping)
    3) pre-process for OCR
    4) run EasyOCR on the whole page
    5) find anchors + digits/checkboxes by layout rules
    """
    state: Dict[str, Any] = {
        "uploaded_file": uploaded_file,
    }

    def report(frac: float, msg: str) -> None:
        if progress_callback:
            progress_callback(frac, msg)

    # 1) load
    report(0.05, "Image loaded")
    raw_bgr = _load_image(uploaded_file)
    state["raw_bgr"] = raw_bgr

    # 2) geometry normalization (scale+pad, no cutting)
    report(0.2, "Geometry normalized")
    canonical_bgr, was_warped = _deskew(raw_bgr)
    state["canonical_bgr"] = canonical_bgr
    state["was_warped"] = was_warped

    # 3) pre-processing
    report(0.35, "OCR pre-processing")
    ocr_gray = _prep_for_ocr(canonical_bgr)
    state["ocr_gray"] = ocr_gray

    # 4) run EasyOCR once on the full page
    report(0.6, "Text detection (EasyOCR)")
    ocr_raw = _reader().readtext(canonical_bgr)
    ocr_entries = _build_ocr_index(ocr_raw)

    try:
        import json
        with open("debug_ocr_lines.json", "w", encoding="utf-8") as f:
            json.dump(
                [
                    {"text": e["text"], "norm": e["norm"], "conf": e["conf"]}
                    for e in ocr_entries
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception:
        pass

    state["ocr_entries"] = ocr_entries

    # 5) layout-based field extraction
    report(0.9, "Field extraction")
    overlay, fields, confs = _extract_fields_by_layout(canonical_bgr, ocr_entries)

    if with_confidence:
        results = {
            name: {
                "value": fields.get(name, ""),
                "confidence": float(confs.get(name, 0.0)),
            }
            for name in fields.keys()
        }
    else:
        results = fields

    report(1.0, "Extraction complete")

    if preview:
        return overlay, results
    return results