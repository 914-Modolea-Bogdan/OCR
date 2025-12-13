from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from ocr_utils import extract_fields
from validation import validate_fields


IMPORTANT_FIELDS = [
    "valabil_pentru_luna_digits",
    "valabil_pentru_anul",
    "cod_indemnizatie",
    "cnp",
    "nr_inregistrare",
    "data_acordarii",
    "nr_zile",
    "de_la",
    "pana_la",
    "cod_diagnostic",
    "adult_checkbox",
]


def _flatten_results(results: Dict[str, Any]) -> Dict[str, str]:
    flat: Dict[str, str] = {}
    for name, val in results.items():
        if isinstance(val, dict):
            val = val.get("value", "")
        flat[name] = "" if val is None else str(val)
    return flat


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # classic DP
    prev = list(range(len(b) + 1))
    cur = [0] * (len(b) + 1)
    for i, ca in enumerate(a, start=1):
        cur[0] = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + cost  # substitution
            )
        prev, cur = cur, prev
    return prev[-1]


def _char_accuracy(gt: str, pred: str) -> float:
    if not gt and not pred:
        return 1.0
    dist = _levenshtein(gt, pred)
    max_len = max(len(gt), len(pred))
    if max_len == 0:
        return 1.0
    return 1.0 - dist / max_len


@dataclass
class FieldMetrics:
    field: str
    exact_match: bool
    char_acc: float


@dataclass
class ImageMetrics:
    image_name: str
    field_accuracy: float
    char_accuracy: float
    num_fields: int
    num_validation_issues: int


def evaluate_image(image_path: Path, gt_path: Path) -> ImageMetrics:
    with image_path.open("rb") as f:
        overlay, results = extract_fields(
            f,
            preview=True,
            progress_callback=None,
            with_confidence=True,
        )

    flat = _flatten_results(results)

    with gt_path.open("r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    metrics: List[FieldMetrics] = []

    for field in IMPORTANT_FIELDS:
        gt_val = str(ground_truth.get(field, "") or "")
        pred_val = str(flat.get(field, "") or "")

        exact = gt_val == pred_val
        ca = _char_accuracy(gt_val, pred_val)

        metrics.append(FieldMetrics(field=field, exact_match=exact, char_acc=ca))

    if metrics:
        field_acc = sum(1 for m in metrics if m.exact_match) / len(metrics)
        char_acc = sum(m.char_acc for m in metrics) / len(metrics)
    else:
        field_acc = 0.0
        char_acc = 0.0

    validation_issues = validate_fields(results)

    return ImageMetrics(
        image_name=image_path.name,
        field_accuracy=field_acc,
        char_accuracy=char_acc,
        num_fields=len(metrics),
        num_validation_issues=len(validation_issues),
    )


def run_experiment(data_dir: Path, tag: str, output_dir: Path) -> None:
    data_dir = data_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    for img_path in sorted(data_dir.glob("*.png")):
        gt_path = img_path.with_suffix(".json")
        if not gt_path.exists():
            print(f"[WARN] Ground-truth JSON missing for {img_path.name}, skipping.")
            continue
        pairs.append((img_path, gt_path))

    if not pairs:
        print(f"No PNG+JSON pairs found under {data_dir}")
        return

    image_metrics: List[ImageMetrics] = []
    rows = []

    for img_path, gt_path in pairs:
        print(f"Evaluating {img_path.name} ...")
        m = evaluate_image(img_path, gt_path)
        image_metrics.append(m)
        rows.append({
            "image_name": m.image_name,
            "field_accuracy": m.field_accuracy,
            "char_accuracy": m.char_accuracy,
            "num_fields": m.num_fields,
            "num_validation_issues": m.num_validation_issues,
        })

    df = pd.DataFrame(rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Per-run CSV
    run_csv = output_dir / f"run_{tag}_{timestamp}.csv"
    df.to_csv(run_csv, index=False)
    print(f"Saved per-image metrics to {run_csv}")

    # Append summary row to experiments_log.csv
    log_path = output_dir / "experiments_log.csv"
    summary = {
        "timestamp": timestamp,
        "tag": tag,
        "num_images": len(image_metrics),
        "mean_field_accuracy": df["field_accuracy"].mean(),
        "mean_char_accuracy": df["char_accuracy"].mean(),
        "mean_validation_issues": df["num_validation_issues"].mean(),
    }

    log_exists = log_path.exists()
    with log_path.open("a", encoding="utf-8") as f:
        if not log_exists:
            # write header
            f.write(",".join(summary.keys()) + "\n")
        f.write(",".join(str(summary[k]) for k in summary.keys()) + "\n")

    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def main(
    data_dir: str | Path = "experiments/data",
    tag: str = "baseline",
    output_dir: str | Path = "experiments",
) -> None:
    """
    Run an OCR evaluation experiment.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing .png images and matching .json ground-truth files.
    tag : str
        Short label for this experiment run (stored in experiments_log.csv).
    output_dir : str | Path
        Directory where results and experiment log will be saved.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    run_experiment(data_dir, tag, output_dir)


if __name__ == "__main__":
    # Default run; you can change these or call main() from elsewhere.
    main()
