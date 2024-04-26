import argparse
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from p_tqdm import p_map
from surface_distance import (
    compute_average_surface_distance,
    compute_surface_dice_at_tolerance,
    compute_surface_distances,
)
from util import BodyParts, BodyRegions

metrics = [
    "precision",
    "recall",
    "dice",
    "surface_distance_3mm",
]


def compute_metrics(
    gt: np.ndarray, pred: np.ndarray, spacing: Tuple
) -> Dict[str, float]:
    # There is no GT and no prediction
    if not gt.max() and not pred.max():
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
    # There is GT but no prediction
    elif gt.max() and not pred.max():
        return {
            "tp": 0,
            "fp": 0,
            "fn": gt.sum(),
            **{m: 0 for m in metrics},
        }
    # There is prediction but no GT
    elif pred.max() and not gt.max():
        return {
            "tp": 0,
            "fp": pred.sum(),
            "fn": 0,
            **{m: 0 for m in metrics},
        }
    else:
        tp = (gt & pred).sum()
        fp = (~gt & pred).sum()
        fn = (gt & ~pred).sum()
        sd = compute_surface_distances(gt, pred, spacing)
        avg_gt_to_pred, avg_pred_to_gt = compute_average_surface_distance(sd)
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp),
            "recall": tp / (tp + fn),
            "dice": tp / (tp + 0.5 * fp + 0.5 * fn),
            "avg_surface_distance": (avg_gt_to_pred + avg_pred_to_gt) / 2,
            "surface_distance_1mm": compute_surface_dice_at_tolerance(sd, 1.0),
            "surface_distance_2mm": compute_surface_dice_at_tolerance(sd, 2.0),
            "surface_distance_3mm": compute_surface_dice_at_tolerance(sd, 3.0),
        }


def _worker(
    subject: str,
    gt_dir: Path,
    pred_dir: Path,
    class_map: dict,
    ignore_label: Optional[int] = None,
) -> List[Dict[str, Any]]:
    sitk_gt = sitk.ReadImage(str(gt_dir / f"{subject}.nii.gz"))
    sitk_label = sitk.ReadImage(str(pred_dir / f"{subject}.nii.gz"))

    spacing1 = sitk_gt.GetSpacing()
    spacing2 = sitk_label.GetSpacing()

    assert spacing1 == spacing2, "ground truth and prediction have different spacing"

    gt_arr = sitk.GetArrayFromImage(sitk_gt)
    pred_arr = sitk.GetArrayFromImage(sitk_label)

    if ignore_label is not None:
        mask = gt_arr == ignore_label
        gt_arr[mask] = 0
        pred_arr[mask] = 0

    r = []
    for roi_name, idx in class_map.items():
        gt = gt_arr == idx
        pred = pred_arr == idx
        res = compute_metrics(gt, pred, spacing1)
        for k, v in res.items():
            r.append(
                {
                    "subject": subject,
                    "metric": k,
                    "label": roi_name,
                    "value": v,
                }
            )
    return r


def eval_strategy(
    gt_dir: Path,
    pred_dir: Path,
    res_dir: Path,
    dataset: str,
    ignore_label: Optional[int] = None,
) -> None:
    if dataset == "regions":
        class_map = {
            BodyRegions(i).name.lower(): int(BodyRegions(i))
            for i in range(1, len(BodyRegions))
        }
    elif dataset == "parts":
        class_map = {
            BodyParts(i).name.lower(): int(BodyParts(i))
            for i in range(1, len(BodyParts))
        }
    else:
        raise ValueError("Invalid dataset")

    if not pred_dir.exists():
        print(f"Folder {pred_dir} does not exist")
        return

    subjects = sorted([x.stem.split(".")[0] for x in gt_dir.glob("*.nii.gz")])

    # Use multiple threads to calculate the metrics
    res = p_map(
        partial(
            _worker,
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            class_map=class_map,
            ignore_label=ignore_label,
        ),
        subjects,
        num_cpus=8,
        disable=False,
    )
    res = pd.DataFrame([rr for r in res for rr in r])

    res_dir.mkdir(exist_ok=True)

    res.to_excel(
        res_dir / f"results_{dataset}_{pred_dir.name}.xlsx",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-folder",
        required=True,
        type=Path,
        help="Path where the ground truth is stored.",
    )
    parser.add_argument(
        "--pred-folder",
        required=True,
        type=Path,
        help="Path where the predictions are stored.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        choices=["regions", "parts"],
        help="Type of evaluation to perform.",
    )
    parser.add_argument(
        "--results-folder",
        type=Path,
        default="results",
        help="Folder where the results will be stored.",
    )
    parser.add_argument(
        "--ignore-label",
        type=int,
        default=None,
        help="Label to ignore in the evaluation (used for sparse evaluation)",
    )
    args = parser.parse_args()

    eval_strategy(
        gt_dir=args.gt_folder,
        pred_dir=args.pred_folder,
        res_dir=args.results_folder,
        dataset=args.dataset,
        ignore_label=args.ignore_label,
    )
