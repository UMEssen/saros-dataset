import argparse
import json
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
import pandas as pd
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from tqdm import tqdm
from util import BodyParts, BodyRegions, load_nibabel_image_with_axcodes


def generate_dataset(
    source_root: Path,
    target_root: Path,
    info_df: pd.DataFrame,
    dataset: str,
) -> None:
    if dataset == "regions":
        label_map = {
            BodyRegions(i).name.lower(): int(BodyRegions(i))
            for i in sorted(BodyRegions)
        }
        filename = "body-regions.nii.gz"
        number = 557
    else:
        label_map = {
            BodyParts(i).name.lower(): int(BodyParts(i)) for i in sorted(BodyParts)
        }
        filename = "body-parts.nii.gz"
        number = 558

    task_name = f"Dataset{number}_BCA_2d_{args.dataset}"

    nnunet_folder = target_root / "nnUNet_training"
    target_dir = nnunet_folder / "nnUNet_raw" / task_name
    preprocessed_dir = nnunet_folder / "nnUNet_preprocessed" / task_name
    preprocessed_dir.mkdir(exist_ok=True, parents=True)

    num_training_cases = 0
    splits: List[Dict[str, List]] = []
    for _ in range(5):
        splits.append({"train": [], "val": []})
    for row in tqdm(info_df.itertuples(), total=len(info_df)):
        if row.split in {"fold-1", "fold-2", "fold-3", "fold-4", "fold-5"}:
            split = "Tr"
        elif row.split == "test":
            split = "Ts"
        else:
            raise ValueError(row.split)

        old_img = source_root / row.id / "image.nii.gz"
        old_label = source_root / row.id / filename

        img_nib = load_nibabel_image_with_axcodes(nib.load(old_img))
        img = img_nib.get_fdata()
        label_nib = load_nibabel_image_with_axcodes(nib.load(old_label))
        label = label_nib.get_fdata()
        # print(label.shape)
        # Get only the z slices that do not have 255
        annotated_slices = np.where(np.all(label != 255, axis=(0, 1)))[0]
        # print(annotated_slices)
        for sl in annotated_slices:
            img_sl = img[..., sl]
            label_sl = label[..., sl]
            assert 255 not in label_sl
            new_id = f"{row.id}_{sl}"
            if "fold" in row.split:
                num_training_cases += 1
                fold_id = int(row.split.split("-")[1]) - 1
                splits[fold_id]["val"].append(new_id)
                for i in range(5):
                    if i != fold_id:
                        splits[i]["train"].append(new_id)
            new_img = target_dir / f"images{split}" / f"{new_id}_0000.nii.gz"
            new_label = target_dir / f"labels{split}" / f"{new_id}.nii.gz"

            new_img.parent.mkdir(parents=True, exist_ok=True)
            new_label.parent.mkdir(parents=True, exist_ok=True)

            new_img_nib = nib.Nifti1Image(img_sl, img_nib.affine)
            new_label_nib = nib.Nifti1Image(label_sl, label_nib.affine)
            nib.save(new_img_nib, new_img)
            nib.save(new_label_nib, new_label)

    with (preprocessed_dir / "splits_final.json").open("w") as f:
        json.dump(splits, f)

    generate_dataset_json(
        output_folder=str(target_dir),
        channel_names={0: "CT"},
        labels=label_map,
        num_training_cases=num_training_cases,
        file_ending=".nii.gz",
        dataset_name=task_name,
        license="hands off!",
        dataset_release="v3",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help="Path to SAROS dataset.",
    )
    parser.add_argument(
        "--target-root",
        required=True,
        type=Path,
        help="Path to store the nnUNet dataset.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["parts", "regions"],
        help="Which dataset to generate.",
    )
    parser.add_argument(
        "--info-csv",
        default=Path("Segmentation Info.csv"),
        type=Path,
        help="Path to the file with the information about the cases.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.info_csv)

    generate_dataset(
        source_root=args.source_root,
        target_root=args.target_root,
        info_df=df,
        dataset=args.dataset,
    )
