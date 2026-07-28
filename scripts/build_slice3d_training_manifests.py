"""Add a patient-grouped SLICE-3D subset to the existing training manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path("data/combined_v2"))
    parser.add_argument("--slice-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-benign", type=int, default=8000)
    parser.add_argument("--val-benign", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2051)
    return parser.parse_args()


def capped_benign(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    positive = frame[frame["label"] == 1]
    negative = frame[frame["label"] == 0]
    if len(negative) > cap:
        negative = negative.sample(cap, random_state=seed)
    return pd.concat([positive, negative], ignore_index=True).sample(
        frac=1, random_state=seed
    )


def main() -> None:
    args = parse_args()
    image_root = (
        args.slice_dir / "images" / "ISIC_2024_Permissive_Training_Input"
    ).resolve()
    truth = pd.read_csv(args.slice_dir / "ground_truth.csv")
    metadata = pd.read_csv(image_root / "metadata.csv", low_memory=False)
    frame = metadata.merge(truth, on="isic_id", how="inner", validate="one_to_one")
    frame = frame[frame["patient_id"].notna()].copy()
    frame["label"] = frame["malignant"].astype(int)

    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=args.seed
    )
    train_index, val_index = next(
        splitter.split(frame, frame["label"], groups=frame["patient_id"])
    )
    slice_train = capped_benign(
        frame.iloc[train_index].copy(), args.train_benign, args.seed
    )
    slice_val = capped_benign(
        frame.iloc[val_index].copy(), args.val_benign, args.seed + 1
    )

    def project(source: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame()
        result["lesion_id"] = source["isic_id"]
        result["isic_id"] = source["isic_id"]
        result["image_type"] = "clinical: total-body lesion crop"
        result["diagnosis"] = source["label"].map(
            {0: "SLICE_BENIGN", 1: "SLICE_MALIGNANT"}
        )
        result["label"] = source["label"]
        result["skin_tone_class"] = pd.NA
        result["age_approx"] = source["age_approx"]
        result["sex"] = source["sex"]
        result["site"] = source["anatom_site_general"]
        result["image_path"] = source["isic_id"].map(
            lambda value: str((image_root / f"{value}.jpg").resolve())
        )
        result["source_domain"] = "SLICE3D_total_body"
        result["patient_id"] = source["patient_id"]
        result["img_id"] = source["isic_id"]
        return result

    base_train = pd.read_csv(args.base_dir / "train.csv")
    base_val = pd.read_csv(args.base_dir / "val.csv")
    train = pd.concat([base_train, project(slice_train)], ignore_index=True)
    val = pd.concat([base_val, project(slice_val)], ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.output_dir / "train.csv", index=False)
    val.to_csv(args.output_dir / "val.csv", index=False)
    print(
        f"train={len(train)} val={len(val)} "
        f"slice_train={len(slice_train)} slice_val={len(slice_val)} "
        f"slice_malignant={slice_train.label.sum()}/{slice_val.label.sum()}"
    )


if __name__ == "__main__":
    main()
