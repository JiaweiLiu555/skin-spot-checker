from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

HIGHER_CONCERN_CLASSES = ("AKIEC", "BCC", "MAL_OTH", "MEL", "SCCKA")
ALL_DIAGNOSIS_CLASSES = (
    "AKIEC",
    "BCC",
    "BEN_OTH",
    "BKL",
    "DF",
    "INF",
    "MAL_OTH",
    "MEL",
    "NV",
    "SCCKA",
    "VASC",
)


def assign_labels(ground_truth: pd.DataFrame) -> pd.DataFrame:
    missing = set(ALL_DIAGNOSIS_CLASSES).difference(ground_truth.columns)
    if missing:
        raise ValueError(f"Ground truth is missing diagnosis columns: {sorted(missing)}")
    output = ground_truth.copy()
    output["diagnosis"] = output[list(ALL_DIAGNOSIS_CLASSES)].idxmax(axis=1)
    output["label"] = output[list(HIGHER_CONCERN_CLASSES)].max(axis=1).astype(int)
    return output[["lesion_id", "diagnosis", "label"]]


def stratified_lesion_split(frame: pd.DataFrame, seed: int):
    lesions = frame[["lesion_id", "label"]].drop_duplicates()
    if lesions.groupby("lesion_id")["label"].nunique().max() != 1:
        raise ValueError("A lesion has conflicting labels.")
    train_lesions, holdout_lesions = train_test_split(
        lesions,
        test_size=0.20,
        random_state=seed,
        stratify=lesions["label"],
    )
    val_lesions, test_lesions = train_test_split(
        holdout_lesions,
        test_size=0.50,
        random_state=seed + 1,
        stratify=holdout_lesions["label"],
    )
    ids = (
        set(train_lesions["lesion_id"]),
        set(val_lesions["lesion_id"]),
        set(test_lesions["lesion_id"]),
    )
    return tuple(frame[frame["lesion_id"].isin(group)].copy() for group in ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MILK10k clinical-image manifests.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--image-type", default="clinical: close-up")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)
    required = {"lesion_id", "isic_id", "image_type"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    metadata = metadata[metadata["image_type"] == args.image_type].copy()
    labels = assign_labels(pd.read_csv(args.ground_truth))
    frame = metadata.merge(labels, on="lesion_id", how="inner", validate="many_to_one")

    frame["image_path_absolute"] = frame["isic_id"].map(
        lambda image_id: (args.images / f"{image_id}.jpg").resolve()
    )
    missing_images = (~frame["image_path_absolute"].map(Path.exists)).sum()
    frame = frame[frame["image_path_absolute"].map(Path.exists)].copy()
    if frame.empty:
        raise FileNotFoundError("No metadata rows matched image files.")
    if frame["label"].nunique() < 2:
        raise ValueError("Both higher-concern and lower-concern examples are required.")

    train, val, test = stratified_lesion_split(frame, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_columns = [
        column
        for column in (
            "lesion_id",
            "isic_id",
            "image_type",
            "diagnosis",
            "label",
            "skin_tone_class",
            "age_approx",
            "sex",
            "site",
            "image_path",
        )
        if column in frame.columns or column == "image_path"
    ]
    summary = {
        "source": "MILK10k",
        "image_type": args.image_type,
        "higher_concern_classes": list(HIGHER_CONCERN_CLASSES),
        "lower_concern_classes": [
            value for value in ALL_DIAGNOSIS_CLASSES if value not in HIGHER_CONCERN_CLASSES
        ],
        "missing_images": int(missing_images),
        "splits": {},
        "diagnosis_counts": frame["diagnosis"].value_counts().to_dict(),
    }
    for name, split in (("train", train), ("val", val), ("test", test)):
        split["image_path"] = split["image_path_absolute"].map(
            lambda path: os.path.relpath(path, start=args.output.resolve())
        )
        split[manifest_columns].to_csv(args.output / f"{name}.csv", index=False)
        summary["splits"][name] = {
            "images": int(len(split)),
            "higher_concern": int(split["label"].sum()),
            "lower_concern": int((1 - split["label"]).sum()),
            "lesions": int(split["lesion_id"].nunique()),
        }
        print(f"{name}: {summary['splits'][name]}")
    (args.output / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Skipped {missing_images} rows with no matching image file.")


if __name__ == "__main__":
    main()
