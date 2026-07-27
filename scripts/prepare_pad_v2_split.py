from __future__ import annotations

"""Create a locked, patient-disjoint PAD-UFES-20 train/val/test split."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


POSITIVE = {"BCC", "MEL", "SCC"}


def split_once(frame, groups, seed):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_indices, holdout_indices = next(
        splitter.split(frame, y=frame["diagnostic"], groups=groups)
    )
    return frame.iloc[train_indices].copy(), frame.iloc[holdout_indices].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--file-index", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=Path("data/ood/pad_ufes_images"))
    parser.add_argument("--output", type=Path, default=Path("data/pad_v2"))
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    indexed = {
        Path(item["path"]).name
        for item in json.loads(args.file_index.read_text())
        if item.get("type") == "file" and item.get("path", "").endswith(".png")
    }
    frame = pd.read_csv(args.metadata)
    frame = frame[frame["img_id"].isin(indexed)].copy()
    frame["diagnosis"] = frame["diagnostic"].str.upper()
    frame["label"] = frame["diagnosis"].isin(POSITIVE).astype(int)
    frame["skin_tone_class"] = frame["fitspatrick"]

    development, test = split_once(frame, frame["patient_id"], args.seed)
    train, val = split_once(development, development["patient_id"], args.seed + 1)
    splits = {"train": train, "val": val, "test": test}
    patient_sets = {name: set(part.patient_id) for name, part in splits.items()}
    assert patient_sets["train"].isdisjoint(patient_sets["val"])
    assert patient_sets["train"].isdisjoint(patient_sets["test"])
    assert patient_sets["val"].isdisjoint(patient_sets["test"])

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": "PAD-UFES-20 indexed subset",
        "split_unit": "patient_id",
        "seed": args.seed,
        "test_locked_before_v2_training": True,
        "splits": {},
    }
    for name, part in splits.items():
        part = part.copy()
        part["image_path"] = part["img_id"].map(
            lambda image: os.path.relpath(args.images / image, start=args.output)
        )
        columns = [
            "patient_id",
            "lesion_id",
            "img_id",
            "diagnosis",
            "label",
            "skin_tone_class",
            "image_path",
        ]
        part[columns].to_csv(args.output / f"{name}.csv", index=False)
        summary["splits"][name] = {
            "images": int(len(part)),
            "patients": int(part.patient_id.nunique()),
            "lesions": int(part.lesion_id.nunique()),
            "higher_concern": int(part.label.sum()),
            "lower_concern": int((1 - part.label).sum()),
            "diagnoses": part.diagnosis.value_counts().sort_index().to_dict(),
            "downloaded_images_present": int(
                part.img_id.map(lambda image: (args.images / image).exists()).sum()
            ),
        }
    (args.output / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
