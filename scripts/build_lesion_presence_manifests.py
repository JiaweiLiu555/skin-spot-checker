"""Build case/patient-separated manifests for the visible-lesion gate."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-dir", type=Path, default=Path("data/combined_v16"))
    parser.add_argument("--scin-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2052)
    return parser.parse_args()


def scin_split(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cases = frame[["case_id", "category"]].drop_duplicates()
    train_cases: list[str] = []
    val_cases: list[str] = []
    for _, category_cases in cases.groupby("category"):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, val_idx = next(
            splitter.split(category_cases, groups=category_cases["case_id"])
        )
        train_cases.extend(category_cases.iloc[train_idx]["case_id"].tolist())
        val_cases.extend(category_cases.iloc[val_idx]["case_id"].tolist())
    return (
        frame[frame["case_id"].isin(train_cases)].copy(),
        frame[frame["case_id"].isin(val_cases)].copy(),
    )


def project_scin(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[["case_id", "category", "image_path"]].copy()
    result["label"] = result["category"].eq("GROWTH_OR_MOLE").astype(int)
    result["group_id"] = "SCIN_" + result["case_id"].astype(str)
    result["source_domain"] = "SCIN_phone"
    return result[["image_path", "label", "group_id", "source_domain", "category"]]


def project_existing(frame: pd.DataFrame, sample: int, seed: int) -> pd.DataFrame:
    if len(frame) > sample:
        frame = frame.sample(sample, random_state=seed)
    result = pd.DataFrame(
        {
            "image_path": frame["image_path"],
            "label": 1,
            "group_id": frame["source_domain"].astype(str)
            + "_"
            + frame["patient_id"].fillna(frame["lesion_id"]).astype(str),
            "source_domain": frame["source_domain"],
            "category": "VISIBLE_LESION",
        }
    )
    return result


def main() -> None:
    args = parse_args()
    scin = pd.read_csv(args.scin_manifest, dtype={"case_id": str})
    scin_train, scin_val = scin_split(scin, args.seed)
    combined_train = pd.read_csv(args.combined_dir / "train.csv")
    combined_val = pd.read_csv(args.combined_dir / "val.csv")

    pad_train = combined_train[combined_train["source_domain"] == "PAD_UFES_phone"]
    pad_val = combined_val[combined_val["source_domain"] == "PAD_UFES_phone"]
    slice_train = combined_train[
        combined_train["source_domain"] == "SLICE3D_total_body"
    ]
    slice_val = combined_val[combined_val["source_domain"] == "SLICE3D_total_body"]

    train = pd.concat(
        [
            project_scin(scin_train),
            project_existing(pad_train, 636, args.seed),
            project_existing(slice_train, 700, args.seed),
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=args.seed)
    val = pd.concat(
        [
            project_scin(scin_val),
            project_existing(pad_val, 159, args.seed),
            project_existing(slice_val, 175, args.seed),
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.output_dir / "train.csv", index=False)
    val.to_csv(args.output_dir / "val.csv", index=False)
    print(
        f"train={len(train)} positives={train.label.sum()} "
        f"val={len(val)} positives={val.label.sum()}"
    )


if __name__ == "__main__":
    main()
