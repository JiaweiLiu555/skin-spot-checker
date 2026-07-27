from __future__ import annotations

"""Combine MILK clinical and patient-separated PAD phone manifests."""

import argparse
import os
from pathlib import Path

import pandas as pd


def resolved_frame(path: Path, source: str):
    frame = pd.read_csv(path).copy()
    frame["image_path"] = frame["image_path"].map(
        lambda value: str((path.parent / str(value)).resolve())
        if not Path(str(value)).is_absolute()
        else str(Path(str(value)).resolve())
    )
    frame["source_domain"] = source
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--phone-dir", type=Path, default=Path("data/pad_v2"))
    parser.add_argument("--output", type=Path, default=Path("data/combined_v2"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        clinical = resolved_frame(args.clinical_dir / f"{split}.csv", "MILK10k_clinical")
        phone = resolved_frame(args.phone_dir / f"{split}.csv", "PAD_UFES_phone")
        combined = pd.concat([clinical, phone], ignore_index=True, sort=False)
        combined["image_path"] = combined["image_path"].map(
            lambda value: os.path.relpath(value, start=args.output.resolve())
        )
        combined.to_csv(args.output / f"{split}.csv", index=False)
        print(
            split,
            combined.groupby(["source_domain", "label"]).size().to_dict(),
            flush=True,
        )


if __name__ == "__main__":
    main()
