from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

REPOSITORY = "SalmaneExploring/pad-ufes-20"
POSITIVE = {"BCC", "MEL", "SCC"}
TARGETS = {"MEL": 13, "BCC": 69, "SCC": 68, "ACK": 50, "NEV": 50, "SEK": 50}


def download(path: str, destination: Path, timeout: int) -> tuple[str, str]:
    if destination.exists() and destination.stat().st_size > 1_000:
        return path, "existing"
    url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/main/{quote(path)}"
    temporary = destination.with_suffix(".part")
    try:
        with requests.get(url, timeout=timeout, stream=True) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(128 * 1024):
                    handle.write(chunk)
        if temporary.stat().st_size <= 1_000:
            raise RuntimeError("downloaded file was too small")
        temporary.replace(destination)
        return path, "downloaded"
    except Exception as error:
        temporary.unlink(missing_ok=True)
        return path, f"error: {error}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a labeled PAD-UFES-20 smartphone OOD sample.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--file-index", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=Path("data/ood/pad_ufes_images"))
    parser.add_argument("--manifest", type=Path, default=Path("data/ood/pad_ufes_ood.csv"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    items = json.loads(args.file_index.read_text())
    paths = {
        Path(item["path"]).name: item["path"]
        for item in items
        if item.get("type") == "file" and item.get("path", "").endswith(".png")
    }
    metadata = pd.read_csv(args.metadata)
    metadata = metadata[metadata["img_id"].isin(paths)].copy()
    metadata = metadata.drop_duplicates(["diagnostic", "lesion_id"], keep="first")

    selections = []
    for offset, (diagnosis, target) in enumerate(TARGETS.items()):
        candidates = metadata[metadata["diagnostic"] == diagnosis]
        if len(candidates) < target:
            target = len(candidates)
        selections.append(candidates.sample(n=target, random_state=args.seed + offset))
    sample = pd.concat(selections, ignore_index=True)
    sample["source_path"] = sample["img_id"].map(paths)
    sample["diagnosis"] = sample["diagnostic"].str.upper()
    sample["label"] = sample["diagnosis"].isin(POSITIVE).astype(int)
    sample["skin_tone_class"] = sample["fitspatrick"]

    args.images.mkdir(parents=True, exist_ok=True)
    futures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for row in sample.itertuples(index=False):
            destination = args.images / row.img_id
            futures[executor.submit(download, row.source_path, destination, args.timeout)] = row.img_id
        failures = []
        for completed, future in enumerate(as_completed(futures), start=1):
            path, status = future.result()
            if status.startswith("error"):
                failures.append((path, status))
            if completed % 50 == 0 or completed == len(futures):
                print(f"Completed {completed}/{len(futures)}; failures={len(failures)}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} PAD-UFES-20 image downloads failed")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    sample["image_path"] = sample["img_id"].map(
        lambda name: os.path.relpath(args.images / name, start=args.manifest.parent)
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
    sample[columns].to_csv(args.manifest, index=False)
    print(sample["diagnosis"].value_counts().sort_index())
    print(f"Saved OOD manifest to {args.manifest}")


if __name__ == "__main__":
    main()
