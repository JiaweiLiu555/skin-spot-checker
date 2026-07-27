from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

ID_PATTERN = re.compile(r"^ISIC_[0-9]+$")
BASE_URL = "https://isic-archive.s3.amazonaws.com/images"


def download_one(image_id: str, output_dir: Path, timeout: int) -> tuple[str, str]:
    if not ID_PATTERN.fullmatch(image_id):
        return image_id, "invalid id"
    destination = output_dir / f"{image_id}.jpg"
    if destination.exists() and destination.stat().st_size > 1_000:
        return image_id, "existing"
    temporary = destination.with_suffix(".jpg.part")
    try:
        with requests.get(f"{BASE_URL}/{image_id}.jpg", timeout=timeout, stream=True) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    handle.write(chunk)
        if temporary.stat().st_size <= 1_000:
            temporary.unlink(missing_ok=True)
            return image_id, "too small"
        temporary.replace(destination)
        return image_id, "downloaded"
    except Exception as error:  # Report every failed ID and continue the resumable download.
        temporary.unlink(missing_ok=True)
        return image_id, f"error: {error}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MILK10k clinical close-up images.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-type", default="clinical: close-up")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    frame = pd.read_csv(args.metadata)
    image_ids = frame.loc[frame["image_type"] == args.image_type, "isic_id"].astype(str).tolist()
    args.output.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, image_id, args.output, args.timeout): image_id
            for image_id in image_ids
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            image_id, status = future.result()
            category = status.split(":", 1)[0]
            counts[category] = counts.get(category, 0) + 1
            if status.startswith("error") or status in {"invalid id", "too small"}:
                failures.append((image_id, status))
            if completed % 250 == 0 or completed == len(futures):
                print(f"Completed {completed}/{len(futures)}; status={counts}", flush=True)
    if failures:
        failure_path = args.output / "download_failures.csv"
        pd.DataFrame(failures, columns=["isic_id", "error"]).to_csv(failure_path, index=False)
        raise RuntimeError(f"{len(failures)} downloads failed; see {failure_path}")
    (args.output / "download_failures.csv").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
