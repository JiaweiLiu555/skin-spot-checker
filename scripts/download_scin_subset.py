"""Download consented SCIN subsets needed for phone-domain robustness.

SCIN is governed by its own data-use license. This script downloads only the
LOOKS_HEALTHY and GROWTH_OR_MOLE categories from the official public bucket and
keeps the case identifier so future splits can remain case-grouped.
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

import pandas as pd


BUCKET_ROOT = "https://storage.googleapis.com/dx-scin-public-data/"
DEFAULT_CATEGORIES = ("LOOKS_HEALTHY", "GROWTH_OR_MOLE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    return parser.parse_args()


def download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(destination)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def main() -> None:
    args = parse_args()
    cases = pd.read_csv(args.cases, dtype={"case_id": str})
    selected = cases[cases["related_category"].isin(args.categories)].copy()
    rows: list[dict[str, str]] = []

    for case in selected.itertuples(index=False):
        for column in ("image_1_path", "image_2_path", "image_3_path"):
            source_path = getattr(case, column)
            if not isinstance(source_path, str) or not source_path:
                continue
            category = case.related_category.lower()
            filename = Path(source_path).name
            destination = args.output_dir / "images" / category / filename
            download(BUCKET_ROOT + source_path, destination)
            rows.append(
                {
                    "case_id": case.case_id,
                    "category": case.related_category,
                    "image_path": str(destination.resolve()),
                    "source_path": source_path,
                }
            )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.output_dir / "subset_manifest.csv", index=False)
    print(
        f"Downloaded {len(manifest)} images from {manifest['case_id'].nunique()} "
        f"case-grouped SCIN contributions."
    )


if __name__ == "__main__":
    main()
