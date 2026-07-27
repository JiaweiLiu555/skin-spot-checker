from __future__ import annotations

"""Download every indexed PAD-UFES image, reusing existing files."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts.prepare_pad_ood import download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-index", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    paths = [
        item["path"]
        for item in json.loads(args.file_index.read_text())
        if item.get("type") == "file" and item.get("path", "").endswith(".png")
    ]
    args.images.mkdir(parents=True, exist_ok=True)
    failures = []
    counts = {"existing": 0, "downloaded": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download, path, args.images / Path(path).name, args.timeout): path
            for path in paths
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            path, status = future.result()
            if status.startswith("error"):
                failures.append((path, status))
            else:
                counts[status] = counts.get(status, 0) + 1
            if completed % 100 == 0 or completed == len(futures):
                print(
                    f"completed={completed}/{len(futures)} existing={counts['existing']} "
                    f"downloaded={counts['downloaded']} failures={len(failures)}",
                    flush=True,
                )
    if failures:
        error_path = args.images.parent / "pad_download_failures.json"
        error_path.write_text(json.dumps(failures, indent=2))
        raise RuntimeError(f"{len(failures)} downloads failed; see {error_path}")


if __name__ == "__main__":
    main()
