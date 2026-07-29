"""Replay a batch against a running API, as if it were a day of production traffic.

    python scripts/replay_batch.py --batch resisc45_real_n500 --url http://localhost:8000

Everything up to brick 4 measured the model by calling it in-process. This sends
the same tiles over HTTP to the real service, so what gets measured afterwards is
what the deployed thing actually saw and logged -- including the encode/decode,
the upload validation, and the sampling decision. Stdlib only, so it runs from
any of the project venvs.
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH_ROOT = ROOT / "data" / "batches"


def post_image(url: str, path: Path, timeout: float = 30.0) -> dict:
    boundary = uuid.uuid4().hex
    content_type = mimetypes.guess_type(path.name)[0] or "image/png"
    payload = path.read_bytes()

    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode(),
        f"Content-Type: {content_type}\r\n\r\n".encode(),
        payload,
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    request = urllib.request.Request(
        f"{url}/predict", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    batch_dir = BATCH_ROOT / args.batch
    rows = list(csv.DictReader((batch_dir / "manifest.csv").open()))
    if args.limit:
        rows = rows[: args.limit]

    start = time.perf_counter()
    failures = 0
    for i, row in enumerate(rows, 1):
        try:
            post_image(args.url, batch_dir / "images" / row["filename"])
        except Exception as exc:  # noqa: BLE001 - report and keep going
            failures += 1
            if failures <= 3:
                print(f"  ! {row['filename']}: {exc}")
        if i % 100 == 0:
            print(f"  {i}/{len(rows)} sent")

    elapsed = time.perf_counter() - start
    print(f"replayed {len(rows) - failures}/{len(rows)} tiles from {args.batch} "
          f"in {elapsed:.1f}s ({len(rows) / elapsed:.1f} req/s), {failures} failures")


if __name__ == "__main__":
    main()
