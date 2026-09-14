from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path


FILES = {
    "connections_biological.csv.gz": (
        "https://zenodo.org/api/records/21549559/files/connections_biological.csv.gz/content",
        "2af9b8db6a8dbbc02cd3cc36f2631c19",
    ),
    "neuron_annotations_flywire_v2.1.0.tsv.gz": (
        "https://zenodo.org/api/records/21549559/files/neuron_annotations_flywire_v2.1.0.tsv.gz/content",
        "55789fa30a854bd00451af16441a7d3d",
    ),
}


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_data(data_dir: str | Path, *, force: bool = False) -> list[Path]:
    destination = Path(data_dir)
    destination.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for filename, (url, expected_md5) in FILES.items():
        target = destination / filename
        if target.exists() and _md5(target) == expected_md5 and not force:
            print(f"verified {target}", file=sys.stderr)
            downloaded.append(target)
            continue
        staged = target.with_suffix(target.suffix + ".part")
        print(f"downloading {url} -> {target}", file=sys.stderr)
        with urllib.request.urlopen(url) as response, staged.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        actual_md5 = _md5(staged)
        if actual_md5 != expected_md5:
            staged.unlink(missing_ok=True)
            raise ValueError(f"Checksum mismatch for {filename}: {actual_md5} != {expected_md5}")
        staged.replace(target)
        downloaded.append(target)
    return downloaded
