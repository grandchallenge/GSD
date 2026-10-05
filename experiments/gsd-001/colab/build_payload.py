#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GSD_ROOT = ROOT / "experiments" / "gsd-001"

EXPECTED_GDSUITE_BLOBS = {
    "README.md": "e6a336fb540d6b47cd5cc34f495eafa8fd53a115",
    "config.yaml": "4b91818725ba025e95f4bf73397cdc8e76e5bf59",
    "run_eval.py": "cf73f27027654e5c76edb038f743fbeed5e801e8",
}

EXCLUDED_PARTS = {
    "__pycache__", ".pytest_cache", "runs", ".cache", ".venv", "receipts"
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".tar.gz"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def verify_gdsuite(upstream: Path) -> None:
    for name, expected in EXPECTED_GDSUITE_BLOBS.items():
        path = upstream / name
        if not path.is_file():
            raise RuntimeError(f"missing locked GDsuite file: {path}")
        actual = git_blob_sha1(path)
        if actual != expected:
            raise RuntimeError(f"GDsuite blob mismatch {name}: {actual} != {expected}")


def atlas_files() -> list[tuple[Path, str]]:
    rows = []
    for path in sorted(GSD_ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in rel.parts):
            continue
        if any(path.name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES):
            continue
        rows.append((path, rel.as_posix()))
    return rows


def payload_files(upstream: Path) -> list[tuple[Path, str]]:
    rows = atlas_files()
    for name in sorted(EXPECTED_GDSUITE_BLOBS):
        rows.append((upstream / name, f"external/GDsuite/{name}"))
    return rows


def tar_bytes(rows: list[tuple[Path, str]]) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for path, arcname in rows:
            data = path.read_bytes()
            info = tarfile.TarInfo(name=arcname)
            info.size = len(data)
            info.mode = 0o755 if os.access(path, os.X_OK) else 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", mtime=0, filename="") as gz:
        gz.write(raw.getvalue())
    return compressed.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    upstream = args.upstream_dir.resolve()
    verify_gdsuite(upstream)
    rows = payload_files(upstream)
    payload = tar_bytes(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)

    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    manifest = {
        "schema_version": 1,
        "kind": "GSD_COLAB_SOURCE_PAYLOAD",
        "source_commit": source_commit,
        "payload_sha256": sha256_bytes(payload),
        "file_count": len(rows),
        "files": [
            {
                "path": arcname,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path, arcname in rows
        ],
        "gdsuite_git_blobs": EXPECTED_GDSUITE_BLOBS,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[GSD] payload={args.output} sha256={manifest['payload_sha256']} "
        f"files={manifest['file_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
