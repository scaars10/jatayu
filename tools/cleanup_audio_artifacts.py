from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from storage import AudioArtifactStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete old audio artifacts from the managed spool directory.",
    )
    parser.add_argument(
        "--older-than-hours",
        type=float,
        default=24.0,
        help="Delete files older than this many hours. Defaults to 24.",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=None,
        help="Override the managed spool directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be deleted without removing them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = AudioArtifactStore(args.root_dir)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=args.older_than_hours)
    deleted = 0
    scanned = 0

    if not store.root_dir.exists():
        print(f"No spool directory found at {store.root_dir}")
        return

    for path in sorted(store.root_dir.glob("*")):
        if not path.is_file():
            continue

        scanned += 1
        modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if modified_at >= cutoff:
            continue

        print(f"delete {path}")
        if not args.dry_run:
            store.delete(path)
        deleted += 1

    print(f"scanned={scanned} deleted={deleted} root={store.root_dir}")


if __name__ == "__main__":
    main()
