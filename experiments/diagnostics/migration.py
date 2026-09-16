"""Preview, apply, verify, or reverse a frozen file-by-file migration manifest."""

import argparse
import hashlib
import json
from pathlib import Path


def verify_file(path: Path, record: dict) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or non-regular artifact: {path}")
    if path.stat().st_size != record["size"] or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError(f"Artifact content changed: {path}")


def migrate(root: Path, records: list[dict], *, apply: bool = False, reverse: bool = False) -> dict:
    """Preflight the entire move before mutation; resume only verified moves.

    Two existing paths are a collision even if their bytes agree. Nothing is
    deleted or merged. Reversal uses the same hashes and collision checks.
    """
    pending = []
    destinations = set()
    root = root.resolve()
    for record in records:
        old, new = record["old"], record["new"]
        if reverse:
            old, new = new, old
        source, target = root / old, root / new
        for path in (source, target):
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"Path escapes migration root: {path}")
        if target in destinations:
            raise ValueError(f"Duplicate destination: {target}")
        destinations.add(target)
        if source == target:
            verify_file(source, record)
        elif source.exists():
            if target.exists():
                raise ValueError(f"Destination collision: {target}")
            verify_file(source, record)
            pending.append((source, target, record))
        else:
            verify_file(target, record)
    if apply:
        for source, target, record in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
            verify_file(target, record)
    return {"files": len(records), "pending": len(pending), "moved": len(pending) if apply else 0, "reverse": reverse}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            migrate(args.root, json.loads(args.manifest.read_text())["files"], apply=args.apply, reverse=args.reverse)
        )
    )


if __name__ == "__main__":
    main()
