#!/usr/bin/env python3
"""Zeigt die ersten N Zeilen eines lokal gespeicherten t2-ragbench-Splits (load_from_disk)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_from_disk

DEFAULT_REL_PATH = "data/t2-ragbench/ConvFinQA"


def main() -> None:
    parser = argparse.ArgumentParser(description="Erste N Einträge von lokalem HF-Dataset ausgeben.")
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help=f"Ordner mit dataset_dict.json (Standard: Testframework/{DEFAULT_REL_PATH}).",
    )
    parser.add_argument(
        "--split",
        default="turn_0",
        help="Split-Name (Standard: turn_0).",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=10,
        help="Anzahl Zeilen (Standard: 10).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=500,
        help="Strings länger als N werden gekürzt (0 = keine Kürzung).",
    )
    args = parser.parse_args()

    framework_root = Path(__file__).resolve().parent.parent
    base: Path = args.path if args.path is not None else framework_root / DEFAULT_REL_PATH

    if not (base / "dataset_dict.json").is_file():
        raise SystemExit(f"Kein Dataset unter {base} (dataset_dict.json fehlt).")

    ds = load_from_disk(str(base))
    if args.split not in ds:
        raise SystemExit(f"Split '{args.split}' nicht vorhanden. Verfügbar: {list(ds.keys())}")

    n = min(args.num, len(ds[args.split]))
    rows = ds[args.split].select(range(n))
    max_c = args.max_chars

    def shorten(obj: object) -> object:
        if max_c <= 0:
            return obj
        if isinstance(obj, str) and len(obj) > max_c:
            return obj[:max_c] + f"… ({len(obj)} Zeichen)"
        if isinstance(obj, list):
            return [shorten(x) for x in obj]
        if isinstance(obj, dict):
            return {k: shorten(v) for k, v in obj.items()}
        return obj

    for i, row in enumerate(rows):
        print(f"\n--- [{i}] ---")
        out = shorten(dict(row)) if max_c > 0 else dict(row)
        print(json.dumps(out, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
