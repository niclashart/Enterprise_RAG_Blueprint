#!/usr/bin/env python3
"""Lädt das Hugging-Face-Dataset `G4KMU/t2-ragbench` nach `Testframework/data/`.

Liegt unter `scripts/`, damit `import datasets` die Hugging-Face-Bibliothek
bezieht und nicht das lokale Paket `Testframework/datasets`.

Subsets laufen als Konfigurationen: FinQA, ConvFinQA, TAT-DQA (Standard: alle).

Bei Rate-Limits hilft `HF_TOKEN` in der Umgebung;
https://huggingface.co/settings/tokens

Usage:
    cd Testframework
    pip install -r requirements.txt
    python scripts/download_t2_ragbench.py
    python scripts/download_t2_ragbench.py --config ConvFinQA
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset

DEFAULT_REPO_ID = "G4KMU/t2-ragbench"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="t2-ragbench von Hugging Face nach data/ speichern (save_to_disk)."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Dataset-Repo auf dem Hub (Standard: G4KMU/t2-ragbench).",
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        metavar="NAME",
        help=(
            "Nur diese Konfiguration laden (z. B. ConvFinQA). "
            "Mehrfach angeben für mehrere. Standard: alle Konfigurationen."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Ziel-Basisordner (Standard: Testframework/data).",
    )
    args = parser.parse_args()

    framework_root = Path(__file__).resolve().parent.parent
    data_dir: Path = args.data_dir if args.data_dir is not None else framework_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    repo_id = args.repo_id
    slug = repo_id.split("/")[-1]
    out_root = data_dir / slug

    configs = list(args.configs) if args.configs else get_dataset_config_names(repo_id)

    for config_name in configs:
        print(f"Lade {repo_id} (config={config_name}) …")
        ds = load_dataset(repo_id, config_name)
        target = out_root / config_name
        ds.save_to_disk(str(target))
        print(f"Gespeichert unter {target}")


if __name__ == "__main__":
    main()
