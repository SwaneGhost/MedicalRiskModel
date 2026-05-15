"""Download the WiDS Datathon 2020 Kaggle competition data.

The script downloads into a stable project path instead of leaving the files
only in KaggleHub's cache. It skips the download when the target directory
already contains files unless --force is provided.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
from pathlib import Path


COMPETITION = "widsdatathon2020"
DEFAULT_OUTPUT_DIR = Path("data/raw/widsdatathon2020")


def directory_has_files(path: Path) -> bool:
    return path.exists() and any(item.is_file() for item in path.rglob("*"))


def copy_competition_files(download_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for source in download_path.iterdir():
        destination = output_dir / source.name
        if source.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def configure_kaggle_auth(
    kaggle_username: str | None = None,
    kaggle_key: str | None = None,
    kaggle_json: Path | None = None,
) -> None:
    """Set Kaggle credentials for KaggleHub when explicit credentials are given."""
    if kaggle_json is not None:
        kaggle_json = kaggle_json.expanduser().resolve()
        with kaggle_json.open(encoding="utf-8") as credentials_file:
            credentials = json.load(credentials_file)

        kaggle_username = kaggle_username or credentials.get("username")
        kaggle_key = kaggle_key or credentials.get("key")

    if kaggle_username:
        os.environ["KAGGLE_USERNAME"] = kaggle_username
    if kaggle_key:
        os.environ["KAGGLE_KEY"] = kaggle_key


def download_wids_data(
    output_dir: Path,
    force: bool = False,
    kaggle_username: str | None = None,
    kaggle_key: str | None = None,
    kaggle_json: Path | None = None,
) -> Path:
    output_dir = output_dir.expanduser().resolve()

    if directory_has_files(output_dir) and not force:
        print(f"Data already exists at: {output_dir}")
        print("Use --force to download again and overwrite copied files.")
        return output_dir

    configure_kaggle_auth(
        kaggle_username=kaggle_username,
        kaggle_key=kaggle_key,
        kaggle_json=kaggle_json,
    )

    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: kagglehub. Install project requirements first:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc

    os.environ.setdefault("KAGGLEHUB_CACHE", str(output_dir.parent / ".kagglehub_cache"))

    print(f"Downloading Kaggle competition: {COMPETITION}")
    downloaded_path = Path(kagglehub.competition_download(COMPETITION))
    print(f"KaggleHub cache path: {downloaded_path}")

    copy_competition_files(downloaded_path, output_dir)
    print(f"Competition files are available at: {output_dir}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to store competition files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again and overwrite files copied into the output directory.",
    )
    parser.add_argument(
        "--kaggle-username",
        default=os.getenv("KAGGLE_USERNAME"),
        help="Kaggle username. Defaults to the KAGGLE_USERNAME environment variable.",
    )
    parser.add_argument(
        "--kaggle-key",
        default=os.getenv("KAGGLE_KEY"),
        help="Kaggle API key. Defaults to the KAGGLE_KEY environment variable.",
    )
    parser.add_argument(
        "--kaggle-json",
        type=Path,
        help="Path to a kaggle.json file containing username and key fields.",
    )
    parser.add_argument(
        "--prompt-credentials",
        action="store_true",
        help="Prompt for missing Kaggle credentials without echoing the API key.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.prompt_credentials:
        args.kaggle_username = args.kaggle_username or input("Kaggle username: ").strip()
        args.kaggle_key = args.kaggle_key or getpass.getpass("Kaggle API key: ").strip()

    download_wids_data(
        args.output_dir,
        force=args.force,
        kaggle_username=args.kaggle_username,
        kaggle_key=args.kaggle_key,
        kaggle_json=args.kaggle_json,
    )
