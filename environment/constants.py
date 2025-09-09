"""File that contains constants."""

from functools import lru_cache
from pathlib import Path
from typing import Final, Set


@lru_cache
def get_test_case_names() -> Set[str]:
    """Get supported categories (e.g., test cases)"""

    directory = Path(DATA_DIRECTORY)
    if not directory.exists():
        raise FileNotFoundError(f"Directory '{DATA_DIRECTORY}/' does not exists.")
    if not directory.is_dir():
        raise FileNotFoundError(f"'{DATA_DIRECTORY}/' is not a directory.")

    categories = {f.stem for f in directory.glob("*.json") if f.is_file()}
    return categories


@lru_cache
def get_supported_categories() -> Set[str]:
    """Get supported categories (e.g., test cases)"""
    all_categories = TEST_CASE_NAMES
    # TODO(sabina): add groupings
    return all_categories


TEST_CASE_NAMES: Final[Set[str]] = get_test_case_names()
SUPPORTED_CATEGORIES: Final[Set[str]] = get_supported_categories()
DATA_DIRECTORY = Path("data_all/data_en")
