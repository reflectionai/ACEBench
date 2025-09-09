"""Libraries for files."""

from typing import Any, TypedDict
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


# class TestCase(TypedDict):
#     """Structure of a test case in SWEBench."""


def load_test_cases(test_file_names: list[Path]) -> dict[str, object]:
    """Load test cases."""
    cases: dict[str, object] = {}
    for file_path in test_file_names:
        try:
            with file_path.open(encoding="utf-8") as file:
                cases.extend(json.loads(line) for line in file)
        except FileNotFoundError:
            logger.error("Missing test file: %s", file_path)
        except json.JSONDecodeError as err:
            logger.error("Malformed JSON in %s: %s", file_path, err)
    return cases


"""AI's suggestion:

from __future__ import annotations
from typing import TypedDict, List, Dict, Union, Optional, Any
from typing_extensions import NotRequired   # Python ≥3.11 or `typing` back-port

# --------------------------------------------------------------------
# 1.  Recursive JSON helper types (replace Any)
JSONPrimitive = Union[str, int, float, bool, None]
JSONValue     = Union[JSONPrimitive, List["JSONValue"], Dict[str, "JSONValue"]]

# --------------------------------------------------------------------
# 2.  Ground-truth value (only for “possible_answer” test cases)
PosArgs = List[str]              # e.g. ["latitude", "longitude"]
KwArgs  = Dict[str, JSONValue]   # e.g. {"startDate": "2024-01-01", ...}
GroundTruth = Union[
    str,                         # special-irrelevant cases
    Dict[str, Union[PosArgs, KwArgs]],
]

# --------------------------------------------------------------------
# 3.  Unified test-case record
class TestCase(TypedDict, total=False):   # all keys optional except id
    id: str                               # always present

    # For evaluation-by-answer files
    ground_truth: NotRequired[GroundTruth]

    # For agent-execution files
    question:      NotRequired[str]
    initial_config: NotRequired[Dict[str, JSONValue]]
    path:           NotRequired[List[JSONValue]]
    function:       NotRequired[List[Dict[str, JSONValue]]]

    # Any future extension fields survive as `JSONValue`
    # (keeps the loader forward-compatible)
    extra: NotRequired[Dict[str, JSONValue]]
"""
