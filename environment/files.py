"""Libraries for files."""

from typing import Optional
import logging
from pathlib import Path
import json
import random

logger = logging.getLogger(__name__)


# class TestCase(TypedDict):
#     """Structure of a test case in SWEBench."""


def load_test_case(
    test_file_path: Path, test_number: Optional[int]
) -> dict[str, object]:
    """Load test cases."""
    case = ""
    try:
        all_test_cases_in_file: dict[int, dict[str, object]] = {}
        count = 0
        with test_file_path.open(encoding="utf-8") as file:
            for line in file:
                line_json = json.loads(line)
                if "id" not in line_json:
                    raise ValueError(
                        f"Malformated file {test_file_path}, as 'id' key is not"
                        f" present on line:\n'{line}'"
                    )
                if not isinstance(line_json, dict):
                    raise ValueError(
                        f"Malformated file {test_file_path}, contains a line "
                        f"that is not a dict:\n'{line}'"
                    )

                if test_number:
                    id_val: object = line_json["id"]
                    if not isinstance(id_val, str):
                        raise ValueError(
                            f"Malformated file {test_file_path}, as value of "
                            f"'id' key is not a string:\n'{line}'"
                        )
                    if id_val.endswith(f"_{test_number}"):
                        return line_json

                all_test_cases_in_file[count] = line_json
                count += 1

        if test_number:
            raise ValueError(
                f"Malformated file {test_file_path}, as there is no id value"
                f" that ends with _{test_number}"
            )

        random_number = 0
        if len(all_test_cases_in_file) > 0:
            random.randint(0, len(all_test_cases_in_file) - 1)
        return all_test_cases_in_file[random_number]

    except FileNotFoundError as err:
        logger.error("Missing test file of %s: %s", test_file_path, err)
        raise err
    except json.JSONDecodeError as err:
        logger.error("Malformed JSON in %s: %s", test_file_path, err)
        raise err

    return case


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
