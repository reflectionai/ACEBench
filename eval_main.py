"""Eval Main."""

import sys
import os
import json
from pathlib import Path

from absl import flags, app
from model_eval import checker as eval_checker
from model_eval import utils as eval_utils
from model_eval import evaluation_helper as eval_helper
from model_eval import export as eval_export
from category import ACE_DATA_CATEGORY
from model_inference.utils import decode_ast
from environment.evaluation import eval as eval_lib


sys.path.append("../")
RESULT_TABLE = {}


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "language",
    "en",
    "Language for model output, choose 'en' for English or 'zh' for Chinese.",
)
flags.DEFINE_list("model", ["gpt-4o"], "A list of model names to evaluate.")
flags.DEFINE_list(
    "category", ["test_all"], "A list of test categories to run the evaluation on."
)


def runner(model_names, categories, paths, language):
    """Main runner function."""
    for model_name in model_names:
        for category in categories:
            print(f"🔍 Running test: {category}")
            # str legacy helper.
            model_result_path = eval_utils.build_result_path(
                paths["INPUT_PATH"], model_name, category, "_result.json"
            )
            model_results = eval_helper.load_file(model_result_path)
            eval_results = eval_lib.run_eval(
                model_name, category, model_results, paths, language
            )

            eval_export.save_eval_results(
                model_name, category, paths, language, eval_results
            )
    eval_helper.update_result_table_with_score_file(
        RESULT_TABLE, str(paths["OUTPUT_PATH"])
    )
    eval_helper.generate_result_csv(RESULT_TABLE, str(paths["OUTPUT_PATH"]))


def main(argv: list[str]):
    """Main."""
    del argv  # unused although it's needed for app.run

    paths = eval_lib.get_paths(FLAGS.language)
    test_categories = [
        category
        for test_category in (FLAGS.category or [])
        for category in (ACE_DATA_CATEGORY.get(test_category, [test_category]))
    ]

    # Extract and normalize model names
    model_names = [model_name.replace("/", "_") for model_name in (FLAGS.model or [])]

    # Call the main function
    runner(model_names, test_categories, paths, FLAGS.language)

    print(f"Models being evaluated: {model_names}")
    print(f"Test categories being used: {test_categories}")


if __name__ == "__main__":
    app.run(main)
