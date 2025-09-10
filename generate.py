"""Entrypoint for ACEBench."""

import json
import os
from concurrent import futures
from pathlib import Path
from typing import Mapping, Any, Iterable
import logging


from absl import flags, app
from tqdm import tqdm
from environment import files as files_lib


import category as category_lib
from model_inference.inference_map import inference_map


logger = logging.getLogger(__name__)

RESULT_PATHS = {
    "zh": Path("result_all/result_zh/"),
    "en": Path("result_all/result_en/"),
}
DATA_PATHS = {"zh": Path("data_all/data_zh/"), "en": Path("data_all/data_en/")}


FLAGS = flags.FLAGS

flags.DEFINE_list("model", ["gpt-4o"], "Name of the model(s) to use.")
flags.DEFINE_string("model_path", None, "Path to the model for local models.")
flags.DEFINE_string("user_model", "gpt-4o", "Model used by the user role in the agent.")
flags.DEFINE_list("category", ["test_all"], "Category of the model you want to test.")
flags.DEFINE_float(
    "temperature", 0.7, "Temperature parameter to control randomness of model output."
)
flags.DEFINE_float(
    "top_p", 1.0, "Top-p parameter to control diversity of model output."
)
flags.DEFINE_integer("max_tokens", 1200, "Maximum number of tokens to generate.")
flags.DEFINE_integer("num_gpus", 1, "Number of GPUs to use.")
flags.DEFINE_float("gpu_memory_utilization", 0.9, "GPU memory utilization rate.")
flags.DEFINE_string(
    "language",
    "en",
    "Language for model output, choose 'en' for English or 'zh' for Chinese.",
)
flags.DEFINE_integer("num_threads", 5, "Number of threads to use for concurrency.")
flags.DEFINE_integer(
    "max_dialog_turns",
    40,
    "Maximum number of dialog turns allowed for agent interactions.",
)
flags.register_validator(
    "language",
    (lambda lang: lang in DATA_PATHS),
    message="Language must be 'en' or 'zh'.",
)


def sort_json(file):
    """Sort json."""
    data = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    if "multi_turn" in file and "agent" not in file:
        data = sorted(data, key=lambda x: tuple(map(int, x["id"].split("_")[-2:])))
    else:
        data = sorted(data, key=lambda x: int(x["id"].split("_")[-1]))
    with open(file, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")


def generate_signal(model_name: str, test_case: dict[str, object]):
    """Generate Signal."""
    model_inference = inference_map[model_name](
        model_name,
        FLAGS.model_path,
        FLAGS.temperature,
        FLAGS.top_p,
        FLAGS.max_tokens,
        FLAGS.max_dialog_turns,
        FLAGS.user_model,
        FLAGS.language,
    )
    result_path = RESULT_PATHS[FLAGS.language]

    test_name: str = test_case["id"]
    question = test_case["question"]
    functions = test_case["function"]
    time = test_case.get("time", "")
    profile = test_case.get("profile", "")
    if isinstance(functions, (dict, str)):
        functions = [functions]

    if "agent" in test_name:
        result, process = model_inference.inference(
            question, functions, time, profile, test_case, test_name
        )
        result_to_write = {"id": test_name, "result": result, "process": process}
    else:
        result = model_inference.inference(
            question, functions, time, profile, test_case, test_name
        )
        result_to_write = {"id": test_name, "result": result}

    print("Done inference. writing result to: ", result_path)

    model_inference.write_result(result_to_write, model_name, result_path)


def generate_results(
    model_name: str, test_cases: list[Mapping[str, Any]], completed_ids: set[str]
) -> None:
    """Runs inference for the given 'model_name' and write to corresponding result file."""

    with futures.ThreadPoolExecutor(max_workers=FLAGS.num_threads) as executor:
        futures_list = [
            executor.submit(generate_signal, model_name, test_case)
            for test_case in test_cases
            if test_case["id"] not in completed_ids
        ]

        with tqdm(total=len(futures_list), desc="Processing Tasks", leave=True) as pbar:
            for future in futures.as_completed(futures_list):
                try:
                    future.result()  # Catch exceptions in tasks
                    pbar.update(1)
                except Exception as e:
                    logging.exception("Task raised an exception: %s", e)
                    # You can choose whether to continue executing tasks after catching an exception, or to terminate the program
                    raise


def get_result_filepath(folder_path: str, test_name: str):
    """Simple formatting helper for result file name for the given test."""
    filename = f"data_{test_name}_result.json"
    return os.path.join(folder_path, filename)


def collect_completed_test_ids(
    test_names: Iterable[str], result_path: Path
) -> set[str]:
    """Gather list of test IDs that has already been generated."""
    completed_ids = set()
    for test_name in test_names:
        file_path = result_path / f"data_{test_name}_result.json"
        if not file_path.exists():
            continue
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                completed_ids.add(json.loads(line)["id"])

    return completed_ids


def main(argv: list[str]):
    """Main."""
    del argv  # unused although it's needed for app.run

    data_path: Path = DATA_PATHS[FLAGS.language]
    result_path: Path = RESULT_PATHS[FLAGS.language]
    test_names = {
        test_name
        for category in FLAGS.category
        for test_name in category_lib.ACE_DATA_CATEGORY[category]
    }

    for model_name in FLAGS.model:
        model_path = result_path / model_name
        model_path.mkdir(parents=True, exist_ok=True)

        completed_ids = collect_completed_test_ids(test_names, model_path)

        test_files = [data_path / f"data_{test_name}.json" for test_name in test_names]
        test_cases_total: list[str] = []
        for test_file in test_files:
            test_cases_total.append(files_lib.load_test_cases(test_file))

        if len(test_cases_total) > 0:
            generate_results(model_name, test_cases_total, completed_ids)

        # Multithreading may disrupt the order of execution, so the result ids need to be reordered
        for test_name in test_names:
            sort_json(get_result_filepath(model_path, test_name))


if __name__ == "__main__":
    app.run(main)
