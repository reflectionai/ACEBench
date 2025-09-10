import glob
import json
import os

from model_eval import utils as eval_utils
from model_eval.checker import *

REST_API_GROUND_TRUTH_FILE_PATH = "api_status_check_ground_truth_REST.json"
EXECTUABLE_API_GROUND_TRUTH_FILE_PATH = "api_status_check_ground_truth_executable.json"

COLUMNS = [
    "Model",
    "bool",
    "enum",
    "number",
    "list",
    "object_short",
    "object_deep",
    "atom",
    "single_turn_singal_function",
    "single_turn_parallel_function",
    "single_turn",
    "multiple_turn_switch",
    "multiple_turn_adjust",
    "multiple_turn",
    "similar_api",
    "profile",
    "normal_summary",
    "incomplete",
    "error",
    "irrelevant",
    "special_summary",
    "agent_multi_turn",
    "agent_multi_turn_process",
    "agent_multi_step",
    "agent_multi_step_process",
    "agent_summary",
    "Summary",
]

closed_model_list = [
    "o1-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-11-20",
    "gpt-4-turbo-2024-04-09",
    "qwen-max",
    "doubao-pro-32k",
    "claude-3-5-sonnet-20241022",
    "gemini-1.5-pro",
    "deepseek-chat",
]

V100_x8_PRICE_PER_HOUR = 22.032


def extract_after_test(input_string):
    parts = input_string.split("data_")[1].split("_result")[0].split(".json")[0]
    return parts


def find_file_with_suffix(folder_path, suffix):
    json_files_pattern = os.path.join(folder_path, "*.json")
    for json_file in glob.glob(json_files_pattern):
        if suffix == "multi_turn":
            json_file = folder_path + "data_multi_turn.json"
            return json_file
        if suffix in json_file.split("/")[-1]:
            return json_file


def load_file(file_path):
    result = []
    with open(file_path, encoding="utf-8") as f:
        file = f.readlines()
        for line in file:
            result.append(json.loads(line))
    return result


def is_empty_output(decoded_output):
    if not is_function_call_format_valid(decoded_output):
        return True
    if len(decoded_output) == 0:
        return True
    if len(decoded_output) == 1 and len(decoded_output[0]) == 0:
        return True


def multiplt_turn_accuracy(score_list):
    end_score_list = []
    process_score_list = []
    for score in score_list:
        if False in score["valid"]:
            end_score = 0
        else:
            end_score = 1
        process_score = score["valid"].count(True) / len(score["valid"])
        process_score = round(process_score, 3)

        end_score_list.append(end_score)
        process_score_list.append(process_score)
    end_score_total = round(sum(end_score_list) / len(end_score_list), 3)
    process_score_total = round(sum(process_score_list) / len(process_score_list), 3)
    return end_score_total, process_score_total


def calculate_weighted_accuracy(accuracy_dict_list):
    total_count = 0
    total_accuracy = 0
    for accuracy_dict in accuracy_dict_list:
        total_count += accuracy_dict["total_count"]
        total_accuracy += accuracy_dict["accuracy"] * accuracy_dict["total_count"]

    if total_count == 0:
        return {"accuracy": 0, "total_count": 0}

    return {
        "accuracy": round(total_accuracy / total_count, 3),
        "total_count": total_count,
    }


def calculate_unweighted_accuracy(accuracy_dict_list):
    total_accuracy = 0
    for accuracy_dict in accuracy_dict_list:
        total_accuracy += accuracy_dict["accuracy"]

    if len(accuracy_dict_list) == 0:
        return {"accuracy": 0, "total_count": 0}

    return {
        "accuracy": round(total_accuracy / len(accuracy_dict_list), 3),
        "total_count": 0,
    }


def update_result_table_with_score_file(leaderboard_table, score_path):
    entries = os.scandir(score_path)

    # Filter out the subdirectories
    subdirs = [entry.path for entry in entries if entry.is_dir()]

    # Traverse each subdirectory
    for subdir in subdirs:
        # Pattern to match JSON files in this subdirectory
        json_files_pattern = os.path.join(subdir, "*.json")
        model_name = os.path.basename(subdir)
        # Find and process all JSON files in the subdirectory
        for model_score_json in glob.glob(json_files_pattern):
            if "process" not in model_score_json:
                if "agent" not in model_score_json:
                    metadata = load_file(model_score_json)[0]
                    accuracy, total_count = (
                        metadata["accuracy"],
                        metadata["total_count"],
                    )
                    test_category = model_score_json.split("_score.json")[0].split("/")[
                        -1
                    ]
                    test_category = test_category.split("\\")[-1]
                    if model_name not in leaderboard_table:
                        leaderboard_table[model_name] = {}
                    if test_category not in leaderboard_table[model_name]:
                        leaderboard_table[model_name][test_category] = {
                            "accuracy": accuracy,
                            "total_count": total_count,
                        }
                else:
                    metadata = load_file(model_score_json)[0]
                    accuracy, process_accuracy, total_count = (
                        metadata["end_to_end_accuracy"],
                        metadata["process_accuracy"],
                        metadata["total_count"],
                    )
                    test_category = model_score_json.split("_score.json")[0].split("/")[
                        -1
                    ]
                    test_category = test_category.split("\\")[-1]
                    if model_name not in leaderboard_table:
                        leaderboard_table[model_name] = {}
                    if test_category not in leaderboard_table[model_name]:
                        leaderboard_table[model_name][test_category] = {
                            "accuracy": accuracy,
                            "process_accuracy": process_accuracy,
                            "total_count": total_count,
                        }


def collapse_json_objects(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    objects = []
    depth = 0
    obj_start = 0
    for i, char in enumerate(content):
        if char == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                obj = content[obj_start : i + 1]
                objects.append(obj)

    with open(file_path, "w") as out_file:
        for obj in objects:
            json_obj = json.loads(obj)
            compact_json = json.dumps(json_obj, separators=(",", ":"))
            out_file.write(compact_json + "\n")


def convert_answer(answer):
    if answer == "":
        return answer
    result = [
        f"{key}({', '.join([f'{k}={v}' if isinstance(v, str) else f'{k}={v}' for k, v in value.items()])})"
        for key, value in answer.items()
    ]
    return result


def merge_result(folder_path):
    # Get all Excel files in the folder
    excel_files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]

    # List to store data from all files
    all_data = []

    # Read each Excel file and add its data to all_data
    for file in excel_files:
        if "special" not in file and "similar" not in file:
            file_path = os.path.join(folder_path, file)
            df = pd.read_excel(file_path)  # Read Excel file
            all_data.append(df)

    for file in excel_files:
        if "special" in file or "similar" in file:
            file_path = os.path.join(folder_path, file)
            df = pd.read_excel(file_path)  # Read Excel file
            all_data.append(df)

    # Merge all DataFrames
    merged_data = pd.concat(all_data, ignore_index=True)
    model_name = (str(folder_path)).split("/")[-1]
    save_name = model_name + "_output.xlsx"

    # Write the merged data to a new Excel file
    merged_data.to_excel(os.path.join(folder_path, save_name), index=False)
