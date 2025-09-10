import glob
import json
import os

from model_eval import utils as eval_utils
from model_eval.checker import *

from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Alignment


# Calculate weighted accuracy
def generate_result_csv(leaderboard_table, output_path):
    data_close = []
    data_open = []
    for model_name, value in leaderboard_table.items():
        unusal_lose = value.get(
            "data_special_incomplete", {"accuracy": 0, "total_count": 0}
        )
        unusal_error = value.get(
            "data_special_error_param", {"accuracy": 0, "total_count": 0}
        )
        unusal_exceeding = value.get(
            "data_special_irrelevant", {"accuracy": 0, "total_count": 0}
        )

        atom_bool = value.get(
            "data_normal_atom_bool", {"accuracy": 0, "total_count": 0}
        )
        atom_enum = value.get(
            "data_normal_atom_enum", {"accuracy": 0, "total_count": 0}
        )
        atom_number = value.get(
            "data_normal_atom_number", {"accuracy": 0, "total_count": 0}
        )  # updated
        atom_list = value.get(
            "data_normal_atom_list", {"accuracy": 0, "total_count": 0}
        )  # updated
        atom_object_deep = value.get(
            "data_normal_atom_object_deep", {"accuracy": 0, "total_count": 0}
        )  # updated
        atom_object_short = value.get(
            "data_normal_atom_object_short", {"accuracy": 0, "total_count": 0}
        )  # updated

        normal_ss = value.get(
            "data_normal_single_turn_single_function", {"accuracy": 0, "total_count": 0}
        )  # updated
        normal_sp = value.get(
            "data_normal_single_turn_parallel_function",
            {"accuracy": 0, "total_count": 0},
        )  # updated
        normal_ms = value.get(
            "data_normal_multi_turn_user_switch", {"accuracy": 0, "total_count": 0}
        )  # updated
        normal_ma = value.get(
            "data_normal_multi_turn_user_adjust", {"accuracy": 0, "total_count": 0}
        )  # updated
        normal_similar = value.get(
            "data_normal_similar_api", {"accuracy": 0, "total_count": 0}
        )  # updated
        normal_profile = value.get(
            "data_normal_preference", {"accuracy": 0, "total_count": 0}
        )  # updated

        agent_turn = value.get(
            "data_agent_multi_turn",
            {"accuracy": 0, "process_accuracy": 0, "total_count": 0},
        )
        agent_step = value.get(
            "data_agent_multi_step",
            {"accuracy": 0, "process_accuracy": 0, "total_count": 0},
        )

        special_total = calculate_unweighted_accuracy(
            [unusal_lose, unusal_error, unusal_exceeding]
        )

        normal_total = calculate_unweighted_accuracy(
            [
                normal_ss,
                normal_sp,
                normal_ms,
                normal_ma,
                normal_similar,
                normal_profile,
                atom_bool,
                atom_enum,
                atom_number,
                atom_list,
                atom_object_deep,
                atom_object_short,
            ]
        )

        agent_total = calculate_unweighted_accuracy([agent_turn, agent_step])

        atom_total = calculate_unweighted_accuracy(
            [
                atom_bool,
                atom_enum,
                atom_number,
                atom_list,
                atom_object_deep,
                atom_object_short,
            ]
        )

        singal_turn_total = calculate_unweighted_accuracy([normal_ss, normal_sp])

        multi_turn_total = calculate_unweighted_accuracy([normal_ms, normal_ma])

        summary = (
            special_total["accuracy"] * 0.2676
            + normal_total["accuracy"] * 0.578
            + agent_total["accuracy"] * 0.1545
        )
        summary = round(summary, 3)

        if model_name in closed_model_list:
            data_close.append(
                [
                    model_name,
                    atom_bool["accuracy"],
                    atom_enum["accuracy"],
                    atom_number["accuracy"],
                    atom_list["accuracy"],
                    atom_object_deep["accuracy"],
                    atom_object_short["accuracy"],
                    atom_total["accuracy"],
                    normal_ss["accuracy"],
                    normal_sp["accuracy"],
                    singal_turn_total["accuracy"],
                    normal_ms["accuracy"],
                    normal_ma["accuracy"],
                    multi_turn_total["accuracy"],
                    normal_similar["accuracy"],
                    normal_profile["accuracy"],
                    normal_total["accuracy"],
                    unusal_lose["accuracy"],
                    unusal_error["accuracy"],
                    unusal_exceeding["accuracy"],
                    special_total["accuracy"],
                    agent_turn["accuracy"],
                    agent_turn["process_accuracy"],
                    agent_step["accuracy"],
                    agent_step["process_accuracy"],
                    agent_total["accuracy"],
                    summary,
                ]
            )
        else:
            data_open.append(
                [
                    model_name,
                    atom_bool["accuracy"],
                    atom_enum["accuracy"],
                    atom_number["accuracy"],
                    atom_list["accuracy"],
                    atom_object_deep["accuracy"],
                    atom_object_short["accuracy"],
                    atom_total["accuracy"],
                    normal_ss["accuracy"],
                    normal_sp["accuracy"],
                    singal_turn_total["accuracy"],
                    normal_ms["accuracy"],
                    normal_ma["accuracy"],
                    multi_turn_total["accuracy"],
                    normal_similar["accuracy"],
                    normal_profile["accuracy"],
                    normal_total["accuracy"],
                    unusal_lose["accuracy"],
                    unusal_error["accuracy"],
                    unusal_exceeding["accuracy"],
                    special_total["accuracy"],
                    agent_turn["accuracy"],
                    agent_turn["process_accuracy"],
                    agent_step["accuracy"],
                    agent_step["process_accuracy"],
                    agent_total["accuracy"],
                    summary,
                ]
            )

    # Sort data_close by summary in descending order
    sorted_data_close = sorted(data_close, key=lambda x: x[-1], reverse=True)

    # Sort data_open by summary in descending order
    sorted_data_open = sorted(data_open, key=lambda x: x[-1], reverse=True)

    # Merge sorted data_close and data_open
    data = sorted_data_close + sorted_data_open

    wb = Workbook()
    ws = wb.active

    data.insert(0, COLUMNS)

    for i, row in enumerate(data):
        for j, value in enumerate(row):
            cell = ws.cell(row=i + 1, column=j + 1, value=value)
            cell.alignment = Alignment(horizontal="center", vertical="center")

    filepath = os.path.join(output_path, "result.xlsx")
    wb.save(filepath)


def convert_result_to_excel(model_name, category, paths: Path, language):
    """Convert result for the given model and category into ../result_excel/."""
    input_path = paths["INPUT_PATH"]
    prompt_path = paths["PROMPT_PATH"]
    possible_answer_path = paths["POSSIBLE_ANSWER_PATH"]
    score_path = paths["OUTPUT_PATH"]

    prompt_file = eval_utils.build_data_path(prompt_path, category)
    answer_file = eval_utils.build_data_path(possible_answer_path, category)
    result_file = eval_utils.build_result_path(
        input_path, model_name, category, "_result.json"
    )
    score_file = eval_utils.build_result_path(
        score_path, model_name, category, "_score.json"
    )

    prompt_list = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            prompt_id = data["id"]
            if "time" not in list(data.keys()):
                time = ""
            else:
                time = data["time"]
            if "profile" not in list(data.keys()):
                profile = ""
            else:
                profile = data["profile"]
            functions = data["function"]
            question = data["question"]
            function_prompt = ""
            for function in functions:
                function_prompt = function_prompt + str(function) + "\n"

            system_prompt = ""
            user_prompt = ""
            if language == "zh":
                if "special" in category:
                    system_prompt = prompt_zh.SYSTEM_PROMPT_FOR_SPECIAL_DATA_ZH.format(
                        time=time, function=functions
                    )
                elif "preference" in category:
                    system_prompt = (
                        prompt_zh.SYSTEM_PROMPT_FOR_PREFERENCE_DATA_ZH.format(
                            profile=profile, function=functions
                        )
                    )
                else:
                    system_prompt = prompt_zh.SYSTEM_PROMPT_FOR_NORMAL_DATA_ZH.format(
                        time=time, function=functions
                    )
                user_prompt = prompt_zh.USER_PROMPT_ZH.format(question=question)
            elif language == "en":
                if "special" in category:
                    system_prompt = prompt_en.SYSTEM_PROMPT_FOR_SPECIAL_DATA_EN.format(
                        time=time, function=functions
                    )
                elif "preference" in category:
                    system_prompt = (
                        prompt_en.SYSTEM_PROMPT_FOR_PREFERENCE_DATA_EN.format(
                            profile=profile, function=functions
                        )
                    )
                else:
                    system_prompt = prompt_en.SYSTEM_PROMPT_FOR_NORMAL_DATA_EN.format(
                        time=time, function=functions
                    )
                user_prompt = prompt_en.USER_PROMPT_EN.format(question=question)

            prompt = system_prompt + "\n" + user_prompt
            prompt_list.append(
                {"id": prompt_id, "prompt": prompt, "question": question}
            )

    with open(answer_file, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            answer = json.loads(line)["ground_truth"]
            if "special" in category:
                continue
            if isinstance(answer, list):
                answer_list = []
                for answer_item in answer:
                    answer_list.append(convert_answer(answer_item))
                prompt_list[index]["expected_answer"] = answer_list
            else:
                answer = convert_answer(answer)
                prompt_list[index]["expected_answer"] = answer

    with open(result_file, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            result = json.loads(line)["result"]
            prompt_list[index]["model_answer"] = result
            prompt_list[index]["flag"] = "true"
            prompt_list[index]["error_reason"] = ""

    with open(score_file, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index < 1:
                continue

            data = json.loads(line)
            prompt_list[index - 1]["flag"] = "false"
            prompt_list[index - 1]["error_reason"] = data["error"]

    df = pd.DataFrame(prompt_list)

    folder_path = Path("..", "result_excel", language, model_name)
    # if language == "zh":
    #     folder_path = Path("../result_excel/zh/") / model_name
    # elif language == "en":
    #     folder_path = Path("../result_excel/en/") / model_name
    save_path = folder_path / f"data_{category}.xlsx"
    # Create folder if it doesn't exist
    folder_path.mkdir(parents=True, exist_ok=True)
    df.to_excel(save_path, index=False)


def save_eval_results(model_name, test_category, paths, language, eval_result) -> None:
    output_filename = "data_" + test_category + "_score.json"
    output_file_dir = paths["OUTPUT_PATH"] / model_name
    eval_utils.save_score_as_json(output_filename, eval_result, output_file_dir)
    convert_result_to_excel(model_name, test_category, paths, language)


"""


    output_filename = "data_" + test_category + "_score.json"
    output_file_dir = paths["OUTPUT_PATH"] / model_name
    eval_utils.save_score_as_json(output_filename, eval_result, output_file_dir)
    eval_helper.convert_result_to_excel(model_name, test_category, paths, language)
    return accuracy


    output_file_name = "data_" + test_category + "_score.json"
    output_file_dir = paths["OUTPUT_PATH"] / model_name
    eval_utils.save_score_as_json(output_filename, eval_result, output_file_dir)
    eval_helper.convert_result_to_excel(model_name, test_category, paths, language)
    return end_accuracy


    output_file_name = "data_" + test_category + "_score.json"
    output_file_dir = paths["OUTPUT_PATH"] / model_name
    eval_utils.save_score_as_json(output_filename, wrong_list, output_file_dir)
    eval_helper.convert_result_to_excel(model_name, category, paths, language)
    return accuracy


    output_file_name = "data_" + test_category + "_score.json"
    output_file_dir = paths["OUTPUT_PATH"] / model_name
    eval_utils.save_score_as_json(output_filename, result, output_file_dir)
    # TODO(sabina): missing export to excel.
    return accuracy, process_accuracy

"""
