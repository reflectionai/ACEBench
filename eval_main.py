"""Eval Main."""

import sys
import os
import json
from pathlib import Path

from absl import flags, app
from model_eval import checker as eval_checker
from model_eval import utils as eval_utils
from model_eval import evaluation_helper as eval_helper
from category import ACE_DATA_CATEGORY
from model_inference.utils import decode_ast


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


def get_paths(language: str):
    """Get Paths."""
    base_paths = {
        "zh": {
            "INPUT_PATH": Path("result_all/result_zh/"),
            "PROMPT_PATH": Path("data_all/data_zh/"),
            "POSSIBLE_ANSWER_PATH": Path("data_all/data_zh/possible_answer/"),
            "OUTPUT_PATH": Path("score_all/score_zh/"),
        },
        "en": {
            "INPUT_PATH": Path("result_all/result_en/"),
            "PROMPT_PATH": Path("data_all/data_en/"),
            "POSSIBLE_ANSWER_PATH": Path("data_all/data_en/possible_answer/"),
            "OUTPUT_PATH": Path("score_all/score_en/"),
        },
    }
    return base_paths[language]


def normal_single_turn_eval(
    model_results, prompts, possible_answers, test_category, model_name, paths
):
    """Normal Single Turn Eval."""
    if not all(len(x) == len(model_results) for x in [prompts, possible_answers]):
        raise ValueError(
            f"The length of the model result ({len(model_results)}) does not "
            "match the length of the prompt ({len(prompt)}) or possible answer"
            f" ({len(possible_answers)}). Please check the input files for "
            "completeness."
        )

    eval_result = []
    correct_count = 0
    for i, (model_result, prompt, possible_answer) in enumerate(
        zip(model_results, prompts, possible_answers)
    ):
        prompt_id = prompt["id"]
        question = prompt["question"]
        result = model_result["result"]
        prompt_item = prompt["function"]
        answer = possible_answer["ground_truth"]

        result_raw = result
        result_raw = "".join(result_raw.split())
        try:
            result = decode_ast(model_name, result_raw)
        except Exception as e:
            eval_result.append(
                {
                    "id": prompt_id,
                    "valid": False,
                    "error": [f"Invalid syntax. Failed to decode AST. {str(e)}"],
                    "error_type": "wrong_output_format",
                    "model_result_raw": result_raw,
                    "possible_answer": answer,
                }
            )
            continue
        decoder_output = eval_utils.is_function_call_format_valid(result)
        if not decoder_output:
            eval_result.append(
                {
                    "id": prompt_id,
                    "valid": False,
                    "error": [
                        "The output format does not meet the specified requirements."
                    ],
                    "error_type": "wrong_output_format",
                    "model_result_raw": str(result_raw),
                    "possible_answer": answer,
                }
            )
            continue

        if not isinstance(answer, list):
            answer = [answer]

        errors = []
        for answer_ in answer:
            check_result = eval_checker.normal_checker(
                prompt_item,
                result,
                answer_,
                question,
                test_category,
            )
            if check_result["valid"]:
                correct_count += 1
                break
            else:
                errors.append(
                    {
                        "error": check_result["error"],
                        "error_type": check_result["error_type"],
                    }
                )

        if errors:
            temp = {
                "id": prompt_id,
                "valid": False,
                "error": errors[0]["error"],
                "error_type": errors[0]["error_type"],
                "model_result": result_raw,
                "possible_answer": answer_,  # this sounds wrong?
            }
            eval_result.append(temp)

    accuracy = round((correct_count / len(model_results)), 3)
    eval_result.insert(
        0,
        {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(model_results),
        },
    )

    output_filename = "data_" + test_category + "_score.json"
    output_file_dir = paths["OUTPUT_PATH"] / model_name
    eval_utils.save_score_as_json(output_filename, eval_result, output_file_dir)
    eval_helper.convert_result_to_excel(
        model_name, test_category, paths, FLAGS.language
    )
    return accuracy


def normal_multi_turn_eval(
    model_results, prompts, possible_answers, test_category, model_name, paths
):
    """Normal Multi Turn"""
    if not all(len(x) == len(model_results) for x in [prompts, possible_answers]):
        raise ValueError(
            f"The length of the model result ({len(model_results)}) does not "
            f"match the length of the prompt ({len(prompts)}) or possible answer"
            f" ({len(possible_answers)}). Please check the input files for "
            "completeness."
        )

    eval_result = []
    correct_count = 0

    process_score_list = []
    score_list = []

    for i, (result, prmpt, answer) in enumerate(
        zip(model_results, prompts, possible_answers)
    ):
        result_id = result["id"]
        turn = prmpt["id"].split("_")[-2]
        model_result_id = result["id"].split("_")[-1]  # find a btter name..
        question = prmpt["question"]
        res = result["result"]
        prompt_function = prmpt["function"]
        answer_ground_truth = answer["ground_truth"]

        res_raw = res
        res_raw = "".join(res_raw.split())
        try:
            res = decode_ast(model_name, res_raw)
        except Exception as e:
            eval_result.append(
                {
                    "id": result_id,
                    "turn": turn,
                    "valid": False,
                    "error": [f"Invalid syntax. Failed to decode AST. {str(e)}"],
                    "error_type": "wrong_output_format",
                    "model_result": res_raw,
                    "possible_answer": answer_ground_truth,
                    "process": False,
                    "process_score": 0,
                }
            )
            process_score_list.append(0)
            if len(score_list) > 0 and turn == score_list[-1]["turn"]:
                score_list[-1]["valid"].append(False)
                score_list[-1]["number"] = model_result_id
            else:
                score_list.append(
                    {"turn": turn, "number": model_result_id, "valid": [False]}
                )
            continue

        # Check if the output format meets the requirements
        decoder_output = eval_utils.is_function_call_format_valid(res)
        if not decoder_output:
            eval_result.append(
                {
                    "id": result_id,
                    "turn": turn,
                    "valid": False,
                    "error": [
                        "The output format does not meet the specified requirements."
                    ],
                    "error_type": "wrong_output_format",
                    "model_result": str(res),
                    "possible_answer": answer_ground_truth,
                    "process": False,
                    "process_score": 0,
                }
            )
            process_score_list.append(0)
            if len(score_list) > 0 and turn == score_list[-1]["turn"]:
                score_list[-1]["valid"].append(False)
                score_list[-1]["number"] = model_result_id
            else:
                score_list.append(
                    {"turn": turn, "number": model_result_id, "valid": [False]}
                )
            continue

        if not isinstance(answer_ground_truth, list):
            answer_ground_truth = [answer_ground_truth]

        errors = []
        # Filter from multiple candidate answers
        for answ in answer_ground_truth:
            check_result = eval_checker.normal_checker(
                prompt_function,
                res,
                answ,
                question,
                test_category,
            )

            if check_result["valid"]:
                correct_count += 1
                process_score_list.append(1)
                break
            else:
                errors.append(
                    {
                        "error": check_result["error"],
                        "error_type": check_result["error_type"],
                    }
                )

        if not check_result["valid"]:
            temp = {
                "id": result_id,
                "turn": turn,
                "valid": False,
                "error": errors[0]["error"],
                "error_type": errors[0]["error_type"],
                "model_result": res_raw,
                "possible_answer": answer_ground_truth,
            }
            eval_result.append(temp)

        turn = result["id"].split("_")[-2]
        item = result["id"].split("_")[-1]
        if len(score_list) > 0 and turn == score_list[-1]["turn"]:
            score_list[-1]["valid"].append(check_result["valid"])
            score_list[-1]["number"] = item
        else:
            score_list.append(
                {"turn": turn, "number": item, "valid": [check_result["valid"]]}
            )

    if len(score_list) == 0:
        end_accuracy, process_accuracy = 0, 0
    else:
        end_accuracy, process_accuracy = eval_helper.multiplt_turn_accuracy(score_list)

    eval_result.insert(
        0,
        {
            "accuracy": end_accuracy,
            "correct_count": correct_count,
            "total_count": len(model_results),
            "process_accuracy": process_accuracy,
        },
    )

    output_file_name = "data_" + test_category + "_score.json"
    output_file_dir = paths["OUTPUT_PATH"] / model_name
    eval_utils.save_score_as_json(output_file_name, eval_result, output_file_dir)
    eval_helper.convert_result_to_excel(
        model_name, test_category, paths, FLAGS.language
    )
    return end_accuracy


def special_eval(model_results, prompts, possible_answers, category, model_name, paths):
    """Special Eval."""
    if not all(len(x) == len(model_results) for x in [prompts, possible_answers]):
        raise ValueError(
            f"The length of the model result ({len(model_results)}) does not "
            f"match the length of the prompt ({len(prompts)}) or possible "
            f"answer ({len(possible_answers)}). Please check the input files "
            "for completeness."
        )

    wrong_list = []
    correct_count = 0
    eval_result = []
    for i, (res, answer, prmpt) in enumerate(
        zip(model_results, possible_answers, prompts)
    ):
        prompt_id = prmpt["id"]
        m_res = res["result"]
        answer_ground_truth = answer["ground_truth"]
        eval_result.append(
            {
                "id": prompt_id,
                "valid": True,
                "error": [],
                "error_type": "",
                "model_result_decoded": str(m_res),
                "possible_answer": answer_ground_truth,
            }
        )
        if "incomplete" in category:
            for name, values in answer_ground_truth.items():
                if "Missing necessary parameters" not in m_res:
                    eval_result[i]["valid"] = False
                    eval_result[i]["error"] = [
                        "The user's instruction is missing necessary "
                        f"parameters ({values}) for the ({name}), but the "
                        "model failed to correctly point it out"
                    ]
                    eval_result[i]["error_type"] = "error_detection"
                elif name not in m_res:
                    eval_result[i]["valid"] = False
                    eval_result[i]["error"] = [
                        "The user's instruction is missing necessary "
                        f"parameters ({values}) for the ({name}), but the "
                        "model failed to correctly point it out"
                    ]
                    eval_result[i]["error_type"] = "error_correction"
                else:
                    for value in values:
                        if value not in m_res:
                            eval_result[i]["valid"] = False
                            eval_result[i]["error"] = [
                                "The user's instruction is missing necessary "
                                f"parameters ({value}) for the ({name}), but "
                                "the model failed to correctly point it out"
                            ]
                            eval_result[i]["error_type"] = "error_correction"
        elif "error" in category:
            for name, values in answer_ground_truth.items():
                if "There is incorrect value" not in m_res:
                    eval_result[i]["valid"] = False
                    eval_result[i]["error"] = [
                        "The user's instruction contains incorrect values "
                        f"({values}) of the parameters ({name}), but the model "
                        "failed to correctly point it out"
                    ]
                    eval_result[i]["error_type"] = "error_detection"
                else:
                    for value in values:
                        if value not in m_res:
                            eval_result[i]["valid"] = False
                            eval_result[i]["error"] = [
                                "The user's instruction contains incorrect "
                                f"values ({values}) of the parameters ({name}),"
                                " but the model failed to correctly point it out"
                            ]
                            eval_result[i]["error_type"] = "error_correction"
        elif "irrelevant" in category:
            if "the limitations of the function" not in m_res:
                eval_result[i]["valid"] = False
                eval_result[i]["error"] = [
                    "The model cannot solve this problem, due to the "
                    "limitations of the function"
                ]
                eval_result[i]["error_type"] = "error_detection"

        if eval_result[i]["valid"]:
            correct_count += 1

    for item in eval_result:
        if not item["valid"]:
            wrong_list.append(item)
    accuracy = correct_count / len(model_results)
    wrong_list.insert(
        0,
        {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(model_results),
        },
    )
    output_file_name = "data_" + category + "_score.json"
    output_file_dir = paths["OUTPUT_PATH"] / model_name
    eval_utils.save_score_as_json(output_file_name, wrong_list, output_file_dir)
    eval_helper.convert_result_to_excel(model_name, category, paths, FLAGS.language)
    return accuracy


def agent_eval(
    model_result, prompt, possible_answer, test_category, model_name, language, paths
):
    """Agent Eval."""
    if not all(len(x) == len(model_result) for x in [prompt, possible_answer]):
        raise ValueError(
            f"The length of the model result ({len(model_result)}) does not "
            f"match the length of the prompt ({len(prompt)}) or possible "
            f"answer ({len(possible_answer)}). Please check the input files for"
            " completeness."
        )

    result = []
    correct_index = []
    correct_count = 0

    for i, (res, ans) in enumerate(zip(model_result, possible_answer)):
        res_list = res["result"]
        ans_list = ans["ground_truth"]

        if not isinstance(ans_list, list):
            ans_list = [ans_list]

        result_tmp = {"id": i, "valid": True, "error": [], "error_type": ""}
        is_valid = True
        checker_result = {"valid": True}

        if len(ans_list) != len(res_list):
            result_tmp["valid"] = False
            result_tmp["error_type"] = "wrong number of class"
            is_valid = False
        else:
            for ans in ans_list:
                matched_dict = None
                for r in res_list:
                    if r.keys() == ans.keys():
                        matched_dict = r
                        break
                if matched_dict:
                    checker_result = eval_checker.agent_checker(matched_dict, ans)

                if not checker_result["valid"]:
                    result_tmp["valid"] = False
                    result_tmp["error"].append(checker_result["error"])
                    result_tmp["error_type"] = checker_result["error_type"]
                    is_valid = False

        if not is_valid:
            result.append(result_tmp)
        else:
            correct_count += 1
            correct_index.append(i)

    accuracy = round(correct_count / len(model_result), 3)
    process_accuracy = agent_eval_process(
        model_name,
        model_result,
        possible_answer,
        test_category,
        correct_index,
        language,
    )
    result.insert(
        0,
        {
            "end_to_end_accuracy": accuracy,
            "process_accuracy": process_accuracy,
            "correct_count": correct_count,
            "total_count": len(model_result),
        },
    )
    output_file_name = "data_" + test_category + "_score.json"
    output_file_dir = paths["OUTPUT_PATH"] / model_name
    eval_utils.save_score_as_json(output_file_name, result, output_file_dir)
    # TODO(sabina): missing export to excel.
    return accuracy, process_accuracy


def get_accuracy_for_call_process(call_process, results):
    result_len = len(results)
    milestone_len = len(call_process)
    result_indices = []
    current_index = 0
    # Iterate through each element in call_process and search sequentially
    for stone in call_process:
        # Start searching from the current index until the corresponding call_process element is found
        while current_index < result_len:
            if results[current_index].strip() == stone.strip():
                result_indices.append(current_index)
                current_index += 1
                break
            current_index += 1

    # Calculate call_process accuracy using floating-point division
    if milestone_len == 0:
        # this is used for eval calculation
        accuracy = 1.00
    else:
        accuracy = len(result_indices) / milestone_len
    return accuracy


def agent_eval_process(
    model_name, model_results, possible_answers, test_category, correct_list, language
):
    """Agent Eval Process."""
    individual_accuracies = []  # Used to store the accuracy of each data point
    total_accuracy = 0  # Store the total accuracy of all data

    for i, (results, answer) in enumerate(zip(model_results, possible_answers)):
        if i in correct_list:
            accuracy = 1.00
            total_accuracy += 1.00
            continue

        call_processes = answer["mile_stone"]
        result = results["process"]
        if isinstance(call_processes[0], list):
            max_accuracy = -1
            for call_process in call_processes:
                accuracy = get_accuracy_for_call_process(call_process, result)
                rounded_accuracy = round(accuracy, 3)
                if rounded_accuracy > max_accuracy:
                    max_accuracy = rounded_accuracy
                    name = test_category + "_" + str(i)

        # For a single answer, calculate directly
        else:
            accuracy = get_accuracy_for_call_process(call_processes, result)
            rounded_accuracy = round(accuracy, 3)
            name = test_category + "_" + str(i)

        # Save the accuracy of each data point
        if accuracy != 1.00:
            individual_accuracies.append(
                {
                    name: {
                        "process_accuracy": rounded_accuracy,
                        "model_output": result,
                        "call_process": call_processes,
                    }
                }
            )

        # Accumulate total accuracy
        total_accuracy += accuracy

    # Calculate the overall accuracy of all entries
    overall_accuracy = total_accuracy / len(model_results)
    overall_accuracy = round(overall_accuracy, 3)  # Keep two decimal places
    if language == "zh":
        file_name = (
            "./score_all/score_zh/"
            + model_name
            + "/data_"
            + test_category
            + "_process.json"
        )
    elif language == "en":
        file_name = (
            "./score_all/score_en/"
            + model_name
            + "/data_"
            + test_category
            + "_process.json"
        )
    # Write individual_accuracies to JSON file line by line
    with open(file_name, "w", encoding="utf-8") as f:
        for entry in individual_accuracies:
            json.dump(entry, f, ensure_ascii=False)
            f.write(
                "\n"
            )  # Write a newline character to make each JSON object occupy a separate line

    # Return the accuracy of each data point and the overall accuracy
    return overall_accuracy


def run_eval(model_name, category, paths, language):
    """Run evaluation for the model given the category."""
    # str legacy helper.
    model_result_path = eval_utils.build_result_path(
        paths["INPUT_PATH"], model_name, category, "_result.json"
    )
    model_results = eval_helper.load_file(model_result_path)
    prompt_path = eval_utils.build_data_path(paths["PROMPT_PATH"], category)
    prompt = eval_helper.load_file(prompt_path)
    possible_answer_path = eval_utils.build_data_path(
        paths["POSSIBLE_ANSWER_PATH"], category
    )
    possible_answers = eval_helper.load_file(possible_answer_path)

    if "special" in category:
        accuracy = special_eval(
            model_results,
            prompt,
            possible_answers,
            category,
            model_name,
            paths,
        )
        print(
            f"Model: {model_name} | ✔️ Test '{category}' is done! "
            f"🚀 Accuracy: {accuracy}."
        )
    elif "agent" in category:
        end_accuracy, process_accuracy = agent_eval(
            model_results,
            prompt,
            possible_answers,
            category,
            model_name,
            language,
            paths,
        )
        print(
            f"Model: {model_name} | ✔️ Test '{category}' is done! | "
            f"End_to_End Accuracy: {end_accuracy} | Process Accuracy: "
            f"{process_accuracy}"
        )
    elif "normal_multi_turn" in category:
        end_accuracy = normal_multi_turn_eval(
            model_results,
            prompt,
            possible_answers,
            category,
            model_name,
            paths,
        )
        print(
            f"Model: {model_name} | ✔️ Test '{category}' is done! | "
            f"Accuracy: {end_accuracy}"
        )
    else:
        accuracy = normal_single_turn_eval(
            model_results,
            prompt,
            possible_answers,
            category,
            model_name,
            paths,
        )
        print(
            f"Model: {model_name} | ✔️ Test '{category}' is done! | Accuracy: {accuracy}"
        )


def runner(model_names, categories, paths, language):
    """Main runner function."""
    for model_name in model_names:
        for category in categories:
            print(f"🔍 Running test: {category}")
            run_eval(model_name, category, paths, language)

    eval_helper.update_result_table_with_score_file(
        RESULT_TABLE, str(paths["OUTPUT_PATH"])
    )
    eval_helper.generate_result_csv(RESULT_TABLE, str(paths["OUTPUT_PATH"]))


def main(argv: list[str]):
    """Main."""
    del argv  # unused although it's needed for app.run

    paths = get_paths(FLAGS.language)
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
