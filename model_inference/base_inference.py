import json, os

from pathlib import Path
from typing import Optional, Any


class BaseHandler:
    model_name: str

    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        temperature: float = 0.7,
        top_p: int = 1,
        max_tokens: int = 1000,
        language: str = "zh",
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.language = language

    def inference(
        self,
        prompt: str,
        functions,
        test_category: str,
        *args: Any,  # extra positional arguments
        **kwargs: Any,
    ):
        # This method is used to retrive model response for each model.
        pass

    def write_result(self, result, model_name, result_path: Path):
        """Append result to model's result path."""
        model_dir = result_path / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        # os.makedirs(f"{result_path}{model_name}", exist_ok=True)

        if isinstance(result, dict):
            result = [result]

        for entry in result:
            test_id = entry["id"]
            if "normal_multi_turn" not in test_id:
                test_category = test_id.rsplit("_", 1)[0]
            else:
                test_category = "_".join(test_id.split("_")[:-2])

            # file_to_write = (
            #     f"{result_path}{model_name}/data_{test_category}_result.json"
            # )
            model_path = model_dir / f"data_{test_category}_result.json"
            # with open(file_to_write, "a+", encoding="utf-8") as f:
            with model_path.open("a+", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
