import os

import subprocess
from openai import OpenAI
import vllm
import time
from model_inference import gemini


def get_free_gpu(use_gpu_num):
    # Use nvidia-smi command to get GPU usage information
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    # Parse the output
    gpu_info = result.stdout.strip().split("\n")
    free_gpus = [
        (int(info.split(",")[0]), int(info.split(",")[1])) for info in gpu_info
    ]

    # Sort by free memory in descending order
    free_gpus.sort(key=lambda x: x[1], reverse=True)

    # Return GPU IDs with the most free memory
    return ",".join(str(i[0]) for i in free_gpus[:use_gpu_num])


class LLMInfer(object):
    def __init__(
        self,
        model_path,
        temperature=0.001,
        top_p=1,
        max_tokens=1000,
        language="zh",
        max_model_len=8192,
        tensor_parallel_size=1,
    ) -> None:
        gpu_ids = get_free_gpu(use_gpu_num=tensor_parallel_size)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        self.sampling_params = vllm.SamplingParams(
            temperature=0.0, max_tokens=1024, top_p=0.9
        )
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.llm = vllm.LLM(
            model=model_path,
            dtype="float16",
            trust_remote_code=True,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.9,
        )

    def _format_prompt(self, messages):
        # Qwen is using its prompting mode, not the tool use mode
        """
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n",
        """
        formatted_prompt = ""

        for message in messages:
            formatted_prompt += (
                f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            )

        formatted_prompt += "<|im_start|>assistant\n"

        return formatted_prompt

    def inference(self, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        question = [
            prompt,
        ]
        outputs = self.llm.generate(question, self.sampling_params, use_tqdm=False)
        if len(question) == 1:
            result = outputs[0].outputs[0].text
        else:
            result = []
            for idx, output in enumerate(outputs):
                out_generated_text = output.outputs[0].text
                result.append(out_generated_text)
        return result


class Llama(LLMInfer):
    def __init__(
        self,
        model_path,
        temperature=0.001,
        top_p=1,
        max_tokens=1000,
        language="zh",
        max_model_len=8192,
        tensor_parallel_size=2,
    ) -> None:
        super().__init__(
            model_path,
            temperature,
            top_p,
            max_tokens,
            language,
            max_model_len,
            tensor_parallel_size,
        )
