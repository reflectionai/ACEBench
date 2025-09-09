
import os

from openai import OpenAI
from model_inference import gemini, llm_infer





class Deepseek(object):
    def __init__(self, model_name, model_path=None, temperature=0.001, top_p=1, max_tokens=1000, language="zh") -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = "https://api.deepseek.com"
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, timeout=1000, max_retries=1, base_url=base_url)

    def creat_message(self, system_prompt=None, user_prompt=None, few_shot_examples=None):
        messages = []
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}]
        if few_shot_examples:
            for item in few_shot_examples:
                user, assistant = item["user"], item["assistant"]
                messages.append({"role": "system", "name": "example_user", "content": user})
                messages.append({"role": "system", "name": "example_assistant", "content": assistant})
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
        return messages
    

class YourClass:
    def __init__(self, model_name):
        self.model_name = model_name
        self.last_request_time = 0  # To track the time of the last request


    def inference(self, system_prompt, user_prompt):
        messages = self.creat_message(system_prompt=system_prompt, user_prompt=user_prompt)
        result = self.request_openai(messages=messages, model=self.model_name)
        return result


class Kimi(object):
    def __init__(self, model_name, model_path=None, temperature=0.001, top_p=1, max_tokens=1000, language="zh") -> None:
        api_key = os.getenv("KIMI_API_KEY")
        base_url = os.getenv("KIMI_BASE_URL")
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, timeout=1000, max_retries=1, base_url=base_url)

    def creat_message(self, system_prompt=None, user_prompt=None):
        messages = []
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}]
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
        return messages

    def inference(self, system_prompt, user_prompt):
        messages = self.creat_message(system_prompt=system_prompt, user_prompt=user_prompt)
        response = self.client.chat.completions.create(  
            model = self.model_name,
            messages=messages,
            max_tokens=1024,
            temperature=0.0,
            top_p=0,
            seed=42,
        )
        return response.choices[0].message.content


model_dict = {}
def get_model(model_name, model_path):
    global model_dict
    if model_name in model_dict:
        model = model_dict[model_name]
    else:
        model_name_lower = model_name.lower()
        if "qwen" in model_name_lower:
            model = llm_infer.LLMInfer(model_path)
        elif "llama" in model_name_lower:
            model = llm_infer.Llama(model_path)
        elif "deepseek" in model_name_lower:
            model = Deepseek(model_name)
        elif "gemini" in model_name_lower:
            model = gemini.Gemini(model_name)
        elif "kimi" in model_name_lower:
            model = Kimi(model_name)
        elif model_path:
            model = llm_infer.Llama(model_path)
        else:
            raise("Unsupported model")
        model_dict[model_name] = model
    return model

