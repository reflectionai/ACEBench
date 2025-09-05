
import os

import google.generativeai as genai 
import time


class Gemini(object):
    def __init__(self, model_name, model_path=None, temperature=0.001, top_p=1, max_tokens=1000, language="zh") -> None:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"), transport='rest')
        self.model_name = model_name
        self.last_request_time = 0  # To track the time of the last request

    def creat_message(self, system_prompt=None, user_prompt=None, few_shot_examples=None):
        messages = []
        if few_shot_examples:
            for item in few_shot_examples:
                user, assistant = item["user"], item["assistant"]
                messages.append({"role": "model", "parts": user})
                messages.append({"role": "model", "parts": assistant})
        if user_prompt:
            messages.append({"role": "user", "parts": user_prompt})
        return messages

    def request_gemini(self, system_prompt, messages):
        try:
            # Check time difference from the last request
            current_time = time.time()
            if current_time - self.last_request_time < 7:
                # Sleep if the time difference is less than 6 seconds
                time.sleep(7 - (current_time - self.last_request_time))

            model = genai.GenerativeModel(self.model_name, system_instruction=system_prompt)
            response = model.generate_content(messages)
            result = response.text
            self.last_request_time = time.time()
            return result
        
        except Exception as e:
            raise e

    def inference(self, system_prompt, user_prompt):
        messages = self.creat_message(system_prompt=system_prompt, user_prompt=user_prompt)
        result = self.request_gemini(system_prompt=system_prompt, messages=messages)
        return result
