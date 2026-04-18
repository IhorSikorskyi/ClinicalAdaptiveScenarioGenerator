import requests

class VirtualPatientEngine:
    def __init__(self, model_name="llama3.1", host="http://localhost:11434"):
        self.model_name = model_name
        self.url = f"{host}/api/generate"

    def generate_response(self, prompt, system_prompt=None):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 100
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(self.url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            return f"Помилка зв'язку з Llama: {e}"