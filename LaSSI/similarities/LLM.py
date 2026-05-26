import json
from openai import OpenAI


class LLMPrompt:
    def __init__(self, model_name="qwen2.5"):
        self.model_name = model_name
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )

    def __call__(self, premise: str, consequence: str) -> float:
        # We explicitly define the expected JSON schema in the prompt
        prompt = (
            "You are a strict logical reasoning assistant.\n\n"
            f"Premise: '{premise}'\n"
            f"Consequence: '{consequence}'\n\n"
            "Analyze if the premise implies the consequence. "
            "Respond ONLY in valid JSON using the following format:\n"
            "{\n"
            "  \"reasoning\": \"Briefly explain your logic here\",\n"
            "  \"implication_score\": 1.0 (if fully true), 0.5 (if partially true), or 0.0 (if false)\n"
            "}"
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}  # Forces JSON output
        )

        raw_content = response.choices[0].message.content
        print(raw_content)

        try:
            # Parse the JSON and extract the float score directly
            data = json.loads(raw_content)
            return float(data.get("implication_score", 0.0))
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[LLM] JSON Parsing Error. Raw output: {raw_content}")
            return 0.5