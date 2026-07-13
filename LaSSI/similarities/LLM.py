import json
import httpx
from openai import OpenAI

# Per-request timeout for Ollama inference. 120s is generous for local models;
# the openai default (600s × 3 attempts) would block runs for 30+ minutes on timeout.
_OLLAMA_TIMEOUT = httpx.Timeout(timeout=120.0, connect=5.0)


class LLMPrompt:
    def __init__(self, model_name="qwen2.5"):
        self.model_name = model_name
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            timeout=_OLLAMA_TIMEOUT,
            max_retries=0,
        )

    _PROMPT_TEMPLATE = (
        "You are a strict logical reasoning assistant.\n\n"
        "Premise: '{premise}'\n"
        "Consequence: '{consequence}'\n\n"
        "Analyze if the premise implies the consequence. "
        "Respond ONLY in valid JSON using the following format:\n"
        "{{\n"
        "  \"reasoning\": \"Briefly explain your logic here\",\n"
        "  \"implication_score\": 1.0 (if fully true), 0.5 (if partially true), or 0.0 (if false)\n"
        "}}"
    )

    def _build_prompt(self, premise: str, consequence: str) -> str:
        """Build the prompt sent to the model. Subclasses override this to
        inject extra grounding (e.g. HOnK ontology facts) while reusing the
        Ollama dispatch/parse plumbing in `_dispatch`."""
        return self._PROMPT_TEMPLATE.format(premise=premise, consequence=consequence)

    def _query(self, premise: str, consequence: str) -> dict:
        """Send one prompt to Ollama and return the parsed JSON dict."""
        return self._dispatch(self._build_prompt(premise, consequence))

    def _dispatch(self, prompt: str, _retries: int = 1) -> dict:
        """Send a fully-built prompt to Ollama and return the parsed JSON dict.

        A failed call (malformed JSON, missing score key, or transport error)
        is retried once before defaulting to 0.5, and every defaulted result
        carries a bracketed marker in `reasoning` so downstream analysis can
        count and exclude them.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                seed=42,
                response_format={"type": "json_object"},
            )
            raw_content = response.choices[0].message.content
            data = json.loads(raw_content)
            if "implication_score" not in data:
                raise ValueError("response JSON lacks 'implication_score'")
            return {
                "implication_score": float(data["implication_score"]),
                "reasoning": data.get("reasoning", ""),
            }
        except (json.JSONDecodeError, ValueError) as e:
            if _retries > 0:
                print(f"[LLM] JSON Parsing Error (retrying): {e}")
                return self._dispatch(prompt, _retries - 1)
            print(f"[LLM] JSON Parsing Error: {e}")
            return {"implication_score": 0.5, "reasoning": f"[parse error] {e}"}
        except Exception as e:
            if _retries > 0:
                print(f"[LLM] Request failed (retrying, {type(e).__name__}): {e}")
                return self._dispatch(prompt, _retries - 1)
            print(f"[LLM] Request failed ({type(e).__name__}): {e}")
            return {"implication_score": 0.5, "reasoning": f"[request error] {e}"}

    def call_with_reasoning(self, premise: str, consequence: str) -> tuple:
        """Return (implication_score: float, reasoning: str)."""
        result = self._query(premise, consequence)
        return result["implication_score"], result["reasoning"]

    def __call__(self, premise: str, consequence: str) -> float:
        return self._query(premise, consequence)["implication_score"]