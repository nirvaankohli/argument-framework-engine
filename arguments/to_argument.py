from pathlib import Path
from dataclasses import asdict
from typing import Any, Optional, List, Dict
import importlib
import importlib.util
import traceback
from dotenv import load_dotenv
import os
import json
import requests

ROOT_DIR = Path(__file__).parent.parent


def _load_data_module() -> Any:
    """Load the format.data module with fallback to file-based import."""
    try:
        return importlib.import_module("format.data")
    except Exception:
        data_path = ROOT_DIR / "format" / "data.py"
        spec = importlib.util.spec_from_file_location("format.data", str(data_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create spec for {data_path}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
            return module
        except Exception as e:
            tb = traceback.format_exc()
            raise ImportError(
                f"Could not import format.data from {data_path}:\n{tb}"
            ) from e


# Load canonical data classes
data_module = _load_data_module()
Argument = getattr(data_module, "Argument")
Utterance = getattr(data_module, "Utterance")


try:

    reddit_utterances = importlib.import_module("scraping.reddit.utterances")

except Exception as e:
    repo_root = Path(__file__).parent.parent

    alt_path = repo_root / "scraping" / "reddit" / "utterances.py"

    spec = importlib.util.spec_from_file_location(
        "scraping.reddit.utterances", str(alt_path)
    )

    if spec is None or spec.loader is None:

        raise ImportError("Could not import scraping.reddit.utterances") from e

    module = importlib.util.module_from_spec(spec)

    try:

        spec.loader.exec_module(module)
        reddit_utterances = module

    except Exception as e2:

        tb = traceback.format_exc()

        raise ImportError(
            f"Could not import scraping.reddit.utterances from {alt_path}:\n{tb}"
        ) from e2


class ArgumentCleaner:

    def __init__(self):

        self.claims_indicators = [
            "i believe",
            "i think",
            "in my opinion",
            "it is clear that",
            "the evidence shows",
            "studies indicate",
            "research suggests",
            "it is obvious that",
            "i am convinced",
            "there is no doubt that",
            " is ",
            " are ",
            " should ",
            " must ",
            " will ",
            " can't ",
        ]

    def looks_like_claim(self, sentence: str) -> bool:

        sentence_lower = sentence.lower()
        return any(phrase in sentence_lower for phrase in self.claims_indicators)

    def convert_utterances_to_arguments(
        self, utterances: List[Any], threshold: float
    ) -> List[Any]:

        llm_client = llm()
        batch_size = 10
        all_claims = []

        # Split utterances into batches
        utterance_batches = []
        current_batch = []

        for utterance in utterances:
            current_batch.append(utterance)
            if len(current_batch) >= batch_size:
                utterance_batches.append(current_batch)
                current_batch = []

        # Don't forget the last batch if it has items
        if current_batch:
            utterance_batches.append(current_batch)

        # Process each batch and collect claims
        for batch in utterance_batches:
            try:
                batch_result = llm_client.process_utterances(batch)
                all_claims.extend(batch_result["claims"])
            except Exception as e:
                print(f"Warning: Failed to process batch: {e}")
                continue

        # Convert claims to arguments
        arguments = []
        next_id = 1

        for claim_data in all_claims:
            confidence = claim_data.get("confidence", 0.0)

            if confidence >= threshold:
                arg = Argument(
                    id=f"A{next_id}",
                    text=claim_data["text"],
                    utterance_id=claim_data["id"],
                    speaker=claim_data.get("speaker_id"),
                    confidence=confidence,
                )
                arguments.append(arg)
                next_id += 1

        return arguments


class llm:

    def __init__(self, api_key: str = "use_local", timeout: float = 200.0) -> None:

        self.system_prompt = """
        You extract atomic argumentative claims from text and return strict JSON.

        A CLAIM is:
        - A self-contained statement that asserts a viewpoint, judgment, or interpretation.
        - Something a reasonable person could agree or disagree with.
        - Not just a question, greeting, joke, or pure description of facts without a stance.

        Your job:
        - Read the provided utterances.
        - Rewrite each claim as a single clear sentence in neutral tone (no “I think”, “IMO”, slang).
        - Return ONLY valid JSON in the schema the user provides. Do not add explanations or commentary.
        """

        self.user_prompt = """

        For each utterance:
        - Extract zero or more atomic CLAIMS.
        - Be a single, self-contained sentence.
        - Express a viewpoint or opinion that can be supported or opposed.
        - Be rewritten in neutral, clear language (remove “IMO”, “I feel”, emojis, etc.).
        - If an utterance has no claims, produce no entries for that utterance.

        Return JSON with this exact shape:

        {
        "claims": [
            {
            "id": "U2",         // Given
            "speaker_id": "S2",           // Given
            "text": "Claim text here.",   // the rewritten claim
            "confidence": 0.0             // float in [0.0, 1.0] - be as specific as possible about how confident you are that this is a real claim - dont be afraid to go into the thousands
            }
        ]
        }

        Use `confidence` to show how sure you are that a sentence is a real claim.

        Here are the utterances:

        {{UTTERANCES_JSON}}

        """

        if api_key == "use_local":

            env_path = Path(ROOT_DIR) / ".env"
            load_dotenv(env_path)
            api_key = os.getenv("API_KEY", "")

            if not api_key:
                raise ValueError(
                    f"API_KEY not found in environment or .env file at {env_path}"
                )

        self.api_key = api_key
        self.api_url = "https://ai.hackclub.com/proxy/v1/chat/completions"
        self.model = "x-ai/grok-4.1-fast"
        self.timeout = timeout

    def process_utterances(self, utterance: List[Any]) -> Dict[str, Any]:

        utterances = [asdict(u) for u in utterance]

        user_content = self.user_prompt.replace(
            "{{UTTERANCES_JSON}}", json.dumps(utterances)
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.7,
            "max_tokens": 20000,
        }

        r = requests.post(
            url=self.api_url, json=data, headers=headers, timeout=self.timeout
        )

        if r.status_code == 200:
            try:
                response_data = r.json()
                content = response_data["choices"][0]["message"]["content"]

                # Save response for debugging
                with open("llm_response.json", "w", encoding="utf-8") as f:
                    json.dump(response_data, f, indent=2)

                return json.loads(content)
            except (KeyError, json.JSONDecodeError) as e:
                raise ValueError(f"Failed to parse LLM response: {e}") from e
        else:
            raise ValueError(
                f"LLM request failed with status {r.status_code}: {r.text}"
            )


if __name__ == "__main__":

    cleaner = ArgumentCleaner()
    utteranceClient = reddit_utterances.RedditScraper()
    post_data = utteranceClient.process_post(
        "https://www.reddit.com/r/Backend/comments/1p21idn/for_experienced_backend_engineers/"
    )


    if post_data is None:

        raise ValueError("Failed to fetch Reddit post data.")

    arguments = cleaner.convert_utterances_to_arguments(
        
        post_data["utterances"], threshold=0.67
    
    )

    for arg in arguments:
        print(asdict(arg))
