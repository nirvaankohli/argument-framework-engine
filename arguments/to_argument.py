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
import math

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
        self,
        utterances: List[Any],
        threshold: float,
        utterance_data: Dict[str, Any] = None,
    ) -> List[Any]:

        # Check if arguments already exist in database
        if utterance_data:
            db_instance = ArgumentDB()
            db_instance.update_db()

            cache_key = f"{utterance_data.get('thread_id', '')}-{threshold}"
            if db_instance.check_if_arguments_in_db(cache_key):
                print("Loading arguments from database cache...")
                cached_args = db_instance.fetch_arguments_from_db(cache_key)
                return [Argument(**arg) for arg in cached_args["arguments"]]

        llm_client = llm()
        batch_count = 2
        per_count = math.ceil(len(utterances) / batch_count)

        while per_count < 10:

            batch_count -= 0.1
            per_count = math.ceil(len(utterances) / batch_count)

            if batch_count <= 1.0:

                per_count = len(utterances)
                break

        batch_size = per_count

        all_claims = []

        utterance_batches = []
        current_batch = []

        for utterance in utterances:

            current_batch.append(utterance)

            if len(current_batch) >= batch_size:

                utterance_batches.append(current_batch)
                current_batch = []

        if current_batch:

            utterance_batches.append(current_batch)

        count = 0

        for batch in utterance_batches:

            try:

                count += 1

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

        print(f"Processed {count} batches of utterances.")

        # Save arguments to database if we have utterance data
        if utterance_data and arguments:
            db_instance = ArgumentDB()
            cache_key = f"{utterance_data.get('thread_id', '')}-{threshold}"
            argument_data = {
                "thread_id": utterance_data.get("thread_id"),
                "subreddit": utterance_data.get("subreddit"),
                "title": utterance_data.get("title"),
                "threshold": threshold,
                "num_arguments": len(arguments),
                "arguments": [asdict(arg) for arg in arguments],
            }
            db_instance.add_arguments_to_db(cache_key, argument_data, utterance_data)

        return arguments


class llm:

    def __init__(self, api_key: str = "use_local", timeout: float = 200.0) -> None:

        self.system_prompt = "Extract argumentative claims from text. Return only JSON. CLAIM = self-contained statement expressing debatable viewpoint. Rewrite claims neutrally."

        self.user_prompt = """

        For each utterance:
        - Extract zero or more atomic CLAIMS.
        - Be a single, self-contained sentence.
        - Express a viewpoint or opinion that can be supported or opposed.
        - Be rewritten in neutral, clear language (remove “IMO”, “I feel”, emojis, etc.).
        - If an utterance has no claims, produce no entries for that utterance.

        Return JSON with this exact shape. Make sure the output is valid JSON.

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

Utterances: {{UTTERANCES_JSON}}"""

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
        self.model = "google/gemini-2.5-flash"
        self.timeout = timeout

    def process_utterances(self, utterance: List[Any]) -> Dict[str, Any]:

        utterances = []
        for u in utterance:
            if isinstance(u, dict):
                utterances.append(u)
            else:
                utterances.append(asdict(u))

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
            "max_tokens": 200000,
        }

        r = requests.post(
            url=self.api_url, json=data, headers=headers, timeout=self.timeout
        )

        if r.status_code == 200:
            try:
                response_data = r.json()
                content = response_data["choices"][0]["message"]["content"]

                import re

                content = re.sub(r"^```json\s*", "", content)
                content = re.sub(r"\s*```$", "", content)

                return json.loads(content)

            except (KeyError, json.JSONDecodeError) as e:
                raise ValueError(f"Failed to parse LLM response: {e}") from e
        else:
            raise ValueError(
                f"LLM request failed with status {r.status_code}: {r.text}"
            )


class ArgumentDB:

    def __init__(self) -> None:

        self.ROOT_DIR = Path(__file__).parent.parent
        self.arg_db_path = self.ROOT_DIR / "data" / "arguments" / "db" / "index.json"

        # Create directory structure if it doesn't exist
        self.arg_db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database file if it doesn't exist
        if not self.arg_db_path.exists():
            with open(self.arg_db_path, "w", encoding="utf-8") as f:
                json.dump({"index": {}}, f, indent=2)

        with open(self.arg_db_path, "r", encoding="utf-8") as f:
            self.arg_db = json.load(f)

    def update_db(self) -> None:

        with open(self.arg_db_path, "r", encoding="utf-8") as f:
            self.arg_db = json.load(f)

    def check_if_arguments_in_db(self, cache_key: str) -> bool:

        if cache_key in self.arg_db["index"].keys():
            return True

        return False

    def add_arguments_to_index(
        self, cache_key: str, subreddit: str, thread_id: str, threshold: float
    ) -> None:

        self.arg_db["index"][
            cache_key
        ] = f"storage/{subreddit}/{thread_id}-args-{threshold}.json"

        with open(self.arg_db_path, "w", encoding="utf-8") as f:
            json.dump(self.arg_db, f, indent=2)

    def add_arguments_to_db(
        self, cache_key: str, data: Dict[str, Any], utterance_data: Dict[str, Any]
    ) -> None:

        subreddit = data["subreddit"]
        thread_id = data["thread_id"]
        threshold = data["threshold"]

        self.add_arguments_to_index(cache_key, subreddit, thread_id, threshold)

        self.new_arguments_dir_path = (
            self.ROOT_DIR / "data" / "arguments" / "db" / "storage" / subreddit
        )
        self.new_arguments_path = (
            self.new_arguments_dir_path / f"{thread_id}-args-{threshold}.json"
        )

        self.new_arguments_dir_path.mkdir(parents=True, exist_ok=True)

        with open(self.new_arguments_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"Saved {data['num_arguments']} arguments to {self.new_arguments_path}")

    def fetch_arguments_from_db(self, cache_key: str) -> Dict[str, Any]:

        if not self.check_if_arguments_in_db(cache_key):
            raise ValueError("Arguments not found in DB")

        path_str = self.arg_db["index"][cache_key]

        full_path = self.ROOT_DIR / "data" / "arguments" / "db" / path_str

        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data


if __name__ == "__main__":

    import time

    t0 = time.time()

    cleaner = ArgumentCleaner()
    utteranceClient = reddit_utterances.RedditScraper()

    print("Fetching Reddit post...")
    print("Processing Reddit post...")

    post_data = utteranceClient.process_post(
        "https://www.reddit.com/r/Backend/comments/1p21idn/for_experienced_backend_engineers/"
    )

    t2 = time.time()
    print(f"Fetched post in {t2 - t0:.2f} seconds.")

    print(f"Fetched post with {len(post_data['utterances'])} utterances.")

    if post_data is None:

        raise ValueError("Failed to fetch Reddit post data.")

    arguments = cleaner.convert_utterances_to_arguments(
        post_data["utterances"], threshold=0.67, utterance_data=post_data
    )

    t1 = time.time()
    print(f"Extracted {len(arguments)} arguments in {t1 - t2:.2f} seconds:")
    print(f"Total time: {t1 - t0:.2f} seconds.")
