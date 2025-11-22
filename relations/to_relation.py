from pathlib import Path
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import is_dataclass
from typing import Any, Optional, List, Dict
import importlib
import importlib.util
from dataclasses import dataclass
from typing import Literal
import traceback
from dotenv import load_dotenv
import os
import json
import requests

# ANNOYING STUFF TO IMPORT CANONICAL DATA CLASSES

ROOT_DIR = Path(__file__).parent.parent

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


try:

    to_argument = importlib.import_module("arguments.to_argument")

except Exception as e:
    repo_root = Path(__file__).parent.parent

    alt_path = repo_root / "arguments" / "to_argument.py"

    spec = importlib.util.spec_from_file_location(
        "arguments.to_argument", str(alt_path)
    )

    if spec is None or spec.loader is None:

        raise ImportError("Could not import scraping.reddit.utterances") from e

    module = importlib.util.module_from_spec(spec)

    try:

        spec.loader.exec_module(module)
        to_argument = module

    except Exception as e2:

        tb = traceback.format_exc()

        raise ImportError(
            f"Could not import arguments.to_argument from {alt_path}:\n{tb}"
        ) from e2

RelationType = Literal["supports", "attacks"]


@dataclass
class Relation:
    from_id: str  # source argument id
    to_id: str  # target argument id
    from_text: str  # source argument text
    to_text: str  # target argument text
    type: RelationType  # "supports" or "attacks"
    strength: float  # 0.0–1.0 confidence
    explanation: str  # short natural-language reason


@dataclass
class Argument:

    id: str  # A1, A2, ...
    text: str  # the claim itself
    utterance_id: str  # which comment/line it came from
    speaker: Optional[str] = None
    confidence: Optional[float] = None  # 0–1 from the LLM


# yay this is real code below me


class ToRelation:

    def __init__(self, arguments) -> None:

        if is_dataclass(arguments[0]):

            self.arguments: List[Argument] = arguments

        else:

            self.arguments = [Argument(**arg) for arg in arguments]

        self.relations: List[Relation] = []

    def extract_relations(self, to_json: bool = False) -> List[Relation]:

        llm_cli = llm()
        llm_response = llm_cli.process_send_request(self.arguments)

        relations_data = llm_response.get("relations", [])
        relations: List[Relation] = []

        # Create a lookup dict for argument texts by ID
        arg_text_lookup = {arg.id: arg.text for arg in self.arguments}

        for rel in relations_data:
            try:
                from_id = rel["from"]
                to_id = rel["to"]

                # Look up the text for from and to arguments
                from_text = arg_text_lookup.get(from_id, "")
                to_text = arg_text_lookup.get(to_id, "")

                relation = Relation(
                    from_id=from_id,
                    to_id=to_id,
                    from_text=from_text,
                    to_text=to_text,
                    type=rel["type"],
                    strength=float(rel["strength"]),
                    explanation=rel["explanation"],
                )
                relations.append(relation)
            except KeyError as e:
                raise ValueError(f"Missing key in LLM response relation: {e}") from e
            except ValueError as e:
                raise ValueError(f"Invalid value in LLM response relation: {e}") from e

        if to_json:
            with open("relations.json", "w", encoding="utf-8") as f:
                json.dump([asdict(rel) for rel in relations], f, indent=2)

        return relations


class llm:

    def __init__(self, api_key: str = "use_local", timeout: float = 200.0) -> None:

        if api_key == "use_local":

            env_path = Path(ROOT_DIR) / ".env"
            load_dotenv(env_path)
            api_key = os.getenv("API_KEY", "")

            if not api_key:
                raise ValueError(
                    f"API_KEY not found in environment or .env file at {env_path}"
                )

        self.default_prompts()

        self.api_key = api_key
        self.api_url = "https://ai.hackclub.com/proxy/v1/chat/completions"
        self.model = "google/gemini-2.5-flash"
        self.timeout = timeout

    def default_prompts(self) -> None:

        self.system_prompt = """
You classify relations between arguments.

Given a JSON array of arguments, you must return ONLY JSON of direct relations:
- "supports" if one argument gives reasons for another.
- "attacks" if one argument gives reasons against another.
If there is no clear relation, omit the pair entirely.
Return strict JSON only.
""".strip()

        self.user_prompt = """
You are given a JSON array of arguments. Each argument has:
- id
- text
- speaker_id
- utterance_id

Task:
- For all ordered pairs (Ai, Aj), i != j, detect direct SUPPORT or ATTACK.
- Only include pairs with a clear relation.
- "from" = argument doing the supporting/attacking.
- "to"   = argument being supported/attacked.

Respond with JSON only:

{
  "relations": [
    {
      "from": "A1",
      "to": "A2",
      "type": "supports" or "attacks",
      "explanation": "Short reason for this relation.",
      "strength": 0.0
    }
  ]
}

Arguments:
{{DATA}}
""".strip()

    def build_user_prompt(self, data: str) -> str:

        return self.user_prompt.replace("{{DATA}}", data)

    def build_system_prompt(self) -> str:

        return self.system_prompt

    def build_payload(self, system: str, user: str) -> Dict[str, Any]:

        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": 200000,
        }

    def process_send_request(self, arguments, per=20):

        split_json = []

        for i in range(0, len(arguments), per):

            split_json.append(arguments[i : i + per])

        response = []

        for args in split_json:

            minimal_args = [
                {
                    "id": arg.id,
                    "text": arg.text,
                    "speaker_id": getattr(arg, "speaker", None),
                    "utterance_id": arg.utterance_id,
                }
                for arg in args
            ]

            user_content = self.build_user_prompt(
                data=json.dumps(minimal_args, separators=(",", ":"))
            )

            system_content = self.build_system_prompt()

            payload = self.build_payload(system_content, user_content)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            r = requests.post(
                url=self.api_url, json=payload, headers=headers, timeout=self.timeout
            )

            if r.status_code == 200:

                try:

                    response_data = r.json()
                    content = response_data["choices"][0]["message"]["content"]

                    with open(
                        f"debug_response_{len(response)}.json", "w", encoding="utf-8"
                    ) as f:
                        json.dump(
                            {"content": content, "full_response": response_data},
                            f,
                            indent=2,
                        )

                    if not content or content.strip() == "":
                        print(
                            f"Warning: Empty content from LLM for batch {len(response)}"
                        )
                        response.append([])
                        continue

                    content = content.strip()
                    if content.startswith("```json"):
                        content = (
                            content.replace("```json", "").replace("```", "").strip()
                        )

                    parsed_content = json.loads(content)
                    relations = parsed_content.get("relations", [])
                    response.append(relations)

                except (KeyError, json.JSONDecodeError) as e:
                    print(f"Error parsing LLM response for batch {len(response)}: {e}")
                    print(f"Content was: {repr(content[:200])}")
                    # Continue with empty relations instead of failing
                    response.append([])
                    continue
            else:
                raise ValueError(
                    f"LLM request failed with status {r.status_code}: {r.text}"
                )

        return {"relations": [item for sublist in response for item in sublist]}


if __name__ == "__main__":

    import time

    s0 = time.time()

    path_here = Path(__file__).parent.resolve()

    if os.path.exists(path_here / "sample_arguments.json"):

        with open(path_here / "sample_arguments.json", "r", encoding="utf-8") as f:
            json_data = json.load(f)

    else:

        json_data = []

    if json_data == []:

        print("No relations found.")

        cleaner = to_argument.ArgumentCleaner()
        utteranceClient = reddit_utterances.RedditScraper()

        post_data = utteranceClient.process_post(
            "https://www.reddit.com/r/Backend/comments/1p21idn/for_experienced_backend_engineers/"
        )

        if post_data is None:

            raise ValueError("Failed to fetch Reddit post data.")

        arguments = cleaner.convert_utterances_to_arguments(
            post_data["utterances"], threshold=0.67
        )

        json_data = [asdict(arg) for arg in arguments]

        with open(path_here / "sample_arguments.json", "w", encoding="utf-8") as f:

            json.dump(json_data, f, indent=2)

    print(f"Loaded {len(json_data)} arguments for relation extraction.")

    s1 = time.time()
    print(f"Data preparation took {s1 - s0:.2f} seconds")

    to_relation = ToRelation(json_data)
    llm_response = to_relation.extract_relations(to_json=True)

    s2 = time.time()

    print(f"Extracted {len(llm_response)} relations:")
    print(f"Relation extraction took {s2 - s1:.2f} seconds")
    print(f"Total time: {s2 - s0:.2f} seconds")