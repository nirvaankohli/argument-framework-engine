import json
from typing import List, Dict, Any, Optional
import requests
from dataclasses import dataclass, asdict
from pathlib import Path


USER_AGENT = "argument-mapper/0.1"


@dataclass
class Utterance:
    id: str
    speaker_id: Optional[str]
    parent_id: Optional[str]
    author: Optional[str]
    text: str
    depth: int


class RedditScraper:
    def __init__(self, user_agent: str = USER_AGENT, timeout: float = 10.0) -> None:
        self.user_agent = user_agent
        self.timeout = timeout

    def to_json_url(self, url: str) -> str:

        if url.endswith(".json"):
            return url
        if not url.endswith("/"):
            url += "/"
        return url + ".json"

    def fetch_reddit_post(self, url: str) -> List[Dict[str, Any]]:

        json_url = self.to_json_url(url)
        resp = requests.get(
            json_url,
            headers={"User-Agent": self.user_agent},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def process_post(self, url: str) -> Dict[str, Any]:

        speaker_map: Dict[str, str] = {}
        speaker_counter = 0

        def new_sid() -> str:
            nonlocal speaker_counter
            speaker_counter += 1
            return f"S{speaker_counter}"

        def get_speaker_id(author: Optional[str]) -> Optional[str]:

            if not author:
                return None
            if author in ("[deleted]", "[removed]"):
                return None
            if author in speaker_map:
                return speaker_map[author]
            sid = new_sid()
            speaker_map[author] = sid
            return sid

        data = self.fetch_reddit_post(url)

        # Basic structural sanity checks
        if not isinstance(data, list) or len(data) < 2:
            raise ValueError(
                "Unexpected Reddit JSON structure: top-level list length < 2"
            )

        post_listing = data[0]["data"]["children"]
        if not post_listing:
            raise ValueError("No post data found in Reddit JSON")

        post_info = post_listing[0]["data"]
        comment_children = data[1]["data"]["children"]

        utterances: List[Utterance] = []

        counter = 0

        def new_uid() -> str:
            nonlocal counter
            counter += 1
            return f"U{counter}"

        # Root post
        root_uid = new_uid()
        root_text = post_info.get("title", "") or ""
        selftext = post_info.get("selftext") or ""
        if selftext:
            root_text += "\n\n" + selftext

        utterances.append(
            Utterance(
                id=root_uid,
                speaker_id=get_speaker_id(post_info.get("author")),
                parent_id=None,
                author=post_info.get("author"),
                text=root_text,
                depth=0,
            )
        )

        # Recursively walk comments
        def walk(children: List[Dict[str, Any]], parent_uid: str, depth: int) -> None:
            nonlocal utterances

            for child in children:
                if child.get("kind") != "t1":
                    # Skip "more" and other non-comment items
                    continue

                cdata = child["data"]
                body = cdata.get("body", "") or ""

                # Skip deleted/removed comments
                if body.strip() in ("[deleted]", "[removed]"):
                    continue

                uid = new_uid()
                utterances.append(
                    Utterance(
                        id=uid,
                        speaker_id=get_speaker_id(cdata.get("author")),
                        parent_id=parent_uid,
                        author=cdata.get("author"),
                        text=body,
                        depth=depth,
                    )
                )

                replies = cdata.get("replies")
                if isinstance(replies, dict):
                    reply_children = replies.get("data", {}).get("children", [])
                    if reply_children:
                        walk(reply_children, uid, depth + 1)

        walk(comment_children, root_uid, depth=1)

        self.result = {
            "thread_id": post_info.get("id"),
            "subreddit": post_info.get("subreddit"),
            "title": post_info.get("title"),
            "utterances": [u for u in utterances],
        }

        return self.result
    

    def return_to_json(self) -> Dict[str, Any]:


        return {
        "thread_id": self.result["thread_id"],
        "subreddit": self.result["subreddit"],
        "title": self.result["title"],
        "utterances": [asdict(u) for u in self.result["utterances"]],
    }


if __name__ == "__main__":

    url = "https://www.reddit.com/r/Backend/comments/1on8vm6/go_vs_rust_which_one_is_better/"

    scraper = RedditScraper()
    post_data = scraper.process_post(url)
    post_data = scraper.return_to_json()

    print(json.dumps(post_data, indent=2))

    

    example_path = Path(__file__).parent.parent / "example" / "example_reddit_post.json"
    example_path.parent.mkdir(parents=True, exist_ok=True)

    with open(example_path, "w", encoding="utf-8") as f:

        json.dump(post_data, f, indent=2, ensure_ascii=False)

    print(f"Wrote {example_path}")
