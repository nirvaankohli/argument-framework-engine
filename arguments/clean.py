from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import importlib.util
import sys

module_dir = Path(__file__).resolve().parent.parent / "format"
scrape_dir = Path(__file__).resolve().parent.parent / "scraping" / "reddit"

sys.path.append(module_dir)
sys.path.append(scrape_dir)

from format.data import Utterance, Argument, Relation 
from reddit.utterances import RedditScraper

class cleaner:

    def __init__(self):

        pass

    def clean_utterance(self, utterance: Utterance) -> Utterance:
        
        
        return utterance
    
if __name__ == "__main__":

    utterances = 