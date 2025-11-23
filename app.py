import streamlit as st
from htbuilder.units import rem
from htbuilder import div, styles


st.set_page_config(page_title="Argument Framework Engine", page_icon="⚖️", layout="wide")


st.html(
    """

<style>

    @import url('https://fonts.googleapis.com/css2?family=Lexend:wght@100..900&display=swap');
    .text {
        text-align: center;
        font-size: calc(2rem + 1.5vw);
        max-width: 100%;
        min-width: 100%;
        line-height: 1.6;
        max-width: 5;
        font-family: 'Lexend', serif;
    }
</style>

<div class="text"><span style="font-size: calc(4rem + 2vw); color: #8C04FC;">❉</span> Argument Framework Engine
</div>


"""
)

reddit_threads = {
    "What are the best activities to do for a High School Comp Sci Club?": "https://www.reddit.com/r/learnprogramming/comments/6zftmb/what_are_some_activities_that_i_can_do_for_my/",
    "Learn CPP or Rust": "https://www.reddit.com/r/learnprogramming/comments/1oeu2qy/learn_c_or_rust/",
    "Are Zombies Human?": "https://www.reddit.com/r/arguments/comments/fklov0/are_zombies_human/",
    "How to deal with outrageous motions?": "https://www.reddit.com/r/Debate/comments/1ozh22a/how_to_deal_with_outrageous_motions/",
    "How to open a club?": "https://www.reddit.com/r/Debate/comments/1p20bto/how_to_open_club/",
    "How to deal with disrespectful opponents?": "https://www.reddit.com/r/Debate/comments/1oh5mdy/how_to_deal_with_disrespectful_opponents/",
    "Starting a CS club in Highschool": "https://www.reddit.com/r/compsci/comments/q3x6il/starting_a_cs_club_in_highschool/",
}


ignore, mid, ignore1 = st.columns([2, 4, 2])

with mid:
    pchoice = st.pills("Curated Arguments", reddit_threads.keys())

    st.markdown("""<h2 style="text-align: center;">OR</h2>""", unsafe_allow_html=True)

    tchoice = st.text_input("Enter a Reddit Thread URL", key="reddit_url_input")


if pchoice:

    st.session_state["reddit_url"] = reddit_threads[pchoice]

if tchoice:

    st.session_state["reddit_url"] = tchoice

from scraping.reddit.utterances import RedditScraper
from arguments.to_argument import ArgumentCleaner
from relations.to_relation import ToRelation

if "reddit_url" in st.session_state:

    reddit_url = st.session_state["reddit_url"]

    cleaner = ArgumentCleaner()
    utteranceClient = RedditScraper()

    post_data = utteranceClient.process_post(reddit_url)

    arguments = cleaner.convert_utterances_to_arguments(
        post_data["utterances"], threshold=0.67, utterance_data=post_data
    )

    to_relation = ToRelation(arguments)
    relations = to_relation.extract_relations(to_json=True)

    st.json(relations)
