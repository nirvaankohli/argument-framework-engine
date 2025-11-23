import streamlit as st
from htbuilder.units import rem
from htbuilder import div, styles
import time
from streamlit_agraph import agraph, Node, Edge, Config
import requests
import os
from dotenv import load_dotenv
from pathlib import Path

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

<div class="text"><span style="font-size: calc(4rem + 2vw); color:#8C04FC;">❉</span> Argument Framework Engine
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

    if "current_reddit_url" not in st.session_state or st.session_state["current_reddit_url"] != reddit_url:
        st.session_state["current_reddit_url"] = reddit_url
        st.session_state["arguments"] = None
        st.session_state["relations"] = None
        st.session_state["post_metadata"] = None
        st.session_state["summary"] = None

    with mid:
        
        if not st.session_state.get("arguments"):
            with st.spinner("Initializing..."):
                cleaner = ArgumentCleaner()
                utteranceClient = RedditScraper()

            with st.spinner("Getting Post Data..."):
                st.success("Initialized!")
                post_data = utteranceClient.process_post(reddit_url)
                
                st.session_state["post_metadata"] = {
                    "subreddit": post_data.get("subreddit", "Unknown"),
                    "title": post_data.get("title", "Unknown"),
                    "author": post_data.get("utterances", [{}])[0].get("author", "Unknown") if post_data.get("utterances") else "Unknown"
                }

            with st.spinner("Converting to Arguments..."):
                st.success("Post Data Recieved!")
                arguments = cleaner.convert_utterances_to_arguments(
                    post_data["utterances"], threshold=0.67, utterance_data=post_data
                )
                st.session_state["arguments"] = arguments

            estimated_time = 1 + len(arguments) * (42/60)
            t0 = time.time()

            with st.spinner(f"Converting to Relations... Estimated Time: {estimated_time} seconds"):
                st.success("Arguments Converted!")
                to_relation = ToRelation(arguments)
                relations = to_relation.extract_relations(to_json=True)
                st.session_state["relations"] = relations
                t1 = time.time()    
                print(f"Time taken: {t1 - t0} seconds")
            
            with st.spinner("Generating summary..."):
                env_path = Path(__file__).parent / ".env"
                load_dotenv(env_path)
                api_key = os.getenv("API_KEY", "")
                
                if api_key:
                    summary_prompt = f"Summarize this Reddit discussion titled '{st.session_state['post_metadata']['title']}' from r/{st.session_state['post_metadata']['subreddit']}. Here are the main arguments:\n\n"
                    for arg in arguments[:min(20, len(arguments))]:
                        summary_prompt += f"- {arg.text}\n"
                    
                    payload = {
                        "model": "qwen/qwen3-32b",
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant that summarizes discussions. Provide a concise 2-3 sentence summary."},
                            {"role": "user", "content": summary_prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 200
                    }
                    
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    }
                    
                    try:
                        r = requests.post(
                            url="https://ai.hackclub.com/proxy/v1/chat/completions",
                            json=payload,
                            headers=headers,
                            timeout=30
                        )
                        if r.status_code == 200:
                            response_data = r.json()
                            summary = response_data["choices"][0]["message"]["content"]
                            st.session_state["summary"] = summary
                        else:
                            st.session_state["summary"] = "Summary generation failed."
                    except Exception as e:
                        st.session_state["summary"] = f"Summary error: {str(e)}"
                else:
                    st.session_state["summary"] = "API key not found."
                
        else:
            arguments = st.session_state["arguments"]
            relations = st.session_state["relations"]

        with st.spinner("Building Argument Framework..."):
            nodes = []
            edges = []
            node_ids = set()

            for relation in relations:
                if relation.from_id not in node_ids:
                    nodes.append(Node(id=relation.from_id, label=relation.from_id, size=25, shape="dot", title=relation.from_text))
                    node_ids.add(relation.from_id)
                
                if relation.to_id not in node_ids:
                    nodes.append(Node(id=relation.to_id, label=relation.to_id, size=25, shape="dot", title=relation.to_text))
                    node_ids.add(relation.to_id)
                
                color = "#00FF00" if relation.type == "supports" else "#FF0000"
                edges.append(Edge(source=relation.from_id, target=relation.to_id, color=color, label=relation.type))

            config = Config(width="100%", height=800, directed=True, physics=True, hierarchical=False)
            st.success("Argument Framework Built!")

    if st.session_state.get("post_metadata"):
        st.markdown(f"### r/{st.session_state['post_metadata']['subreddit']}")
        st.markdown(f"**Post:** {st.session_state['post_metadata']['title']}")
        st.markdown(f"**Posted by:** u/{st.session_state['post_metadata']['author']}")
        if st.session_state.get("summary"):
            st.markdown(f"**Summary:** {st.session_state['summary']}")
        st.divider()

    graph_col, chat_col = st.columns([3, 1])

    with graph_col:
        st.subheader("Interactive Argument Graph")
        selected_node_id = agraph(nodes=nodes, edges=edges, config=config)

    with chat_col:
        st.subheader("Argument Details")
        
        if selected_node_id:
            selected_text = "Unknown"
            for relation in relations:
                if relation.from_id == selected_node_id:
                    selected_text = relation.from_text
                    break
                elif relation.to_id == selected_node_id:
                    selected_text = relation.to_text
                    break
            
            st.info(f"**Node ID:** {selected_node_id}")
            st.write(f"**Content:** {selected_text}")

            if st.button("Add to Chat"):
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                
                context_parts = [f"**Main Node {selected_node_id}:** {selected_text}\n"]
                
                supporting_nodes = []
                attacking_nodes = []
                supported_by = []
                attacked_by = []
                
                for relation in relations:
                    if relation.from_id == selected_node_id:
                        if relation.type == "supports":
                            supporting_nodes.append(f"  - {relation.to_id}: {relation.to_text}")
                        else:
                            attacking_nodes.append(f"  - {relation.to_id}: {relation.to_text}")
                    
                    if relation.to_id == selected_node_id:
                        if relation.type == "supports":
                            supported_by.append(f"  - {relation.from_id}: {relation.from_text}")
                        else:
                            attacked_by.append(f"  - {relation.from_id}: {relation.from_text}")
                
                if supported_by:
                    context_parts.append(f"\n**Supported by:**\n" + "\n".join(supported_by))
                if attacked_by:
                    context_parts.append(f"\n**Attacked by:**\n" + "\n".join(attacked_by))
                if supporting_nodes:
                    context_parts.append(f"\n**Supports:**\n" + "\n".join(supporting_nodes))
                if attacking_nodes:
                    context_parts.append(f"\n**Attacks:**\n" + "\n".join(attacking_nodes))
                
                context_msg = "\n".join(context_parts)
                st.session_state.messages.append({"role": "user", "content": context_msg})
                st.success("Added node and all connected nodes to chat context!")

        else:
            st.write("Click on a node to see details.")

        st.divider()
        st.subheader("Chat with Arguments")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about the arguments..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})            

            env_path = Path(__file__).parent / ".env"
            load_dotenv(env_path)
            api_key = os.getenv("API_KEY", "")
            
            if not api_key:
                response = "Error: API_KEY not found in .env file"
            else:
                metadata_context = ""
                if st.session_state.get("post_metadata"):
                    metadata_context = f"Discussion Context: r/{st.session_state['post_metadata']['subreddit']} - '{st.session_state['post_metadata']['title']}' by u/{st.session_state['post_metadata']['author']}."
                    if st.session_state.get("summary"):
                        metadata_context += f" Summary: {st.session_state['summary']}"

                system_message = f"You are a helpful assistant analyzing an argument framework. {metadata_context} Use the provided context to answer questions."

                payload = {
                    "model": "qwen/qwen3-32b",
                    "messages": [
                        {"role": "system", "content": system_message},
                        *[{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]
                    ],
                    "temperature": 0.7,
                    "max_tokens": 14000,
                }
                            
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                }
                
                with st.spinner("Thinking..."):
                    try:
                        r = requests.post(
                            url="https://ai.hackclub.com/proxy/v1/chat/completions", 
                            json=payload, 
                            headers=headers, 
                            timeout=60
                        )
                        
                        if r.status_code == 200:
                            response_data = r.json()
                            response = response_data["choices"][0]["message"]["content"]
                        else:
                            response = f"Error: {r.status_code} - {r.text}"
                            
                    except Exception as e:
                        response = f"Error: {str(e)}"

            with st.chat_message("assistant"):
                st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})