# Argument Framework Engine

This tool turns Reddit threads into interactive maps of arguments. It reads a discussion, breaks it down into individual points, and connects them to show who supports or attacks whom.

It helps you see the structure of a debate at a glance and tell the strongest points and how everything connects.

## What it does

*   **Reads Reddit**: You give it a link, and it pulls the entire conversation.
*   **Maps the Debate**: It draws a graph where nodes are arguments. Green lines mean support; red lines mean disagreement.
*   **Chat Assistant**: You can click on any argument to ask an AI about it. The AI sees the argument and its surrounding context (what supports it and what attacks it) to give you a better answer.

## How to run it

1.  **Install dependencies**:
    ```bash
    pip install streamlit streamlit-agraph requests python-dotenv htbuilder
    ```

2.  **Set up your key**:
    Create a `.env` file and add your API key:
    ```
    API_KEY=your_key_here
    ```

3.  **Start the app**:
    ```bash
    streamlit run app.py
    ```

## Tech Stack

Built with Python using **Streamlit** for the interface and **NetworkX** concepts for the graph logic. It uses an LLM to process the text and generate summaries.