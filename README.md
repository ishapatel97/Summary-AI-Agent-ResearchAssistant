# Summary: AI Agent ResearchAssistant

App link: https://summary-ai-agent-isha-patel.streamlit.app/

## Authentication
![Authentication](./authentication.png)

## Refine
![Refine](./refine.png)

This project is an Summary: AI Agent ResearchAssistant built with Streamlit using an AI agent utilizing the Grok API (via `langchain_groq`) for natural language processing and the Semantic Scholar API for retrieving research papers. The AI agent, based on Grok's language model (`Llama3-8b-8192`), collaborates with FAISS for efficient similarity search and SentenceTransformer embeddings to rank summaries, enabling researchers to summarize, refine, and evaluate academic papers interactively. It’s designed to streamline literature reviews and support collaborative research workflows. For enhanced performance, you can optionally use OpenAI (a paid service) instead of Grok, though the default setup with Grok is free and effective.

## Setup Instructions
1. **Prerequisites**: Ensure Python is installed on your system.
2. **Groq API Key**: Obtain a Groq API key and set it in a `.env` file as `GROQ_API_KEY=<your_key>`.
3. **Environment Setup**: 
   - Create and activate a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```
4. **Install Dependencies**: Run the following to install required packages:
     ```bash
     pip install -r requirements.txt
     ```
5. **Run the App**: Launch the Streamlit application:
     ```bash
     python -m streamlit run summary_ai_agent.py
     ```

## Features
- **Paper Search**: Fetches papers from the Semantic Scholar API based on user queries (e.g., "machine learning").
- **AI-Powered Summarization**: Generates 3-sentence summaries of paper abstracts using the Grok-based AI agent.
- **Ranking**: Ranks summaries by relevance using FAISS and SentenceTransformer embeddings (`all-MiniLM-L6-v2`).
- **Feedback & Refinement**: Enables users to refine summaries via feedback, processed by the AI agent.
- **Metrics**: Evaluates summary clarity (1-5) and provides improvement suggestions.

## How the AI Agent Works: Step-by-Step
1. **Input Query**: The user enters a research topic (e.g., "machine learning") in the Streamlit interface.
2. **Paper Retrieval**: The AI agent sends the query to the Semantic Scholar API, fetching relevant papers with titles, abstracts, and URLs.
3. **Summarization**: For each paper’s abstract, the AI agent uses a predefined prompt (`"Summarize this paper in 3 sentences"`) and the Grok model to generate concise summaries. (Optionally, switch to OpenAI for potentially better results, though it’s a paid service.)
4. **Embedding & Ranking**: The agent encodes summaries into vectors using SentenceTransformer, then uses FAISS to rank them based on similarity to the query’s embedding.
5. **Display**: Ranked summaries are shown in the Streamlit app, with clarity scores and improvement suggestions generated by the AI agent using a metrics prompt.
6. **Feedback Loop**: Users provide feedback on summaries, which the AI agent refines using a feedback prompt, updating the results interactively.
7. **Iteration**: The process repeats as users refine or explore new queries, leveraging the agent’s continuous learning capabilities.

## Usage
- Enter your Groq API key in the sidebar to activate the AI agent.
- Input a research topic, click "Search and Summarize," and review AI-generated summaries.
- Provide feedback to refine summaries and explore clarity metrics.

## Additional Notes
- The Semantic Scholar API powers paper retrieval—no extra API key needed, though rate limits apply.
- A stable internet connection is required for API calls.
- For superior summarization and refinement, consider using OpenAI (paid) instead of Grok by modifying the code to use the OpenAI API.
- Customize prompts or models in the code for advanced functionality.

This tool harnesses an AI agent to empower researchers, students, and enthusiasts with an efficient, interactive way to navigate academic literature, with the flexibility to upgrade to OpenAI for premium results!
