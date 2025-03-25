##python -m streamlit run summary_ai_agent.py
import faiss
import numpy as np
import requests
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st
from sentence_transformers import SentenceTransformer
import os
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("No Groq API key found in .env file!")

# Sidebar for API key input
st.sidebar.title("API Key Required")
user_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
api_key_valid = user_api_key == groq_api_key

# Load LLM only if API key is valid
if api_key_valid:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
    st.sidebar.write("LLM loaded: ", llm.invoke("Hello!").content)
    model = SentenceTransformer('all-MiniLM-L6-v2')
else:
    st.sidebar.warning("Please enter a valid Groq API key to proceed.")
    llm = None  # Prevent further execution
    model = None

def search_papers(query):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&fields=title,abstract,url"
    response = requests.get(url)
    return response.json().get("data", []) if response.status_code == 200 else []

if api_key_valid:  # Only define functions if key is valid
    summarization_prompt = PromptTemplate(input_variables=["text"], template="Summarize this paper: {text}")
    summarization_chain = summarization_prompt | llm
    feedback_prompt = PromptTemplate(input_variables=["summary", "feedback"], template="Refine this summary based on feedback: {summary}, Feedback: {feedback}")
    feedback_chain = feedback_prompt | llm
    metrics_prompt = PromptTemplate(input_variables=["summary"], template="Score this summary for clarity (1-5) and suggest improvements: {summary}")
    metrics_chain = metrics_prompt | llm

    def summarize_text(text):
        return summarization_chain.invoke({"text": text}).content

    def rank_summaries(query, papers):
        summaries = [summarize_text(paper["abstract"]) for paper in papers if paper.get("abstract")]
        if not summaries:
            return []
        summary_embeddings = model.encode(summaries)
        index = faiss.IndexFlatL2(summary_embeddings.shape[1])
        index.add(summary_embeddings)
        query_vector = model.encode([query])
        _, I = index.search(query_vector, k=3)
        return [summaries[i] for i in I[0]]

# Initialize session state
if 'papers' not in st.session_state:
    st.session_state.papers = []
if 'ranked_summaries' not in st.session_state:
    st.session_state.ranked_summaries = []
if 'refined_summaries' not in st.session_state:
    st.session_state.refined_summaries = {}

st.title("Collaborative Research Assistant")

# Only show search box if API key is valid
if api_key_valid:
    user_query = st.text_input("Research Topic", "machine learning", disabled=False)
    if st.button("Search and Summarize"):
        if user_query:
            with st.spinner("Processing..."):
                st.session_state.papers = search_papers(user_query)
                if not st.session_state.papers:
                    st.warning("No papers found.")
                else:
                    st.session_state.ranked_summaries = rank_summaries(user_query, st.session_state.papers)
                    st.session_state.refined_summaries = {}
                    if not st.session_state.ranked_summaries:
                        st.warning("No summaries generated.")
                    else:
                        st.success(f"Found {len(st.session_state.papers)} papers!")
else:
    user_query = st.text_input("Research Topic", "Enter API key first", disabled=True)
    st.button("Search and Summarize", disabled=True)

# Display summaries if they exist
if api_key_valid and st.session_state.ranked_summaries:
    for i, summary in enumerate(st.session_state.ranked_summaries, 1):
        st.subheader(f"Summary {i}")
        st.write(summary)
        metrics = metrics_chain.invoke({"summary": summary}).content
        st.write("Metrics: ", metrics)
        feedback = st.text_area(f"Feedback for Summary {i}", key=f"fb{i}")
        if st.button("Refine", key=f"ref{i}"):
            if feedback.strip():
                refined = feedback_chain.invoke({"summary": summary, "feedback": feedback}).content
                st.session_state.refined_summaries[i] = refined
                st.success("Refined!")
        if i in st.session_state.refined_summaries:
            st.write("Refined Summary: ", st.session_state.refined_summaries[i])
