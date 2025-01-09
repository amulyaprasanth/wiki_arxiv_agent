import streamlit as st
import asyncio
import time
from LangchainAgents.rag_model import split_and_store_vectors, create_and_invoke_chain, load_pdf

# Give the title to our bot
st.title("RAG Application")

# Create an upload file button
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    with st.spinner("Uploading pdf"):
        # Read the uploaded file
        pages = load_pdf(uploaded_file.getvalue())

    # Display a loading spinner and measure the time taken to create vectors
    with st.spinner("Creating Embeddings..."):
        start_time = time.time()
        vectors = asyncio.run(split_and_store_vectors(pages))
        end_time = time.time()

    # Calculate the loading time
    loading_time = end_time - start_time

    # Display success message and loading time
    st.markdown(f"Loaded in {loading_time:.2f} seconds")