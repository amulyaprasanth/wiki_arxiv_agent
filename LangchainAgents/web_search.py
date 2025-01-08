import asyncio
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# Initialize the ollama
llm = ChatOllama(model="phi", temperature = 0.3)

# Function to extract the text from the webpage
async def load_webpage(page_url: str):
    """
    Load and extract documents from a given webpage URL.

    Args:
        page_url (str): URL of the webpage to load.

    Returns:
        List of documents extracted from the webpage.
    """
    # Initialize the loader with the webpage URL
    loader = WebBaseLoader(web_paths=[page_url])

    # Initialize the doc variable to store the results
    docs = []

    # Load the documents asynchronously
    async for doc in loader.alazy_load():
        docs.append(doc)

    # Return the loaded documents
    return docs

def split_text_documents(docs):
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    return text_splitter.split_documents(docs)

def store_text_documents(split_docs):
    # Initialize Ollama embeddings
    embeddings = OllamaEmbeddings(model="phi")

    # Create FAISS vector store from documents
    vector_store = FAISS.from_documents(split_docs, embeddings)

    return vector_store

def query_vector_store(vector_store, query):
    # Initialize the Ollama model
    ollama = ChatOllama(model="phi")

    # Query the vector store
    results  = vector_store.similarity_search(
    query,
    k=2,
)
    return results


def create_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
            [
                    (
                            "system",
                            "You are a helpful assistant that answers the question given by the user ",
                            {context}
                    ),
                    ("human", "{input}"),
            ]
    )

    prompt = ChatPromptTemplate()

if __name__ == '__main__':
    # Web page URL to load
    page_url = "https://www.yahoo.com/tech/best-laptops-ces-2025-010326261.html"

    # Run the async function using asyncio
    documents = asyncio.run(load_webpage(page_url))

    # Split the loaded documents
    split_docs = split_text_documents(documents)

    # Store the split documents in FAISS vector store
    vector_store = store_text_documents(split_docs)

    # Query the vector store
    query = "What are the best laptops from CES 2025?"
    results = query_vector_store(vector_store, query)

    # Print the query results
    print(results)
