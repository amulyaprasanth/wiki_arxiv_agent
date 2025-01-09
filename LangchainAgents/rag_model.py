import chromadb
from io import BytesIO
from typing import Any, List
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

def load_pdf(pdf_bytestream) -> List:
    """Load a PDF file and return its pages.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        List: A list of pages from the PDF.
    """
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    pdf_stream = BytesIO(pdf_bytestream)
    pdf_reader = PdfReader(pdf_stream)
    pages = [Document(page_content=page.extract_text(), metadata={"source": "uploaded pdf"}) for page in pdf_reader.pages]
    return pages

async def split_and_store_vectors(pages: List) -> Chroma:
    """Split the pages into chunks and store them as vectors.

    Args:
        pages (List): A list of pages to be processed.

    Returns:
        Chroma: A vector store containing the processed vectors.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(pages)

    embeddings = OllamaEmbeddings(model="phi")
    vector_store = Chroma.from_documents(split_docs, embeddings)

    return vector_store

async def read_and_store_pdf(file_path: str) -> Chroma:
    """Read a PDF file, process its pages, and store the resulting vectors.

    Args:
        file_path (str): Path to the PDF file to be processed.

    Returns:
        Chroma: The processed and stored vectors from the PDF pages.
    """
    pages = load_pdf(file_path)
    return await split_and_store_vectors(pages)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

llm = ChatOllama(model="phi")

def create_and_invoke_chain(input: str, vector_store: Chroma) -> Any:
    """Create and invoke a processing chain using a language model and a vector store.

    Args:
        input (str): A string representing the query or question to be processed.
        vector_store (Chroma): The vector store to be used in the chain.

    Returns:
        Any: The result of the chain invocation, typically an answer to the input query.
    """
    # Convert vector store to a retriever
    retriever = vector_store.as_retriever()
    
    # Create a document processing chain using a language model and a prompt
    combine_documents_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    
    # Combine the retriever and document processing chain into a retrieval chain
    chain = create_retrieval_chain(retriever, combine_documents_chain)
    
    # Invoke the combined chain with the input query and return the result
    return chain.invoke(input=input)
