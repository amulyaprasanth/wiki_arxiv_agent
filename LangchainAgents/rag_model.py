from io import BytesIO
from typing import Any, List
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
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
    pdf_stream = BytesIO(pdf_bytestream)
    pdf_reader = PdfReader(pdf_stream)
    pages = [Document(page_content=page.extract_text(), metadata={"source": "uploaded pdf"}) for page in pdf_reader.pages]
    return pages

def split_and_store_vectors(pages: List):
    """Split the pages into chunks and store them as vectors.

    Args:
        pages (List): A list of pages to be processed.

    Returns:
        Chroma: A vector store containing the processed vectors.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    split_docs = text_splitter.split_documents(pages)

    embeddings = OllamaEmbeddings(model="phi")
    vector_store = FAISS.from_documents(split_docs, embeddings)

    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

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


def create_and_invoke_chain(input: str, retriever) -> Any:
    """Create and invoke a processing chain using a language model and a vector store.

    Args:
        input (str): A string representing the query or question to be processed.
        vector_store (Chroma): The vector store to be used in the chain.

    Returns:
        Any: The result of the chain invocation, typically an answer to the input query.
    """

    
    # Create a document processing chain using a language model and a prompt
    combine_documents_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    # Combine the retriever and document processing chain into a retrieval chain
    chain = create_retrieval_chain(retriever, combine_documents_chain)
    
    # Invoke the combined chain with the input query and return the result
<<<<<<< HEAD
    return chain.invoke({"input": input})["answer"]

if __name__ == '__main__':
    with open("../NadaganiAmulyaPrasanthResume_2.pdf", "rb") as f:
        pdf_bytes = f.read()

    docs = load_pdf(pdf_bytes)
    vector_store = split_and_store_vectors(docs)
    result = create_and_invoke_chain("What is the Experience?", vector_store)
    print(result)
=======
    return chain.invoke(input=input)
>>>>>>> b2b4a215e9b62499d70321b3d3631daed5e5d022
