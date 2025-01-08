from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize embeddings
embeddings = OllamaEmbeddings(model="phi")

# Function to process text and create a knowledge base
def process_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(text)
    knowledge_base = FAISS.from_documents(chunks, embeddings)
    return knowledge_base

# Function to load PDF and answer queries
def analyze_pdf(pdf_path, query):
    pdf_reader = PyPDFLoader(pdf_path)
    text = pdf_reader.load()
    knowledge_base = process_text(text)

    retriever = knowledge_base.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    llm = ChatOllama(model="phi")

    prompt_template = """
        Use the following piece of context to answer the question asked.
        Please try to provide the answer only based on the context

        {context}
        Question: {question}

        Helpful Answers:
    """

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
    )

    result = retrieval_qa.invoke({"query": query})

    return result["result"]

if __name__ == '__main__':
    print(analyze_pdf("pdfs/Invoice.pdf",  "What is the product?"))