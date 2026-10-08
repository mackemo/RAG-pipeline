import os
import time
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

# import langchain libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


# Load and chunk docs
# ------------------------------
# load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# load in the pdf files
documents = []

pdf_folder = Path("RAG-pipeline/documents")

for pdf_file in pdf_folder.glob("*.pdf"):
    loader = PyPDFLoader(str(pdf_file))
    documents.extend(loader.load())

# split the pdf files into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# total chunks and average chunk size
total_chunks = len(docs)
average_chunk_size = sum(len(doc.page_content) for doc in docs) / total_chunks

print(f"Total chunks: {total_chunks}")
print(f"Average chunk size: {average_chunk_size:.2f} characters")



# Embed to vector store
# ------------------------------
# create embeddings for the documents
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
# initialize the FAISS vector store
vectorstore = FAISS.from_documents(docs, embeddings)



# LLM and RAG logic
# ------------------------------
# initialize the llm
MODEL_NAME = "gpt-3.5-turbo"

llm = ChatOpenAI(
    model_name=MODEL_NAME,
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
# init the rag retrieval and generator
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# ask question
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
query = "How many seashells are at the favorite beach?"
print("Query:", query)
result = qa_chain.invoke(query)


# display results
print("Answer:", result["result"])
print("\n--- Sources ---")
for i, doc in enumerate(result["source_documents"], 1):
    print(f"\nSource {i}:")
    print(doc.page_content)