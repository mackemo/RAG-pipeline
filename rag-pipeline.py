import os
import time
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

# import langchain libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

pdf_folder = Path("RAG-pipeline/documents")
for f in pdf_folder.iterdir():
    print(f"  | {f.name}")

