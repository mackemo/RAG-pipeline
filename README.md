# PDF Question Answering with RAG

A small retrieval-augmented generation (RAG) example that reads PDF files, splits their text into chunks, creates OpenAI embeddings, and uses a FAISS vector store to answer a question with relevant source passages.

## How it works

`rag-pipeline.py` performs the full workflow:

1. Loads PDF files from `RAG-pipeline/documents/` using LangChain's `PyPDFLoader`.
2. Splits the extracted text into 500-character chunks with 50 characters of overlap.
3. Embeds the chunks with OpenAI and builds an in-memory FAISS index.
4. Asks user input: `How many seashells are at my favorite beach?` using `gpt-3.5-turbo` and prints the answer and retrieved source text.

The repository includes example PDFs in `documents/` (`Lantern.pdf`, `Seashells.pdf`, and `me.pdf`).

## Requirements

- Python 3.9 or newer
- An OpenAI API key with access to the embedding and chat models used by the script
- The Python packages imported by the script: `python-dotenv`, `langchain-community`, `langchain-text-splitters`, `langchain-openai`, `langchain-classic`, and `faiss-cpu`

Install the dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install python-dotenv langchain-community langchain-text-splitters langchain-openai langchain-classic faiss-cpu
```

## Configure and run

Create a `.env` file in the project directory and add your key:

```dotenv
OPENAI_API_KEY=your-api-key
```

Change user input if desired. Can also switch out documents for your own.

Run the script from the directory containing this repository, so its configured `RAG-pipeline/documents/` path resolves correctly:

```bash
python RAG-pipeline/rag-pipeline.py
```

The script prints the number of loaded pages and text chunks, the fixed query, the generated answer, and the text of the returned source chunks. The FAISS index is built in memory each time the script runs.

## Adding documents

Place PDF files in `documents/` and run the script again. Only PDFs directly inside that directory are loaded; nested directories are not searched.

## Notes

- API calls are made to OpenAI for both embeddings and chat completion, so usage may incur charges.
- The question and chat model are currently set directly in `rag-pipeline.py`.
- If no chunks are created, the script's average chunk-size calculation will fail; provide at least one readable PDF with extractable text.
