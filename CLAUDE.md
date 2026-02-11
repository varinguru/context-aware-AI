# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Context-Aware AI Application** built with the LlamaIndex framework that performs semantic search and question answering on indexed documents. The application uses LlamaCloud for managed document indexing and Google Gemini (Gemini 2.5 Flash) as the LLM.

## Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Required Environment Variables

Create a `.env` file in the project root with:

```
GEMINI_API_KEY=<your-google-gemini-api-key>
LLAMACLOUD_API_KEY=<your-llamacloud-api-key>
```

- **GEMINI_API_KEY**: Obtain from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **LLAMACLOUD_API_KEY**: Obtain from [LlamaCloud](https://cloud.llamaindex.ai/)

## Architecture

### Core Components

```
main.py
├── Environment Configuration (dotenv)
├── LLM Configuration (GoogleGenAI)
├── Index Initialization (LlamaCloudIndex)
└── Query Processing Pipeline
    ├── Retrieval (index.as_retriever())
    └── Generation (index.as_query_engine())
```

### Key Dependencies

- **llama-index-indices-managed-llama-cloud**: Managed cloud index for document storage and retrieval
- **llama-index-llms-google-genai**: Google Gemini LLM integration
- **llama-index-core**: Core LlamaIndex framework including Settings
- **python-dotenv**: Environment variable management

### Query Processing Flow

The application follows a two-stage RAG (Retrieval-Augmented Generation) pattern:

1. **Retrieval Stage**: Uses `index.as_retriever().retrieve(query)` to fetch relevant document nodes from LlamaCloud based on semantic similarity
2. **Generation Stage**: Uses `index.as_query_engine().query(query)` to generate a response using the retrieved context

### LLM Configuration Pattern

The LLM is configured globally via `Settings.llm` (see [main.py:12-16](main.py#L12-L16)):

```python
llm = GoogleGenAI(api_key=os.getenv("GEMINI_API_KEY"), model="models/gemini-2.5-flash")
Settings.llm = llm
```

This global setting is used by the query engine for response generation.

### LlamaCloud Index

The index is initialized with a specific index name (see [main.py:20-23](main.py#L20-L23)):

```python
index = LlamaCloudIndex(name="primary-swordtail-2026-02-10", api_key=os.getenv("LLAMACLOUD_API_KEY"))
```

The index name corresponds to a pre-existing index in LlamaCloud containing the documents to be queried.

### Node Metadata

Retrieved nodes contain useful metadata accessible via `node.node.metadata`:
- `file_name`: Source document file name
- `page_label`: Page number in the source document
- Other custom metadata fields depending on indexing configuration
