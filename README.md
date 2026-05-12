# 📚 Paper Reader Agent

An agentic AI system that reads, analyzes, and summarizes research papers using **Retrieval-Augmented Generation (RAG)**. Built with LangChain, ChromaDB, and your choice of LLM provider.

> **Ask questions about any research paper — the agent searches arXiv, downloads PDFs, indexes them into a vector database, and retrieves relevant sections to answer your queries with cited sources.**

---

## Architecture

```
User Query → LangChain Agent → Decides which tool to use
                                    ↓
                    ┌───────────────────────────────────┐
                    │          Agent Tools               │
                    ├───────────────────────────────────┤
                    │ 🔍 search_arxiv_papers            │
                    │ 📥 load_and_index_paper           │
                    │ 📄 list_local_papers              │
                    │ 📂 load_local_pdf_and_index       │
                    │ 🔎 retrieve_from_paper            │
                    └──────────┬────────────────────────┘
                               ↓
                    ┌──────────────────────┐
                    │   RAG Pipeline       │
                    │                      │
                    │  PDF → Chunks →      │
                    │  Embeddings →        │
                    │  ChromaDB →          │
                    │  Semantic Search     │
                    └──────────────────────┘
```

## Features

- **ArXiv Search** — Find papers by topic, keyword, or author directly from the CLI
- **Auto-Download** — Fetches PDFs from arXiv by paper ID
- **RAG Pipeline** — Chunks papers, embeds them with Gemini, stores in ChromaDB for semantic retrieval
- **Source Citations** — Every answer cites the page number and source file
- **Local PDF Support** — Load and analyze your own PDFs from the `papers/` directory
- **Streaming Output** — See the agent's reasoning in real-time (tool calls, results, final answer)
- **Multi-Provider** — Supports Ollama (local), Gemini, OpenAI, and Anthropic

---

## Project Structure

```
paper-reader-agent/
├── app/
│   ├── __init__.py         # Package init
│   ├── config.py           # API keys, model settings, paths
│   ├── loader.py           # PDF loading + arXiv downloading
│   ├── indexer.py          # Chunking + embedding + ChromaDB storage
│   ├── tools.py            # Agent tools (@tool decorated functions)
│   ├── agent.py            # LangChain agent wiring (LLM + tools + prompt)
│   └── prompts.py          # System prompt defining agent behavior
├── papers/                 # Downloaded/local PDFs
├── chroma_db/              # Persisted vector store (auto-created)
├── summaries/              # Generated markdown summaries
├── main.py                 # CLI entry point with conversation loop
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed to git)
└── README.md
```

---

## Quick Start

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd paper-reader-agent
python -m venv venv
source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 2. Configure LLM Provider

Edit `app/config.py` — pick one:

| Provider | Cost | Setup |
|----------|------|-------|
| **Ollama** (recommended) | Free, unlimited | Install [Ollama](https://ollama.com), run `ollama pull llama3.1:8b` |
| **Google Gemini** | Free tier (limited) | Get API key from [AI Studio](https://aistudio.google.com/api-keys) |
| **OpenAI** | Paid | Set `OPENAI_API_KEY` in `.env` |
| **Anthropic** | Paid | Set `ANTHROPIC_API_KEY` in `.env` |

#### Option A: Ollama (Local — default)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Download model
ollama serve          # start the server
ollama pull llama3.1:8b
```

No config changes needed — Ollama is the default.

#### Option B: Google Gemini

```bash
# Add your key to .env
echo 'GOOGLE_API_KEY=your-key-here' > .env
```

Then edit `app/config.py`:
```python
LLM_PROVIDER = "google_genai"
LLM_MODEL = "gemini-2.0-flash"
```

### 3. Set up Embeddings API Key

Embeddings always use Gemini (free tier is very generous). You need a `GOOGLE_API_KEY` in `.env`:

```bash
echo 'GOOGLE_API_KEY=your-key-here' > .env
```

### 4. Run

```bash
python main.py
```

---

## Usage Examples

### Search for papers
```
You → Find recent papers on vision transformers for medical imaging
```

### Analyze a paper by arXiv ID
```
You → Load and analyze paper 1706.03762
```

### Ask questions about a loaded paper
```
You → What loss function did they use and why?
You → What are the main results in Table 2?
You → What are the limitations of this approach?
```

### Work with local PDFs
```
You → List local papers
You → Load and index the YOLOv8 paper
```

### Session commands
```
clear   — Reset conversation history
quit    — Exit the agent
```

---

## How It Works

### The RAG Pipeline

1. **Load** — `PyPDFLoader` extracts text from PDF pages
2. **Chunk** — `RecursiveCharacterTextSplitter` splits text into ~1500-char chunks with 200-char overlap
3. **Embed** — Gemini `gemini-embedding-001` converts chunks to high-dimensional vectors
4. **Store** — ChromaDB persists vectors to disk for fast semantic search
5. **Retrieve** — On query, finds the 6 most semantically similar chunks
6. **Generate** — LLM reads retrieved chunks + your question → produces cited answer

### The Agent Loop

The LangChain agent autonomously decides which tools to call:

```
User: "What is YOLOv8's architecture?"
  → Agent thinks: "I need to search indexed papers"
  → Calls: retrieve_from_paper("YOLOv8 architecture backbone")
  → Gets: 6 relevant chunks with page numbers
  → Generates: Structured answer with citations
```

---

## Configuration Reference

### `app/config.py`

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_PROVIDER` | `"ollama"` | LLM backend: `ollama`, `google_genai`, `openai`, `anthropic` |
| `LLM_MODEL` | `"llama3.1:8b"` | Model name for the chosen provider |
| `EMBEDDING_MODEL` | `"models/gemini-embedding-001"` | Embedding model (always Gemini) |
| `CHUNK_SIZE` | `1500` | Characters per text chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `RETRIEVAL_K` | `6` | Number of chunks retrieved per query |

### Gemini Free Tier Limits

| Model | Requests/Day | Notes |
|-------|-------------|-------|
| `gemini-2.0-flash` | 1500 | Best free option |
| `gemini-2.5-flash` | 20 | Burns fast with agents |
| `gemini-2.5-pro` | 5 | Unusable on free tier |

---

## Debugging

The CLI shows the agent's internal reasoning:

```
🔧 Tool Call: search_arxiv_papers({'query': 'YOLOv8'})
📨 Tool Result: 1. **What is YOLOv8?** ...
🔧 Tool Call: retrieve_from_paper({'query': 'architecture'})
📨 Tool Result: **[Page 3 | paper.pdf]** The backbone uses...

Agent → YOLOv8 uses a CSPDarknet53 backbone (Page 3)...
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Agent Framework** | LangChain + LangGraph |
| **LLM** | Ollama / Gemini / OpenAI / Anthropic |
| **Embeddings** | Google Gemini Embedding |
| **Vector Store** | ChromaDB (local, persisted) |
| **PDF Parsing** | PyPDF |
| **Paper Source** | arXiv API |
| **CLI Interface** | Rich (markdown rendering) |

---

## License

MIT
