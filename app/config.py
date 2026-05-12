import os 
from dotenv import load_dotenv

load_dotenv()

# ========================== LLM Settings ==========================
# --- OPTION 1: Ollama (LOCAL — free, unlimited, no API key needed) ---
# Prerequisites: Install Ollama (https://ollama.com), then run:
#   ollama serve
#   ollama pull llama3.1:8b
LLM_PROVIDER = "ollama"
LLM_MODEL = "llama3.1:8b"

# --- OPTION 2: Google Gemini (API — requires GOOGLE_API_KEY in .env) ---
# Free tier limits per day:
#   gemini-2.0-flash  → 1500 req/day  (best free option)
#   gemini-2.5-flash  → 20 req/day    (burns fast with agents)
#   gemini-2.5-pro    → 5 req/day     (basically unusable free)
# To use Gemini, uncomment these and comment out the Ollama lines above:
# LLM_PROVIDER = "google_genai"
# LLM_MODEL = "gemini-2.0-flash"

# --- OPTION 3: OpenAI (API — requires OPENAI_API_KEY in .env) ---
# LLM_PROVIDER = "openai"
# LLM_MODEL = "gpt-4o"

# --- OPTION 4: Anthropic (API — requires ANTHROPIC_API_KEY in .env) ---
# LLM_PROVIDER = "anthropic"
# LLM_MODEL = "claude-sonnet-4-6"

# Embeddings (keep Gemini — free tier is very generous for embeddings)
EMBEDDING_MODEL = "models/gemini-embedding-001"

# Chunking settings
'''
    Why 1500 character chunks? Research papers have dense, multi-paragraph arguments. 
    Smaller chunks (500 chars) lose context. Larger chunks (3000+) dilute retrieval precision. 
    1500 is a good starting point — benchmark and adjust based on your results.
'''
CHUNK_SIZE = 1500 # characters (~375) tunned for academic papers
CHUNK_OVERLAP = 200 # Characters of overlap between chunks

# Paths
PAPERS_DIR = "./papers"
CHROMA_DIR = "./chroma_db"
SUMMARIES_DIR = "./summaries"

# Retrival
RETRIVEL_K = 6

# Create directories
for d in [PAPERS_DIR, CHROMA_DIR, SUMMARIES_DIR]:
    os.makedirs(d, exist_ok=True)