'''
    Chunk, embed, and store documents in a vector database.
'''
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from app.config import CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_DIR, EMBEDDING_MODEL, LLM_PROVIDER

def get_embedding():
    '''
        Initialize the embedding model.
        Always uses Gemini for embeddings — it's free, fast, and separate from the LLM provider.
        The LLM (Ollama/GPT/Claude) does reasoning. Embeddings are a different service.
    '''
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    
def get_text_splitter():
    '''
        Create a text splitter tuned for academic papers.
    '''
    return RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        # These separators respect paper structure
        separators=[
            "\n\n\n", # Major section break
            "\n\n", # Paragraph breaks
            "\n", # Line breaks
            ". ", # Sentence boundaries
            " ", # Words
            "" # Characters (fallback)
        ],
        length_function = len,
    )

def index_documents (docs: list, collection_name: str = "papers") -> Chroma:
    '''
        Split documents into chunks, embed them, and store in ChromaDB.

        Args:
            docs: List of LangChain Document objects (from loader)
            collection_name: Name for the ChromaDB collection
        
        Returns:
            Chroma vector store instance
    '''

    # 1. Split into chunks
    splitter = get_text_splitter()
    chunks = splitter.split_documents(docs)
    print(f'📦 Split into {len(chunks)} chunks')

    # 2. Embed and store
    embeddings = get_embedding()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=collection_name,
    )
    print(f"✅ Indexed {len(chunks)} chunks into ChromaDB")

    return vector_store

def load_vector_store(collection_name: str = "papers") -> Chroma:
    '''
        Load an existing vector store from disk
    '''
    embeddings = get_embedding()
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=collection_name
    )

'''
    What's happening under the hood:

    1. Splitter breaks a 20-page paper into ~100-200 chunks of ~1500 chars each
    2. Embeddings convert each chunk into a high-dimensional vector (e.g., 768 or 3072 dimensions)
    3. ChromaDB stores these vectors on disk so you can search them semantically later
    
    Used Method - The retriever finds the 6 most semantically similar chunks — NOT keyword matching, but meaning matching.
'''