'''
    Define tools that the agent can use.
'''
from langchain_core.tools import tool
from app.loader import search_arxiv, download_arxiv_paper, load_pdf
from app.indexer import index_documents, load_vector_store
from app.config import RETRIVEL_K

@tool(response_format="content_and_artifact")
def retrieve_from_paper(query: str):
    '''
    Search the indexed research papers for information relevant to the entry.
    Use this tool to find specific details, methods, results, or claims
    from papers that have been loaded into the system.
    '''
    vector_store = load_vector_store()
    retrieved_docs = vector_store.similarity_search(query, k=RETRIVEL_K)

    # Format for the LLM
    serialized = "\n\n---\n\n".join(
        f"**[Page {doc.metadata.get('page', '?')} | {doc.metadata.get('source_file', 'unknown')}]**\n{doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

@tool
def search_arxiv_papers(query: str) -> str:
    '''
    Search arXiv for research papers matching a topic or keyword
    Returns titles, authors, dates, and brief summaries of matching papers.
    Use this when the user wants to find papers on a topic.
    '''
    results = search_arxiv(query, max_results=5)

    if not results:
        return "No papers found for that query."
    
    # Handle rate-limit error from arXiv
    if "error" in results[0]:
        return results[0]["error"]
    
    output = []
    for i, r in enumerate(results, 1):
        output.append(
            f"{i}. **{r['title']}**\n"
            f"   - ID: {r['id']}\n"
            f"   - Authors: {', '.join(r['authors'])}\n"
            f"   - Published: {r['published']}\n"
            f"   - Summary: {r['summary']}\n"
        )
    return "\n".join(output)

@tool
def load_and_index_paper(arxiv_id: str) -> str:
    '''
    Download a paper from arXiv by its ID and index it into the vector store
    for retrievel. Use this when the user wants to analyze a specific paper.
    The arxiv_id should be just the ID like "2301.00234" without the full URL.
    '''
    # Download
    filepath = download_arxiv_paper(arxiv_id)

    # Load PDF
    docs = load_pdf(filepath)

    # Index into vector store
    index_documents(docs)

    return f"Successfully loaded and indexed paper {arxiv_id} ({len(docs)} pages)."

@tool
def list_local_papers() -> str:
    '''
    List all PDF files in the local papers/ directory.
    Use this to see what papers are available before loading one.
    '''
    import os
    from app.config import PAPERS_DIR
    files = [f for f in os.listdir(PAPERS_DIR) if f.endswith('.pdf')]
    if not files:
        return "No PDF files found in the papers/ directory."
    output = "Available papers in ./papers/:\n"
    for i, f in enumerate(files, 1):
        output += f"  {i}. {f}\n"
    return output

@tool
def load_local_pdf_and_index(file_name: str) -> str:
    '''
    Load a local PDF file and index it into the vector store for retrieval.
    Use list_local_papers first to see available files.
    The file_name should be just the filename (e.g. "paper.pdf"), NOT a full path.
    The file must exist in the ./papers/ directory.
    '''
    import os
    from app.config import PAPERS_DIR

    filepath = os.path.join(PAPERS_DIR, file_name)

    if not os.path.exists(filepath):
        available = [f for f in os.listdir(PAPERS_DIR) if f.endswith('.pdf')]
        return (
            f"File '{file_name}' not found in {PAPERS_DIR}/.\n"
            f"Available files: {available}"
        )

    docs = load_pdf(filepath)
    index_documents(docs)
    return f"Successfully loaded and indexed {file_name} ({len(docs)} pages)."