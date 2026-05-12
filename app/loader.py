'''
    Load research papers from local files or arXiv
'''
import os
import re
import time
import arxiv
from langchain_community.document_loaders import PyPDFLoader
from app.config import PAPERS_DIR


def _get_arxiv_client():
    """Create an arXiv client with rate-limit-safe settings."""
    return arxiv.Client(
        page_size=10,        # fetch only 10 per API call (default was 100!)
        delay_seconds=5,     # wait 5 seconds between API calls
        num_retries=5,       # retry up to 5 times on failure
    )


def load_pdf(file_path: str) -> list:
    '''
        Load a local pdf and return LangChain Document objects.
    '''
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Enrich metadata
    for doc in docs:
        doc.metadata["source_file"] = os.path.basename(file_path)

    print(f'✅ Loaded {len(docs)} pages from {os.path.basename(file_path)}')
    return docs


def download_arxiv_paper(paper_id: str) -> str:
    '''
        Download a paper from arXiv by ID
        Returns the local file path of the downloaded PDF
    '''
    # Strip version suffix (e.g., "2501.13400v4" -> "2501.13400")
    paper_id = re.sub(r'v\d+$', '', paper_id)

    client = _get_arxiv_client()
    search = arxiv.Search(id_list=[paper_id])
    paper = next(client.results(search))

    # Sanitize title for filename
    safe_title = re.sub(r'[^\w\s-]', '', paper.title)[:80].strip()
    filename = f"{paper_id.replace('/', '_')}_{safe_title}.pdf"

    filepath = os.path.join(PAPERS_DIR, filename)

    if not os.path.exists(filepath):
        # Extra delay before download to avoid 429
        time.sleep(3)
        paper.download_pdf(dirpath=PAPERS_DIR, filename=filename)
        print(f'✅ Downloaded: {paper.title}')
    else:
        print(f'📄 Already exists: {filename}')

    return filepath


def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    '''
        Search arXiv for papers matching a query
        Returns a list of paper metadata dicts
    '''
    client = _get_arxiv_client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    results = []
    try:
        for paper in client.results(search):
            results.append({
                    "id": paper.entry_id.split("/abs/")[-1],
                    "title": paper.title,
                    "authors": [a.name for a in paper.authors[:5]],
                    "published": paper.published.strftime("%Y-%m-%d"),
                    "summary": paper.summary[:300] + "....",
                    "pdf_url": paper.pdf_url,
            })
    except arxiv.HTTPError:
        return [{"error": "arXiv rate limited. Please wait 30 seconds and try again."}]

    return results