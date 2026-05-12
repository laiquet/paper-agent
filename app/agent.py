'''
Create and configure the LangChain agent.
'''
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent, AgentState
from app.config import LLM_PROVIDER, LLM_MODEL
from app.tools import (
    retrieve_from_paper,
    search_arxiv_papers,
    load_and_index_paper,
    load_local_pdf_and_index,
    list_local_papers,
)
from app.prompts import SYSTEM_PROMPT

def get_llm():
    '''
    Initialize the LLM based on configured provider.
    '''
    provider_map = {
        "google_genai": f"google_genai:{LLM_MODEL}",
        "openai": LLM_MODEL,
        "anthropic": LLM_MODEL,
        "ollama": f"ollama:{LLM_MODEL}",
    }
    model_str = provider_map.get(LLM_PROVIDER, LLM_MODEL)
    return init_chat_model(model_str, temperature=0.2) # We want this LOW because: 
                                                       # - Summarizing papers needs ACCURACY, not creativity
                                                       # - We want consistent, reproducible analysis
                                                       # - The facts are in the retrieved chunks — don't improvise

def create_paper_agent():
    '''
    Create the agentic RAG paper analyzer.
    Returns a compiled agent that can be invoked with messages.
    '''
    llm = get_llm()

    tools = [
        retrieve_from_paper,
        search_arxiv_papers,
        load_and_index_paper,
        list_local_papers,
        load_local_pdf_and_index,
    ]

    agent = create_agent (
        llm,
        tools,
        system_prompt=SYSTEM_PROMPT,
    )

    return agent