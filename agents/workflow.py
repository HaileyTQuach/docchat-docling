from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
from .research_agent import ResearchAgent
from .verification_agent import VerificationAgent
from .relevance_checker import RelevanceChecker
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
import logging

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    question: str
    documents: List[Document]
    draft_answer: str
    verification_report: str
    is_relevant: bool
    retriever: EnsembleRetriever

class AgentWorkflow:
    def __init__(self):
        # TODO
        pass