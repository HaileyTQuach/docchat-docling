from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class RetrieverBuilder:
    def __init__(self):
        # TODO
        pass