from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials, APIClient
from config.settings import settings
import re
import logging

logger = logging.getLogger(__name__)

credentials = Credentials(
                   url = "https://us-south.ml.cloud.ibm.com",
                  )
client = APIClient(credentials)

class RelevanceChecker:
    def __init__(self):
        # TODO
        pass

    def check(self, question: str, retriever, k=3) -> str:
        # TODO
        pass