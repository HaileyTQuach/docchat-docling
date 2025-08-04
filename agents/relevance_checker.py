
from config.settings import settings
import re
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)


genai.configure(api_key=settings.GOOGLE_API_KEY)

class RelevanceChecker:
    def __init__(self):
        # Initialize the Gemini Model
        self.model = genai.GenerativeModel('gemini-2.5-pro')

    def check(self, question: str, retriever, k=3) -> str:
        """
        1. Retrieve the top-k document chunks from the global retriever.
        2. Combine them into a single text string.
        3. Pass that text + question to the LLM for classification.

        Returns: "CAN_ANSWER", "PARTIAL", or "NO_MATCH".
        """

        logger.debug(f"RelevanceChecker.check called with question='{question}' and k={k}")

        # Retrieve doc chunks from the ensemble retriever
        top_docs = retriever.invoke(question)
        if not top_docs:
            logger.debug("No documents returned from retriever.invoke(). Classifying as NO_MATCH.")
            return "NO_MATCH"

        # Combine the top k chunk texts into one string
        document_content = "\n\n".join(doc.page_content for doc in top_docs[:k])

        # Create a prompt for the LLM to classify relevance
        prompt = f"""
        You are an AI relevance checker between a user's question and provided document content.

        **Instructions:**
        - Classify how well the document content addresses the user's question.
        - Respond with only one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH.
        - Do not include any additional text or explanation.

        **Labels:**
        1) "CAN_ANSWER": The passages contain enough explicit information to fully answer the question.
        2) "PARTIAL": The passages mention or discuss the question's topic but do not provide all the details needed for a complete answer.
        3) "NO_MATCH": The passages do not discuss or mention the question's topic at all.

        **Important:** If the passages mention or reference the topic or timeframe of the question in any way, even if incomplete, respond with "PARTIAL" instead of "NO_MATCH".

        **Question:** {question}
        **Passages:** {document_content}

        **Respond ONLY with one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH**
        """

        # Call the LLM
        try:
            response = self.model.generate_content(prompt)
            classification = response.text.strip().upper()

            # Validate the classification
            if classification in ["CAN_ANSWER", "PARTIAL", "NO_MATCH"]:
                logger.debug(f"Relevance classification: {classification}")
                return classification
            else:
                logger.warning(f"LLM returned an invalid classification: '{classification}'. Defaulting to NO_MATCH.")
                return "NO_MATCH"

        except Exception as e:
            logger.error(f"An error occurred during LLM call: {e}")
            return "NO_MATCH"
