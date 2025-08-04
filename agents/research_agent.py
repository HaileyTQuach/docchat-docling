import google.generativeai as genai
from typing import Dict, List
from langchain.schema import Document
from config.settings import settings
import json
import logging

logger = logging.getLogger(__name__)

genai.configure(api_key=settings.GOOGLE_API_KEY)


class ResearchAgent:
    def __init__(self):
        """
        Initialize the research agent with the Google Gemini Model.
        """
        # Initialize the Gemini Model
        print("Initializing ResearchAgent with Google Gemini Model...")
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        print("Gemini Model initialized successfully.")

    def sanitize_response(self, response_text: str) -> str:
        """
        Sanitize the LLM's response by stripping unnecessary whitespace.
        """
        return response_text.strip()

    def generate_prompt(self, question: str, context: str) -> str:
        """
        Generate a structured prompt for the LLM to generate a precise and factual answer.
        """
        prompt = f"""
        You are an AI assistant designed to provide precise and factual answers based on the given context.

        **Instructions:**
        - Answer the following question using only the provided context.
        - Be clear, concise, and factual.
        - Return as much information as you can get from the context.
        
        **Question:** {question}
        
        **Context:**
        {context}
        """
        return prompt

    def run(self, question: str, context_docs: List[Document]) -> Dict:
        """
        Generate a research-based answer using the provided context documents.
        """
        # Combine document contents into a single context string
        context_str = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Generate the prompt
        prompt = self.generate_prompt(question, context_str)
        
        # Call the LLM
        try:
            response = self.model.generate_content(prompt)
            sanitized_response = self.sanitize_response(response.text)
            
            return {
                "answer": sanitized_response,
                "context": context_str,
                "source_documents": [doc.metadata.get('source', 'N/A') for doc in context_docs]
            }
        except Exception as e:
            logger.error(f"An error occurred during LLM call in ResearchAgent: {e}")
            return {
                "answer": "Error: Could not generate an answer.",
                "context": context_str,
                "source_documents": [doc.metadata.get('source', 'N/A') for doc in context_docs]
            }
