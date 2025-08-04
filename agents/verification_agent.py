import json
import google.generativeai as genai
from typing import Dict, List
from langchain.schema import Document
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

genai.configure(api_key=settings.GOOGLE_API_KEY)

class VerificationAgent:
    def __init__(self):
        """
        Initialize the verification agent with the Google Gemini Model.
        """
        # Initialize the Gemini Model
        print("Initializing VerificationAgent with Google Gemini Model...")
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        print("Gemini Model initialized successfully.")

    def sanitize_response(self, response_text: str) -> str:
        """
        Sanitize the LLM's response by stripping unnecessary whitespace.
        """
        return response_text.strip()

    def generate_prompt(self, answer: str, context: str) -> str:
        """
        Generate a structured prompt for the LLM to verify the answer against the context.
        """
        prompt = f"""
        You are an AI assistant designed to verify the accuracy and relevance of answers based on provided context.

        **Instructions:**
        - Verify the following answer against the provided context.
        - Check for:
        1. Direct/indirect factual support (YES/NO)
        2. Unsupported claims (list any if present)
        3. Contradictions (list any if present)
        4. Relevance to the question (YES/NO)
        - Provide additional details or explanations where relevant.
        
        **Answer to Verify:**
        {answer}
        
        **Context:**
        {context}
        """
        return prompt

    def run(self, research_result: Dict) -> Dict:
        """
        Verify the research-based answer against the provided context.
        """
        answer = research_result.get("answer", "")
        context = research_result.get("context", "")
        
        # Generate the prompt
        prompt = self.generate_prompt(answer, context)
        
        # Call the LLM
        try:
            response = self.model.generate_content(prompt)
            sanitized_response = self.sanitize_response(response.text)
            
            # Add the verification result to the research_result
            research_result["verification_result"] = sanitized_response
            return research_result
        except Exception as e:
            logger.error(f"An error occurred during LLM call in VerificationAgent: {e}")
            research_result["verification_result"] = "Error: Could not verify the answer."
            return research_result
