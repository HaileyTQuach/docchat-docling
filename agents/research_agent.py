from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, List
from langchain.schema import Document
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class ResearchAgent:
    def __init__(self):
        """Initialize the research agent with the OpenAI model."""
        self.llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0.3,
            api_key=settings.OPENAI_API_KEY  # Pass the API key here
        )
        self.prompt = ChatPromptTemplate.from_template(
            """Answer the following question based on the provided context. Be precise and factual.
            
            Question: {question}
            
            Context:
            {context}
            
            If the context is insufficient, respond with: "I cannot answer this question based on the provided documents."
            """
        )

        self.prompt_feedback = ChatPromptTemplate.from_template(
        """ Answer the following question based on the provided context. Be precise and factual. Your previous attempt to answer the user's question failed verification. Your task is to generate a new, corrected response based on the feedback provided.

            Feedback : {feedback}
            Question : {question}
            Context : {context}
            
            If the context is insufficient, respond with: "I cannot answer this question based on the provided documents."
            """
        )
        
    def generate(self, question: str, documents: List[Document], verification_report : str) -> Dict:
        """Generate an initial answer using the provided documents."""
        context = "\n\n".join([doc.page_content for doc in documents])
        
        try:
            if verification_report == "":
                chain = self.prompt | self.llm | StrOutputParser()
                answer = chain.invoke({
                    "question" : question,
                    "context" : context
                })
            else:
                chain = self.prompt_feedback | self.llm | StrOutputParser()
                answer = chain.invoke({
                    "feedback" : verification_report,
                    "question" : question,
                    "context" : context
                })
            logger.info(f"Generated answer: {answer}")
            logger.info(f"Context used: {context}")
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
        
        return {
            "draft_answer": answer,
            "context_used": context
        }