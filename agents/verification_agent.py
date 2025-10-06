import json  # Import for JSON serialization
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials, APIClient
from typing import Dict, List
from langchain.schema import Document

credentials = Credentials(
                   url = "https://us-south.ml.cloud.ibm.com",
                  )
client = APIClient(credentials)

class VerificationAgent:
    def __init__(self):
        """
        Initialize the verification agent with the IBM WatsonX ModelInference.
        """
        # Initialize the WatsonX ModelInference
        print("Initializing VerificationAgent with IBM WatsonX ModelInference...")
        self.model = ModelInference(
            model_id="ibm/granite-4-h-small", 
            credentials=credentials,
            project_id="skills-network",
            params={
                "max_tokens": 200,            # Adjust based on desired response length
                "temperature": 0.0,           # Remove randomness for consistency
            }
        )
        print("ModelInference initialized successfully.")

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
        - Respond in the exact format specified below without adding any unrelated information.

        **Format:**
        Supported: YES/NO
        Unsupported Claims: [item1, item2, ...]
        Contradictions: [item1, item2, ...]
        Relevant: YES/NO
        Additional Details: [Any extra information or explanations]

        **Answer:** {answer}
        **Context:**
        {context}

        **Respond ONLY with the above format.**
        """
        return prompt

    def parse_verification_response(self, response_text: str) -> Dict:
        """
        Parse the LLM's verification response into a structured dictionary.
        """
        try:
            lines = response_text.split('\n')
            verification = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().capitalize()
                    value = value.strip()
                    if key in {"Supported", "Unsupported claims", "Contradictions", "Relevant", "Additional details"}:
                        if key in {"Unsupported claims", "Contradictions"}:
                            # Convert string list to actual list
                            if value.startswith('[') and value.endswith(']'):
                                items = value[1:-1].split(',')
                                # Remove any surrounding quotes and whitespace
                                items = [item.strip().strip('"').strip("'") for item in items if item.strip()]
                                verification[key] = items
                            else:
                                verification[key] = []
                        elif key == "Additional details":
                            verification[key] = value
                        else:
                            verification[key] = value.upper()
            # Ensure all keys are present
            for key in ["Supported", "Unsupported Claims", "Contradictions", "Relevant", "Additional Details"]:
                if key not in verification:
                    if key in {"Unsupported Claims", "Contradictions"}:
                        verification[key] = []
                    elif key == "Additional Details":
                        verification[key] = ""
                    else:
                        verification[key] = "NO"

            return verification
        except Exception as e:
            print(f"Error parsing verification response: {e}")
            return None

    def format_verification_report(self, verification: Dict) -> str:
        """
        Format the verification report dictionary into a readable paragraph.
        """
        supported = verification.get("Supported", "NO")
        unsupported_claims = verification.get("Unsupported Claims", [])
        contradictions = verification.get("Contradictions", [])
        relevant = verification.get("Relevant", "NO")
        additional_details = verification.get("Additional Details", "")

        report = f"**Supported:** {supported}\n"
        if unsupported_claims:
            report += f"**Unsupported Claims:** {', '.join(unsupported_claims)}\n"
        else:
            report += f"**Unsupported Claims:** None\n"

        if contradictions:
            report += f"**Contradictions:** {', '.join(contradictions)}\n"
        else:
            report += f"**Contradictions:** None\n"

        report += f"**Relevant:** {relevant}\n"

        if additional_details:
            report += f"**Additional Details:** {additional_details}\n"
        else:
            report += f"**Additional Details:** None\n"

        return report

    def check(self, answer: str, documents: List[Document]) -> Dict:
        """
        Verify the answer against the provided documents.
        """
        print(f"VerificationAgent.check called with answer='{answer}' and {len(documents)} documents.")

        # Combine all document contents into one string without truncation
        context = "\n\n".join([doc.page_content for doc in documents])
        print(f"Combined context length: {len(context)} characters.")

        # Create a prompt for the LLM to verify the answer
        prompt = self.generate_prompt(answer, context)
        print("Prompt created for the LLM.")

        # Call the LLM to generate the verification report
        try:
            print("Sending prompt to the model...")
            response = self.model.chat(
                messages=[
                    {
                        "role": "user",
                        "content": prompt  # Ensure content is a string
                    }
                ]
            )
            print("LLM response received.")
        except Exception as e:
            print(f"Error during model inference: {e}")
            raise RuntimeError("Failed to verify answer due to a model error.") from e

        # Extract and process the LLM's response
        try:
            llm_response = response['choices'][0]['message']['content'].strip()
            print(f"Raw LLM response:\n{llm_response}")
        except (IndexError, KeyError) as e:
            print(f"Unexpected response structure: {e}")
            verification_report = {
                "Supported": "NO",
                "Unsupported Claims": [],
                "Contradictions": [],
                "Relevant": "NO",
                "Additional Details": "Invalid response structure from the model."
            }
            verification_report_formatted = self.format_verification_report(verification_report)
            print(f"Verification report:\n{verification_report_formatted}")
            print(f"Context used: {context}")
            return {
                "verification_report": verification_report_formatted,
                "context_used": context
            }

        # Sanitize the response
        sanitized_response = self.sanitize_response(llm_response) if llm_response else ""
        if not sanitized_response:
            print("LLM returned an empty response.")
            verification_report = {
                "Supported": "NO",
                "Unsupported Claims": [],
                "Contradictions": [],
                "Relevant": "NO",
                "Additional Details": "Empty response from the model."
            }
        else:
            # Parse the response into the expected format
            verification_report = self.parse_verification_response(sanitized_response)
            if verification_report is None:
                print("LLM did not respond with the expected format. Using default verification report.")
                verification_report = {
                    "Supported": "NO",
                    "Unsupported Claims": [],
                    "Contradictions": [],
                    "Relevant": "NO",
                    "Additional Details": "Failed to parse the model's response."
                }

        # Format the verification report into a paragraph
        verification_report_formatted = self.format_verification_report(verification_report)
        print(f"Verification report:\n{verification_report_formatted}")
        print(f"Context used: {context}")

        return {
            "verification_report": verification_report_formatted,
            "context_used": context
        }
