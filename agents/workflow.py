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
        self.researcher = ResearchAgent()
        self.verifier = VerificationAgent()
        self.relevance_checker = RelevanceChecker()
        self.compiled_workflow = self.build_workflow()  # Compile once during initialization
        
    def build_workflow(self):
        """Create and compile the multi-agent workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("check_relevance", self._check_relevance_step)
        workflow.add_node("research", self._research_step)
        workflow.add_node("verify", self._verification_step)
        
        # Define edges
        workflow.set_entry_point("check_relevance")
        workflow.add_conditional_edges(
            "check_relevance",
            self._decide_after_relevance_check,
            {
                "relevant": "research",
                "irrelevant": END
            }
        )
        workflow.add_edge("research", "verify")
        workflow.add_conditional_edges(
            "verify",
            self._decide_next_step,
            {
                "re_research": "research",
                "end": END
            }
        )
        return workflow.compile()
    
    def _check_relevance_step(self, state: AgentState) -> Dict:
        retriever = state["retriever"]
        classification = self.relevance_checker.check(
            question=state["question"], 
            retriever=retriever, 
            k=20
        )

        if classification == "CAN_ANSWER":
            # We have enough info to proceed
            return {"is_relevant": True}

        elif classification == "PARTIAL":
            # There's partial coverage, but we can still proceed
            return {
                "is_relevant": True
            }

        else:  # classification == "NO_MATCH"
            return {
                "is_relevant": False,
                "draft_answer": "This question isn't related (or there's no data) for your query. Please ask another question relevant to the uploaded document(s)."
            }


    def _decide_after_relevance_check(self, state: AgentState) -> str:
        decision = "relevant" if state["is_relevant"] else "irrelevant"
        print(f"[DEBUG] _decide_after_relevance_check -> {decision}")
        return decision
    
    def full_pipeline(self, question: str, retriever: EnsembleRetriever):
        try:
            print(f"[DEBUG] Starting full_pipeline with question='{question}'")
            documents = retriever.invoke(question)
            logger.info(f"Retrieved {len(documents)} relevant documents (from .invoke)")

            initial_state = AgentState(
                question=question,
                documents=documents,
                draft_answer="",
                verification_report="",
                is_relevant=False,
                retriever=retriever
            )
            
            final_state = self.compiled_workflow.invoke(initial_state)
            
            return {
                "draft_answer": final_state["draft_answer"],
                "verification_report": final_state["verification_report"]
            }
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise
    
    def _research_step(self, state: AgentState) -> Dict:
        print(f"[DEBUG] Entered _research_step with question='{state['question']}'")
        result = self.researcher.run(state["question"], state["documents"])
        print("[DEBUG] Researcher returned draft answer.")
        return {"draft_answer": result["answer"]}
    
    def _verification_step(self, state: AgentState) -> Dict:
        print("[DEBUG] Entered _verification_step. Verifying the draft answer...")
        context_str = "\n\n".join([doc.page_content for doc in state["documents"]])
        research_result = {"answer": state["draft_answer"], "context": context_str}
        result = self.verifier.run(research_result)
        print("[DEBUG] VerificationAgent returned a verification report.")
        return {"verification_report": result["verification_result"]}
    
    def _decide_next_step(self, state: AgentState) -> str:
        verification_report = state["verification_report"]
        print(f"[DEBUG] _decide_next_step with verification_report='{verification_report}'")
        if "Supported: NO" in verification_report or "Relevant: NO" in verification_report:
            logger.info("[DEBUG] Verification indicates re-research needed.")
            return "re_research"
        else:
            logger.info("[DEBUG] Verification successful, ending workflow.")
            return "end"
