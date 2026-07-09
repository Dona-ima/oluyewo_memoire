from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    rumor_text: str
    rumor_image: str
    current_date: str
    image_description: str
    lens_results: str         
    sub_questions: List[str]
    search_history: List[str]
    evidence_gathered: List[str]
    confidence_score: float
    iterations: int
    verdict: str
    justification_json: Dict[str, Any]
    performance_metrics: Dict[str, Any]