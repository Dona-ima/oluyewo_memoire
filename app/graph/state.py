from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict, total=False):
    rumor_text: str
    rumor_image: Optional[str]

    # --- OOC
    current_date: str
    classe_ooc: Optional[str]
    s_ooc: Optional[float]
    s_sem_ooc: Optional[float]
    s_ent_ooc: Optional[float]
    entites_rumeur_ooc: Dict[str, List[str]]
    entites_dominantes_ooc: Dict[str, List[str]]
    comparaison_entites_ooc: Dict[str, Any]
    nombre_resultats: int
    resultats: List[Dict[str, Any]]

    # --- Coherence visuelle
    classe_visual: Optional[str]
    s_visual: Optional[float]


    # --- Justification et rapport final (Nœud 2)
    #verdict: Optional[str]
    justification_json: Dict[str, Any]

    # --- Métriques de performance 
    performance_metrics: Dict[str, Any]