from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict, total=False):
    rumor_text: str
    rumor_image: Optional[str]

    # --- OOC (uniquement les scores et entités)
    current_date: str
    classe_ooc: str
    s_sem_ooc: Optional[float]
    s_ent_ooc: Optional[float]
    entites_rumeur_ooc: Dict[str, List[str]]
    entites_dominantes_ooc: Dict[str, List[str]]
    comparaison_entites_ooc: Dict[str, Any]
    nombre_resultats: int
    resultats: List[Dict[str, Any]]

    # --- Cohérence visuelle
    classe_visual: Optional[str]
    s_visual: Optional[float]
    explication_visual: Optional[str]

    # --- Justification et rapport final (Nœud 2)
    justification_json: Dict[str, Any]

    # --- Métriques de performance
    # Chaque branche parallèle (node_recherche_origine, node_comparaison_visuelle)
    # écrit sa propre clé de timing — jamais la même clé, pour éviter tout conflit
    # d'écriture concurrente sur le state LangGraph (voir nodes.py). node_2 les lit
    # toutes les deux pour construire "performance_metrics".
    node_recherche_origine_time_seconds: Optional[float]
    node_comparaison_visuelle_time_seconds: Optional[float]
    performance_metrics: Dict[str, Any]