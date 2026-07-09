from langgraph.graph import StateGraph, START, END

from app.graph.state import AgentState
from app.graph.nodes import (
    agent_verificateur_principal,
    agent_chercheur_web,
    agent_synthetiseur_verdict
)
from app.graph.edges import routeur_fire

def compiler_graphe():
    """
    Assemble les nœuds, configure les liaisons (edges) fixes et conditionnelles,
    puis compile le graphe d'état LangGraph.
    
    Returns:
        CompiledStateGraph: Le graphe prêt à être exécuté par le backend FastAPI.
    """
    
    # 1. Initialisation du graphe avec notre structure de mémoire partagée
    workflow = StateGraph(AgentState)
    
    # 2. Enregistrement des nœuds computationnels (les Agents)
    workflow.add_node("Agent_Verificateur_Principal", agent_verificateur_principal)
    workflow.add_node("Agent_Chercheur_Web", agent_chercheur_web)
    workflow.add_node("Agent_Synthetiseur_Verdict", agent_synthetiseur_verdict)
    
    # 3. Définition des liaisons séquentielles directes (Fixed Edges)
    # Le point de départ pointe toujours vers le vérificateur principal
    workflow.add_edge(START, "Agent_Verificateur_Principal")
    
    # Le chercheur web renvoie toujours ses résultats au vérificateur principal
    workflow.add_edge("Agent_Chercheur_Web", "Agent_Verificateur_Principal")
    
    # Le synthétiseur met fin au cycle du graphe
    workflow.add_edge("Agent_Synthetiseur_Verdict", END)
    
    # 4. Configuration du Carrefour FIRE (Routage Conditionnel)
    # À la sortie du vérificateur principal, on applique la fonction de routage
    workflow.add_conditional_edges(
        "Agent_Verificateur_Principal",
        routeur_fire,
        {
            "vers_recherche": "Agent_Chercheur_Web",
            "vers_synthese": "Agent_Synthetiseur_Verdict"
        }
    )
    
    # 5. Compilation finale du flux
    graphe_compile = workflow.compile()
    return graphe_compile