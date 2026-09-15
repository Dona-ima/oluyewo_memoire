from langgraph.graph import StateGraph, START, END

from app.graph.state import AgentState
from app.graph.nodes import node_1, node_2


def compiler_graphe():
    """
    Assemble le graphe LangGraph selon le flux séquentiel :
    
        START → Nœud 1 → Nœud 2 → END
    
    Le nœud 1 réalise :
        - l'analyse du hors-contexte ;
        - l'analyse de la cohérence image–rumeur.
    
    Le nœud 2 synthétise les résultats produits par le nœud 1.
    
    Returns:
        CompiledStateGraph: Le graphe compilé prêt à être exécuté.
    """

    # 1. Initialisation du graphe
    workflow = StateGraph(AgentState)

    # 2. Enregistrement des nœuds
    workflow.add_node("node_1", node_1)
    workflow.add_node("node_2", node_2)

    # 3. Définition du flux
    workflow.add_edge(START, "node_1")
    workflow.add_edge("node_1", "node_2")
    workflow.add_edge("node_2", END)

    # 4. Compilation
    graphe_compile = workflow.compile()

    return graphe_compile