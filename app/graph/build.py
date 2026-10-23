from langgraph.graph import StateGraph, START, END

from app.graph.state import AgentState
from app.graph.nodes import node_recherche_origine, node_comparaison_visuelle, node_2


def compiler_graphe():
    """
    Assemble le graphe LangGraph selon le flux :

        START ─┬─→ node_recherche_origine ─┬─→ node_2 → END
               └─→ node_comparaison_visuelle ┘

    node_recherche_origine et node_comparaison_visuelle s'exécutent en
    parallèle (fan-out natif depuis START — LangGraph les lance dans le même
    super-step). node_2 ne démarre qu'une fois que les DEUX branches sont
    terminées (fan-in natif : un nœud avec plusieurs arêtes entrantes attend
    que toutes ses sources aient produit un résultat).

    Cette structure permet un suivi de progression fidèle via astream(...,
    stream_mode="updates") côté API : chaque nœud émet un événement distinct
    dès qu'il termine (voir main.py).

    Returns:
        CompiledStateGraph: Le graphe compilé prêt à être exécuté.
    """

    # 1. Initialisation du graphe
    workflow = StateGraph(AgentState)

    # 2. Enregistrement des nœuds
    workflow.add_node("node_recherche_origine", node_recherche_origine)
    workflow.add_node("node_comparaison_visuelle", node_comparaison_visuelle)
    workflow.add_node("node_2", node_2)

    # 3. Définition du flux — fan-out depuis START, fan-in vers node_2
    workflow.add_edge(START, "node_recherche_origine")
    workflow.add_edge(START, "node_comparaison_visuelle")
    workflow.add_edge("node_recherche_origine", "node_2")
    workflow.add_edge("node_comparaison_visuelle", "node_2")
    workflow.add_edge("node_2", END)

    # 4. Compilation
    graphe_compile = workflow.compile()

    return graphe_compile