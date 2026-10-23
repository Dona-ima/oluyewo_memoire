import time

from app.graph.state import AgentState
from app.graph.node_1.n1_smOOC import sous_module_ooc
from app.graph.node_1.n1_smCoherence import sous_module_paligemma
from app.graph.node_2.n2_synthese import sous_module_synthese


def node_recherche_origine(state: AgentState) -> dict:
    """
    Branche parallèle 1 : recherche l'origine de l'image sur le web (recherche
    inversée + comparaison sémantique et d'entités avec le texte de la rumeur).

    Ne retourne QUE les clés que ce nœud modifie — jamais **state. Ce nœud
    s'exécute en parallèle de node_comparaison_visuelle (fan-out depuis START,
    voir build.py) ; si les deux nœuds retournaient l'état complet, LangGraph
    lèverait une erreur de conflit d'écriture concurrente sur les clés
    communes (rumor_text, rumor_image, etc.), même si leur valeur est
    inchangée.
    """
    t_start = time.perf_counter()

    resultat_ooc = sous_module_ooc(
        url_image=state.get("rumor_image"),
        rumeur=state["rumor_text"],
        date_reference=state["current_date"]
    )

    execution_time = round(time.perf_counter() - t_start, 3)

    return {
        "classe_ooc": resultat_ooc["classe_ooc"],
        "s_sem_ooc": resultat_ooc["S_sem"],
        "s_ent_ooc": resultat_ooc["S_ent"],
        "entites_rumeur_ooc": resultat_ooc["entites_rumeur"],
        "entites_dominantes_ooc": resultat_ooc["entites_dominantes"],
        "comparaison_entites_ooc": resultat_ooc["comparaison_entites"],
        "nombre_resultats": len(resultat_ooc["resultats"]),
        "resultats": resultat_ooc["resultats"],
        "node_recherche_origine_time_seconds": execution_time,
    }


def node_comparaison_visuelle(state: AgentState) -> dict:
    """
    Branche parallèle 2 : compare le contenu visuel de l'image au texte de la
    rumeur via PaliGemma (Visual Entailment). Voir la note sur node_recherche_origine
    pour pourquoi ce nœud ne retourne que ses propres clés.
    """
    t_start = time.perf_counter()

    resultat_visual = sous_module_paligemma(
        rumor_text=state["rumor_text"],
        rumor_image=state.get("rumor_image")
    )

    execution_time = round(time.perf_counter() - t_start, 3)

    return {
        "classe_visual": resultat_visual["classe_paligemma"],
        "s_visual": resultat_visual["score_paligemma"],
        "explication_visual": resultat_visual["explication_paligemma"],
        "node_comparaison_visuelle_time_seconds": execution_time,
    }


def node_2(state: AgentState) -> dict:
    """
    Nœud de synthèse : ne s'exécute qu'une fois les DEUX branches parallèles
    terminées (fan-in natif LangGraph, voir build.py). Construit le rapport
    final via le LLM et assemble les métriques de performance à partir des
    deux temps reçus des branches parallèles.
    """
    t_start = time.perf_counter()

    classe_ooc = state.get("classe_ooc")
    if classe_ooc not in {"ooc", "nooc"}:
        raise ValueError("Le nœud 2 requiert une classe N1 binaire ('ooc' ou 'nooc').")

    resultat_ooc = {
        "classe_ooc": classe_ooc,
        "S_sem": state.get("s_sem_ooc"),
        "S_ent": state.get("s_ent_ooc"),
        "entites_rumeur": state.get("entites_rumeur_ooc", {}),
        "entites_dominantes": state.get("entites_dominantes_ooc", {}),
        "comparaison_entites": state.get("comparaison_entites_ooc", {}),
        "nombre_resultats": state.get("nombre_resultats", 0),
        "resultats": state.get("resultats", [])
    }

    resultat_synthese = sous_module_synthese(
        rumor_text=state["rumor_text"],
        resultat_ooc=resultat_ooc,
        classe_visual=state.get("classe_visual"),
        s_visual=state.get("s_visual"),
        explication_visual=state.get("explication_visual"),
    )

    execution_time_n2 = round(time.perf_counter() - t_start, 3)

    # Les deux branches parallèles ont tourné en même temps : le temps
    # "logique" du nœud 1 (tel qu'affiché côté frontend) est donc le plus
    # long des deux, pas leur somme.
    temps_recherche = state.get("node_recherche_origine_time_seconds", 0) or 0
    temps_visuel = state.get("node_comparaison_visuelle_time_seconds", 0) or 0
    temps_node_1 = round(max(temps_recherche, temps_visuel), 3)

    metrics = {
        "node_1_time_seconds": temps_node_1,
        "node_recherche_origine_time_seconds": temps_recherche,
        "node_comparaison_visuelle_time_seconds": temps_visuel,
        "node_2_time_seconds": execution_time_n2,
        "total_execution_time_seconds": round(temps_node_1 + execution_time_n2, 3),
    }

    return {
        "justification_json": resultat_synthese["justification"],
        "performance_metrics": metrics,
    }