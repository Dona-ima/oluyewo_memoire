from state import AgentState
from app.graph.node_1.n1_smOOC import sous_module_ooc
from app.graph.node_1.n1_smCoherence import sous_module_visual
from app.graph.node_2.n2_synthese import sous_module_synthese


def node_1(state: AgentState) -> AgentState:

    rumor_text = state["rumor_text"]
    rumor_image = state.get("rumor_image")
    resultat_ooc = sous_module_ooc(url_image=rumor_image, rumeur=rumor_text)
    resultat_visual = sous_module_visual(rumor_text=rumor_text,rumor_image=rumor_image)
    return {
        **state,

        # ---- OOC ----
        "classe_ooc": resultat_ooc["classe"],
        "s_ooc": resultat_ooc["S_OOC"],
        "s_sem_ooc": resultat_ooc["S_sem"],
        "s_ent_ooc": resultat_ooc["S_ent"],
        "entites_rumeur_ooc": resultat_ooc["entites_rumeur"],
        "entites_dominantes_ooc": resultat_ooc["entites_dominantes"],
        "comparaison_entites_ooc": resultat_ooc["comparaison_entites"],
        "nombre_resultats": len(resultat_ooc["resultats"]),
        "resultats": resultat_ooc["resultats"],
        

        # ---- SigLIP2 ----
        "classe_visual": resultat_visual["statut_visual"],
        "s_visual": resultat_visual["s_visual"],
        
    }

def node_2(state: AgentState) -> AgentState:

    resultat_ooc = {
        "classe": state.get("classe_ooc"),
        "S_ooc": state.get("s_ooc"),
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
        s_visual=state.get("s_visual")
    )

    return {
        **state,
        "justification_json": resultat_synthese[
            "justification"
        ]
    }