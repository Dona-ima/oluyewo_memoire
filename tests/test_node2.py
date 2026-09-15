import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.graph.node_2.n2_synthese import sous_module_synthese


# ============================================================
# DONNÉES DE TEST
# ============================================================

rumor_text = (
    "Cette image montre le président béninois Patrice Talon "
    "lors d'une manifestation à Cotonou en janvier 2025."
)


resultat_ooc = {
    "classe": "ooc",
    "S_ooc": 0.27,
    "S_sem": 0.31,
    "S_ent": 0.20,

    "entites_rumeur": {
        "PERSON": ["Patrice Talon"],
        "LOCATION": ["Cotonou"],
        "DATE": ["janvier 2025"],
        "EVENT": ["manifestation"],
        "ORGANIZATION": [],
        "OTHER": []
    },

    "entites_dominantes": {
        "PERSON": ["Patrice Talon"],
        "LOCATION": ["Abidjan"],
        "DATE": ["2023"],
        "EVENT": ["sommet"],
        "ORGANIZATION": [],
        "OTHER": []
    },

    "comparaison_entites": {
        "PERSON": 1,
        "LOCATION": 0,
        "DATE": 0,
        "EVENT": 0,
        "ORGANIZATION": 0,
        "OTHER": 0
    },

    "nombre_resultats": 3,

    "resultats": [
        {
            "rang_lens": 1,
            "titre": "Patrice Talon lors d'un événement à Abidjan",
            "source": "Example News",
            "lien": "https://example.com/article1",
            "texte_contexte": (
                "La photographie montre Patrice Talon lors "
                "d'un événement organisé à Abidjan en 2023."
            ),
            "score_e5": 0.28,
            "entites": {
                "PERSON": ["Patrice Talon"],
                "LOCATION": ["Abidjan"],
                "DATE": ["2023"],
                "EVENT": ["événement"],
                "ORGANIZATION": [],
                "OTHER": []
            }
        },

        {
            "rang_lens": 2,
            "titre": "Patrice Talon à un sommet régional",
            "source": "Example Media",
            "lien": "https://example.com/article2",
            "texte_contexte": (
                "Cette image a été utilisée pour illustrer "
                "la participation de Patrice Talon à un sommet "
                "régional en 2023."
            ),
            "score_e5": 0.30,
            "entites": {
                "PERSON": ["Patrice Talon"],
                "LOCATION": ["Abidjan"],
                "DATE": ["2023"],
                "EVENT": ["sommet"],
                "ORGANIZATION": [],
                "OTHER": []
            }
        },

        {
            "rang_lens": 3,
            "titre": "Archive : Patrice Talon en déplacement",
            "source": "Example Archive",
            "lien": "https://example.com/article3",
            "texte_contexte": (
                "Photographie d'archive montrant Patrice Talon "
                "lors d'un déplacement en 2023."
            ),
            "score_e5": 0.25,
            "entites": {
                "PERSON": ["Patrice Talon"],
                "LOCATION": ["Abidjan"],
                "DATE": ["2023"],
                "EVENT": [],
                "ORGANIZATION": [],
                "OTHER": []
            }
        }
    ]
}


# ============================================================
# RÉSULTAT DU MODULE DE COHÉRENCE
# ============================================================

classe_visual = "contradiction"
s_visual = 0.81


# ============================================================
# APPEL DU NŒUD 2
# ============================================================

resultat = sous_module_synthese(
    rumor_text=rumor_text,
    resultat_ooc=resultat_ooc,
    classe_visual=classe_visual,
    s_visual=s_visual
)


# ============================================================
# AFFICHAGE
# ============================================================

print("\n" + "=" * 70)
print("RÉSULTAT DU NŒUD 2")
print("=" * 70)

print(resultat["statut"])

print("\n--- JUSTIFICATION ---")

import json

print(
    json.dumps(
        resultat["justification"],
        ensure_ascii=False,
        indent=2
    )
)