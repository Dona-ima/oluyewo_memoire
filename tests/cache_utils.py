"""
cache_utils.py

Gestion du cache de test : un fichier JSON par rumeur (identifiée par
son `rumeur_id`), rempli progressivement au fil des étapes du pipeline.

Objectif : pouvoir arrêter/reprendre le traitement à tout moment (quota
Serper ou Gemini épuisé) SANS perdre le travail déjà payé en appels API,
et sans jamais retraiter une rumeur déjà avancée à une étape donnée.
"""

import json
import os
from typing import Any, Dict, Optional

CACHE_DIR = "cache_test"


def chemin_cache(rumeur_id: str) -> str:
    return os.path.join(CACHE_DIR, f"{rumeur_id}.json")


def charger_cache(rumeur_id: str) -> Optional[Dict[str, Any]]:
    chemin = chemin_cache(rumeur_id)
    if not os.path.exists(chemin):
        return None
    with open(chemin, "r", encoding="utf-8") as f:
        return json.load(f)


def sauvegarder_cache(rumeur_id: str, donnees: Dict[str, Any]) -> None:
    """
    Écriture atomique (fichier temporaire puis renommage) : évite qu'un
    crash pendant l'écriture ne laisse un fichier JSON à moitié écrit et
    donc illisible au prochain lancement.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    chemin = chemin_cache(rumeur_id)
    chemin_tmp = chemin + ".tmp"
    with open(chemin_tmp, "w", encoding="utf-8") as f:
        json.dump(donnees, f, ensure_ascii=False, indent=2)
    os.replace(chemin_tmp, chemin)


def initialiser_cache(
    rumeur_id: str,
    url_image: str,
    rumeur: str,
    label_reel: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retourne le cache existant s'il y en a un (jamais écrasé), sinon crée
    la structure vide de départ.
    """
    existant = charger_cache(rumeur_id)
    if existant is not None:
        return existant

    return {
        "rumeur_id": rumeur_id,
        "url_image": url_image,
        "rumeur": rumeur,
        "label_reel": label_reel,

        # Étape A — recherche inversée + scraping
        "resultats_lens": None,

        # Étape B — contexte + entités par résultat + score E5 (rempli
        # directement dans les éléments de resultats_lens)

        # Étape C — entités de la rumeur
        "entites_rumeur": None,

        # Étape D — entités dominantes des résultats
        "entites_dominantes": None,

        # Étape E — scores finaux
        "S_sem": None,
        "S_ent": None,
        "comparaison_entites": None,
        "S_OOC": None,
        "classe_predite": None,

        "erreur": None,
    }


def lister_ids_cache() -> list:
    if not os.path.isdir(CACHE_DIR):
        return []
    return [
        nom[:-5] for nom in os.listdir(CACHE_DIR)
        if nom.endswith(".json")
    ]