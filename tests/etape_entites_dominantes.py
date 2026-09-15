"""
etape_entites_dominantes.py

Étape 5 du pipeline de test : calcule les "entités dominantes" de
chaque rumeur — les entités qui reviennent dans au moins
SEUIL_DOMINANCE (0.4, jamais recalibré, voir
calibrage/calibration_etape_c_comparaison.py) des résultats Lens —
à partir des entités déjà extraites par résultat (résultat["entites"],
produit par etape_contexte_e5.py).

Un seul appel Gemini par rumeur (calcule_entites_dominantes ->
normaliser_et_regrouper_entites), qui déduplique/normalise les alias
d'entités (ex. "USA"/"US" -> "United States") à travers tous les
résultats Lens de la rumeur, puis un calcul PUR (sans appel réseau)
de la fréquence d'apparition par entité normalisée.

PRÉ-REQUIS : chaque résultat de resultats_lens doit déjà avoir sa clé
"entites" (produite par etape_contexte_e5.py — sous-étape 1,
construction du contexte). Les rumeurs qui ne l'ont pas encore sont
comptées à part et sautées.

Marqueur "déjà traité" : cache["entites_dominantes"] est un dict (le
schéma à 6 catégories renvoyé par calcule_entites_dominantes, y
compris en cas d'échec du LLM — schema_entites_vide() reste un dict
non vide en tant que structure). Le cache est initialisé à [] (liste
vide) par creation_cache_mfc-ooc.py : c'est ce qui distingue "pas
encore traité" (liste) de "traité" (dict), même si le résultat est
vide.

QUOTA : QuotaLLMEpuiseError signale un vrai épuisement JOURNALIER de
Gemini — arrêt propre dès qu'elle survient.

RATE LIMIT : même limite gratuite de 15 req/min sur
gemini-3.5-flash-lite que les autres étapes LLM — appels espacés
proactivement pour éviter les 429 à répétition (voir
etape_entites_rumeur.py pour le détail de ce choix).

Chaque rumeur traitée est sauvegardée immédiatement — reprise exacte
au prochain lancement en cas d'interruption.

Utilisation :
    python etape_entites_dominantes.py
    python etape_entites_dominantes.py --limit 3
    python etape_entites_dominantes.py --statut
"""

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cache_utils import charger_cache, lister_ids_cache, sauvegarder_cache

from app.new_graph.node_1.n1_smOOC import (  # noqa: E402
    QuotaLLMEpuiseError,
    calcule_entites_dominantes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("etape_entites_dominantes")

SEUIL_DOMINANCE = 0.4  # jamais recalibré, voir calibration_etape_c_comparaison.py

# 15 requêtes/minute max côté Gemini (plan gratuit) -> 4s/appel min ; marge à 4.2s.
DELAI_MIN_ENTRE_APPELS_LLM = 4.2
_dernier_appel_llm: Optional[float] = None


def _attendre_limite_rpm() -> None:
    global _dernier_appel_llm
    maintenant = time.monotonic()
    if _dernier_appel_llm is not None:
        attente = DELAI_MIN_ENTRE_APPELS_LLM - (maintenant - _dernier_appel_llm)
        if attente > 0:
            time.sleep(attente)
    _dernier_appel_llm = time.monotonic()


# ------------------------------------------------------------
# ÉTAT D'AVANCEMENT D'UNE RUMEUR
# ------------------------------------------------------------

def entites_par_resultat_disponibles(resultats: List[Dict[str, Any]]) -> bool:
    return bool(resultats) and all("entites" in r for r in resultats)


def entites_dominantes_deja_fait(cache: Dict[str, Any]) -> bool:
    return isinstance(cache.get("entites_dominantes"), dict)


# ------------------------------------------------------------
# ÉTAPE
# ------------------------------------------------------------

def etape_entites_dominantes(limit: Optional[int] = None) -> None:
    rumeurs_traitees, deja_en_cache, en_attente_contexte = 0, 0, 0

    ids = sorted(lister_ids_cache())
    if not ids:
        logger.warning(
            "Aucun cache trouvé dans cache_test/ — lancez d'abord creation_cache_mfc-ooc.py, "
            "etape_recherche_inversee.py, etape_scraping.py puis etape_contexte_e5.py."
        )
        return

    total_fichiers = len(ids)
    fichiers_complets = sum(
        1 for i in ids
        if (c := charger_cache(i)) is not None and entites_dominantes_deja_fait(c)
    )
    logger.info(
        "[EntitésDominantes] Départ : %d/%d fichiers déjà complets, %d restants.",
        fichiers_complets, total_fichiers, total_fichiers - fichiers_complets,
    )

    for id_image in ids:
        cache = charger_cache(id_image)
        if cache is None:
            continue

        if entites_dominantes_deja_fait(cache):
            deja_en_cache += 1
            continue

        resultats = cache.get("resultats_lens") or []
        if not entites_par_resultat_disponibles(resultats):
            en_attente_contexte += 1
            continue

        if limit is not None and rumeurs_traitees >= limit:
            logger.info(
                "[EntitésDominantes] Limite --limit=%d atteinte — arrêt propre. Relancez sans "
                "--limit (ou avec une limite plus haute) pour continuer.",
                limit,
            )
            return

        logger.info(
            "[EntitésDominantes] id_image=%s -> normalisation et calcul de dominance (%d résultats).",
            id_image, len(resultats),
        )
        entites_par_resultat = [
            {"rang_lens": r["rang_lens"], **r["entites"]} for r in resultats
        ]

        _attendre_limite_rpm()
        try:
            cache["entites_dominantes"] = calcule_entites_dominantes(
                entites_par_resultat, seuil=SEUIL_DOMINANCE
            )
        except QuotaLLMEpuiseError as e:
            logger.error("QUOTA GEMINI RÉELLEMENT ÉPUISÉ — arrêt propre : %s", e)
            logger.error(
                "%d rumeurs traitées dans cette exécution. Relancez plus tard.", rumeurs_traitees
            )
            return

        sauvegarder_cache(id_image, cache)
        rumeurs_traitees += 1
        fichiers_complets += 1
        logger.info(
            "[EntitésDominantes] id_image=%s complet -> %d/%d fichiers complets (%d restants).",
            id_image, fichiers_complets, total_fichiers, total_fichiers - fichiers_complets,
        )

    logger.info(
        "[EntitésDominantes] terminé : %d rumeurs traitées, %d déjà en cache, "
        "%d en attente de contexte/entités par résultat. Fichiers complets : %d/%d (%d restants).",
        rumeurs_traitees, deja_en_cache, en_attente_contexte,
        fichiers_complets, total_fichiers, total_fichiers - fichiers_complets,
    )


def afficher_statut() -> None:
    ids = lister_ids_cache()

    total, complets, en_attente_contexte = 0, 0, 0
    for id_image in ids:
        cache = charger_cache(id_image)
        if cache is None:
            continue
        total += 1
        if entites_dominantes_deja_fait(cache):
            complets += 1
        elif not entites_par_resultat_disponibles(cache.get("resultats_lens") or []):
            en_attente_contexte += 1

    print("\n--- Statut entités dominantes ---")
    print(f"Fichiers complets              : {complets} / {total} ({total - complets} restants)")
    print(f"  dont en attente de contexte  : {en_attente_contexte}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calcul des entités dominantes par rumeur (normalisation LLM + seuil de fréquence)."
    )
    parser.add_argument(
        "--statut", action="store_true",
        help="Affiche l'avancement, sans rien exécuter.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Nombre maximum de rumeurs à traiter dans cette exécution (défaut : toutes).",
    )
    args = parser.parse_args()

    if args.statut:
        afficher_statut()
        return

    etape_entites_dominantes(limit=args.limit)


if __name__ == "__main__":
    main()
