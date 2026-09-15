"""
etape_contexte_e5.py

Étape 3 du pipeline de test : pour chaque rumeur déjà scrapée
(cache_test/<id_image>.json, résultats Lens déjà enrichis par
etape_scraping.py), construit le contexte et les entités de chaque
résultat, puis calcule le score E5 par résultat et le S_sem agrégé de
la rumeur.

Deux sous-étapes distinctes, chacune reprise indépendamment si elle a
déjà été faite lors d'une exécution précédente (pas de gaspillage d'un
appel LLM/HF déjà payé sur un travail déjà terminé) :

1. construction_texte_contexte_et_entites(resultats) — UN appel Gemini
   pour toute la rumeur (les 5 résultats à la fois) : reconstruit un
   texte de contexte propre par résultat ("texte_contexte",
   "niveau_contexte") et en extrait les entités ("entites"). Bascule
   automatiquement sur un fallback local (sans LLM, à partir du texte
   Webpage brut) en cas d'échec du LLM — voir
   n1_OOC_calibration.py::_fallback_contexte.
   -> Sautée si tous les résultats ont déjà "texte_contexte".

2. calcule_scores_e5(caption, resultats) + calcule_s_reverse(resultats)
   — deux appels HuggingFace (embeddings E5) : un pour la rumeur, un
   second batché pour tous les contextes exploitables. Écrit
   "score_e5" pour chaque résultat et "S_sem" (moyenne) au niveau de
   la rumeur.
   -> Sautée si tous les résultats ont déjà "score_e5".

QUOTA :
- Gemini (sous-étape 1) : QuotaLLMEpuiseError signale un vrai
  épuisement JOURNALIER détecté par l'API elle-même (pas une
  estimation côté client comme pour les crédits Serper) — le script
  s'arrête proprement dès qu'elle survient, rien à calibrer.
- HuggingFace/E5 (sous-étape 2) : aucune exception de quota dédiée
  côté n1_OOC_calibration.py. Un rate-limit ou un épuisement HF se
  traduit juste par score_e5=None / S_sem=None pour la rumeur
  concernée (loggé en warning), sans arrêt du batch — à surveiller si
  S_sem reste null en masse, ce serait le signe d'un token HF invalide
  ou d'un quota HF épuisé passé inaperçu.

Chaque rumeur traitée est sauvegardée immédiatement après ses deux
sous-étapes — aucune perte de travail en cas d'interruption, reprise
exacte au prochain lancement.

Utilisation :
    python etape_contexte_e5.py
    python etape_contexte_e5.py --statut
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
    calcule_s_reverse,
    calcule_scores_e5,
    construction_texte_contexte_et_entites,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("etape_contexte_e5")

# 15 requêtes/minute max côté Gemini (plan gratuit) -> 4s/appel min ;
# marge à 4.2s. N'espace que les appels LLM (construction du contexte)
# — les appels HuggingFace (E5) ne sont pas concernés par cette limite.
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

def contexte_deja_fait(resultats: List[Dict[str, Any]]) -> bool:
    return bool(resultats) and all("texte_contexte" in r for r in resultats)


def e5_deja_fait(resultats: List[Dict[str, Any]]) -> bool:
    return bool(resultats) and all("score_e5" in r for r in resultats)


def rumeur_complete(resultats: List[Dict[str, Any]]) -> bool:
    return contexte_deja_fait(resultats) and e5_deja_fait(resultats)


# ------------------------------------------------------------
# ÉTAPE
# ------------------------------------------------------------

def etape_contexte_e5(limit: Optional[int] = None) -> None:
    rumeurs_traitees, deja_en_cache, en_attente_scraping = 0, 0, 0

    ids = sorted(lister_ids_cache())
    if not ids:
        logger.warning(
            "Aucun cache trouvé dans cache_test/ — lancez d'abord creation_cache_mfc-ooc.py, "
            "etape_recherche_inversee.py puis etape_scraping.py."
        )
        return

    total_fichiers = len(ids)
    fichiers_complets = sum(
        1 for i in ids
        if (c := charger_cache(i)) is not None and rumeur_complete(c.get("resultats_lens") or [])
    )
    logger.info(
        "[Contexte/E5] Départ : %d/%d fichiers déjà complets, %d restants.",
        fichiers_complets, total_fichiers, total_fichiers - fichiers_complets,
    )

    for id_image in ids:
        cache = charger_cache(id_image)
        resultats = cache.get("resultats_lens") if cache else None
        if not resultats:
            en_attente_scraping += 1
            continue

        if rumeur_complete(resultats):
            deja_en_cache += 1
            continue

        if limit is not None and rumeurs_traitees >= limit:
            logger.info(
                "[Contexte/E5] Limite --limit=%d atteinte — arrêt propre. Relancez sans "
                "--limit (ou avec une limite plus haute) pour continuer.",
                limit,
            )
            return

        if not contexte_deja_fait(resultats):
            logger.info(
                "[Contexte] id_image=%s -> construction du contexte et des entités (%d résultats).",
                id_image, len(resultats),
            )
            _attendre_limite_rpm()
            try:
                construction_texte_contexte_et_entites(resultats)
            except QuotaLLMEpuiseError as e:
                logger.error("QUOTA GEMINI RÉELLEMENT ÉPUISÉ — arrêt propre : %s", e)
                logger.error(
                    "%d rumeurs traitées dans cette exécution. Relancez plus tard.", rumeurs_traitees
                )
                return
        else:
            logger.info(
                "[Contexte] id_image=%s -> contexte déjà présent, passage direct à E5.", id_image
            )

        logger.info("[E5] id_image=%s -> calcul des scores E5.", id_image)
        calcule_scores_e5(cache["caption"], resultats)
        cache["S_sem"] = calcule_s_reverse(resultats)
        if cache["S_sem"] is None:
            logger.warning(
                "[E5] id_image=%s -> S_sem=None (aucun score E5 exploitable).", id_image
            )

        sauvegarder_cache(id_image, cache)
        rumeurs_traitees += 1

        if rumeur_complete(resultats):
            fichiers_complets += 1
            logger.info(
                "[Contexte/E5] id_image=%s complet -> %d/%d fichiers complets (%d restants).",
                id_image, fichiers_complets, total_fichiers, total_fichiers - fichiers_complets,
            )

    logger.info(
        "[Contexte/E5] terminé : %d rumeurs traitées, %d déjà en cache, "
        "%d en attente de recherche/scraping. Fichiers complets : %d/%d (%d restants).",
        rumeurs_traitees, deja_en_cache, en_attente_scraping,
        fichiers_complets, total_fichiers, total_fichiers - fichiers_complets,
    )


def afficher_statut() -> None:
    ids = lister_ids_cache()

    fichiers_avec_resultats, fichiers_complets = 0, 0
    fichiers_contexte_fait, fichiers_e5_fait = 0, 0
    for id_image in ids:
        cache = charger_cache(id_image)
        resultats = cache.get("resultats_lens") if cache else None
        if not resultats:
            continue
        fichiers_avec_resultats += 1
        if contexte_deja_fait(resultats):
            fichiers_contexte_fait += 1
        if e5_deja_fait(resultats):
            fichiers_e5_fait += 1
        if rumeur_complete(resultats):
            fichiers_complets += 1

    print("\n--- Statut contexte/entités + E5 ---")
    print(f"Fichiers avec résultats Lens : {fichiers_avec_resultats} / {len(ids)}")
    print(f"Contexte+entités déjà fait   : {fichiers_contexte_fait} / {fichiers_avec_resultats}")
    print(f"Score E5 déjà fait           : {fichiers_e5_fait} / {fichiers_avec_resultats}")
    print(
        f"Fichiers complets            : {fichiers_complets} / {fichiers_avec_resultats} "
        f"({fichiers_avec_resultats - fichiers_complets} restants)\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construction du contexte/entités par résultat puis calcul du score E5 (S_sem)."
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

    etape_contexte_e5(limit=args.limit)


if __name__ == "__main__":
    main()
