"""
etape_entites_rumeur.py

Étape 4 du pipeline de test : extrait les entités présentes dans la
rumeur elle-même (le "caption"), indépendamment des résultats Lens —
un seul appel Gemini par rumeur (construction_entites_rumeur), qui
n'a besoin que du champ "caption" du cache.

Cette étape est donc indépendante de l'avancement de
etape_contexte_e5.py : elle peut tourner sur les 196 fichiers dès que
creation_cache_mfc-ooc.py a initialisé les caches, sans attendre la
recherche inversée, le scraping ou le calcul E5.

Marqueur "déjà traité" : cache["entites_rumeur"] non vide. Le cache
est initialisé à {} par creation_cache_mfc-ooc.py ; même en cas
d'échec total du LLM, construction_entites_rumeur() renvoie le schéma
vide à 6 catégories (schema_entites_vide(), non vide en tant que
dict — seules les listes internes sont vides), ce qui marque la
rumeur comme traitée et évite de la retenter en boucle sur un échec
récurrent (rumeur vide, par exemple).

QUOTA : QuotaLLMEpuiseError signale un vrai épuisement JOURNALIER de
Gemini (détecté par l'API elle-même) — arrêt propre dès qu'elle
survient, rien à calibrer côté client.

RATE LIMIT (par minute) : le plan gratuit Gemini limite
gemini-3.5-flash-lite à 15 requêtes/minute. appel_llm() retente déjà
automatiquement un 429 "PerMinute" (RateLimitLLMTransitoireError,
backoff exponentiel respectant le retry_delay suggéré par Gemini),
donc rien ne casse — mais laisser le script foncer à pleine vitesse
fait déclencher ce 429 en continu, chaque retry coûtant 30 à 60+
secondes d'attente. On espace donc proactivement les appels pour
rester sous la limite et éviter ces à-coups.

Chaque rumeur traitée est sauvegardée immédiatement — aucune perte de
travail en cas d'interruption, reprise exacte au prochain lancement.

Utilisation :
    python etape_entites_rumeur.py
    python etape_entites_rumeur.py --limit 3
    python etape_entites_rumeur.py --statut
"""

import argparse
import logging
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cache_utils import charger_cache, lister_ids_cache, sauvegarder_cache

from app.new_graph.node_1.n1_smOOC import (  # noqa: E402
    QuotaLLMEpuiseError,
    construction_entites_rumeur,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("etape_entites_rumeur")

# 15 requêtes/minute max (plan gratuit) -> 4s/appel min ; marge à 4.2s.
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


def entites_rumeur_deja_fait(cache: dict) -> bool:
    return bool(cache.get("entites_rumeur"))


def etape_entites_rumeur(limit: Optional[int] = None) -> None:
    rumeurs_traitees, deja_en_cache = 0, 0

    ids = sorted(lister_ids_cache())
    if not ids:
        logger.warning(
            "Aucun cache trouvé dans cache_test/ — lancez d'abord creation_cache_mfc-ooc.py."
        )
        return

    total_fichiers = len(ids)
    fichiers_complets = sum(
        1 for i in ids
        if (c := charger_cache(i)) is not None and entites_rumeur_deja_fait(c)
    )
    logger.info(
        "[EntitésRumeur] Départ : %d/%d fichiers déjà complets, %d restants.",
        fichiers_complets, total_fichiers, total_fichiers - fichiers_complets,
    )

    for id_image in ids:
        cache = charger_cache(id_image)
        if cache is None:
            continue

        if entites_rumeur_deja_fait(cache):
            deja_en_cache += 1
            continue

        if limit is not None and rumeurs_traitees >= limit:
            logger.info(
                "[EntitésRumeur] Limite --limit=%d atteinte — arrêt propre. Relancez sans "
                "--limit (ou avec une limite plus haute) pour continuer.",
                limit,
            )
            return

        logger.info("[EntitésRumeur] id_image=%s -> extraction des entités.", id_image)
        _attendre_limite_rpm()
        try:
            cache["entites_rumeur"] = construction_entites_rumeur(cache.get("caption", ""))
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
            "[EntitésRumeur] id_image=%s complet -> %d/%d fichiers complets (%d restants).",
            id_image, fichiers_complets, total_fichiers, total_fichiers - fichiers_complets,
        )

    logger.info(
        "[EntitésRumeur] terminé : %d rumeurs traitées, %d déjà en cache. "
        "Fichiers complets : %d/%d (%d restants).",
        rumeurs_traitees, deja_en_cache,
        fichiers_complets, total_fichiers, total_fichiers - fichiers_complets,
    )


def afficher_statut() -> None:
    ids = lister_ids_cache()

    total, complets = 0, 0
    for id_image in ids:
        cache = charger_cache(id_image)
        if cache is None:
            continue
        total += 1
        if entites_rumeur_deja_fait(cache):
            complets += 1

    print("\n--- Statut entités rumeur ---")
    print(f"Fichiers complets : {complets} / {total} ({total - complets} restants)\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extraction des entités présentes dans chaque rumeur (caption) via Gemini."
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

    etape_entites_rumeur(limit=args.limit)


if __name__ == "__main__":
    main()
