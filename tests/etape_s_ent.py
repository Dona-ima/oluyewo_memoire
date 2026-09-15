"""
etape_s_ent.py

Étape 6 du pipeline de test : compare les entités de la rumeur
(entites_rumeur, produit par etape_entites_rumeur.py) aux entités
dominantes des résultats Lens (entites_dominantes, produit par
etape_entites_dominantes.py), catégorie par catégorie, pour calculer
S_ent et le détail comparaison_entites.

Un seul appel Gemini par rumeur (calcule_s_ent -> comparer_entites),
et SEULEMENT si au moins une catégorie a des entités des deux côtés à
comparer réellement (sinon calcule_s_ent renvoie un score directement,
sans appel réseau — catégories automatiquement notées incompatibles).

PRÉ-REQUIS : cache["entites_rumeur"] non vide (etape_entites_rumeur.py)
ET cache["entites_dominantes"] est un dict (etape_entites_dominantes.py).
Les rumeurs qui n'ont pas encore les deux sont comptées à part et
sautées.

Marqueur "déjà traité" : présence de la clé "categories_comparables"
dans le cache. Ce n'est PAS un champ du schéma initial de
creation_cache_mfc-ooc.py (contrairement à S_ent/comparaison_entites,
initialisés à null/{} — donc ambigus : un S_ent=None légitime, "aucune
catégorie comparable des deux côtés", est indiscernable d'un S_ent pas
encore calculé). "categories_comparables" est toujours une liste
(même vide) une fois calcule_s_ent() appelé, jamais absente — c'est
donc le seul marqueur non ambigu disponible ici.

QUOTA : QuotaLLMEpuiseError signale un vrai épuisement JOURNALIER de
Gemini — arrêt propre dès qu'elle survient. (Bug corrigé dans
n1_OOC_calibration.py::calcule_s_ent avant l'écriture de ce script :
la fonction avalait silencieusement cette exception au lieu de la
laisser remonter — un épuisement de quota pendant cette étape aurait
été pris à tort pour "aucune entité comparable" sur chaque rumeur
suivante, sans jamais s'arrêter proprement.)

RATE LIMIT : même limite gratuite de 15 req/min sur
gemini-3.5-flash-lite que les autres étapes LLM — appels espacés
proactivement (voir etape_entites_rumeur.py pour le détail).

Chaque rumeur traitée est sauvegardée immédiatement — reprise exacte
au prochain lancement en cas d'interruption.

Utilisation :
    python etape_s_ent.py
    python etape_s_ent.py --limit 3
    python etape_s_ent.py --statut
"""

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cache_utils import charger_cache, lister_ids_cache, sauvegarder_cache

from app.new_graph.node_1.n1_smOOC import (  # noqa: E402
    QuotaLLMEpuiseError,
    calcule_s_ent,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("etape_s_ent")

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

def prerequis_disponibles(cache: Dict[str, Any]) -> bool:
    return bool(cache.get("entites_rumeur")) and isinstance(cache.get("entites_dominantes"), dict)


def s_ent_deja_fait(cache: Dict[str, Any]) -> bool:
    return "categories_comparables" in cache


# ------------------------------------------------------------
# ÉTAPE
# ------------------------------------------------------------

def etape_s_ent(limit: Optional[int] = None) -> None:
    rumeurs_traitees, deja_en_cache, en_attente_prerequis = 0, 0, 0

    ids = sorted(lister_ids_cache())
    if not ids:
        logger.warning(
            "Aucun cache trouvé dans cache_test/ — lancez d'abord etape_entites_rumeur.py "
            "et etape_entites_dominantes.py."
        )
        return

    total_fichiers = len(ids)
    fichiers_complets = sum(
        1 for i in ids
        if (c := charger_cache(i)) is not None and s_ent_deja_fait(c)
    )
    logger.info(
        "[S_ent] Départ : %d/%d fichiers déjà complets, %d restants.",
        fichiers_complets, total_fichiers, total_fichiers - fichiers_complets,
    )

    for id_image in ids:
        cache = charger_cache(id_image)
        if cache is None:
            continue

        if s_ent_deja_fait(cache):
            deja_en_cache += 1
            continue

        if not prerequis_disponibles(cache):
            en_attente_prerequis += 1
            continue

        if limit is not None and rumeurs_traitees >= limit:
            logger.info(
                "[S_ent] Limite --limit=%d atteinte — arrêt propre. Relancez sans "
                "--limit (ou avec une limite plus haute) pour continuer.",
                limit,
            )
            return

        logger.info("[S_ent] id_image=%s -> comparaison des entités.", id_image)
        _attendre_limite_rpm()
        try:
            resultat = calcule_s_ent(cache["entites_rumeur"], cache["entites_dominantes"])
        except QuotaLLMEpuiseError as e:
            logger.error("QUOTA GEMINI RÉELLEMENT ÉPUISÉ — arrêt propre : %s", e)
            logger.error(
                "%d rumeurs traitées dans cette exécution. Relancez plus tard.", rumeurs_traitees
            )
            return

        if resultat is None:
            logger.warning(
                "[S_ent] id_image=%s -> échec inattendu (voir logs ci-dessus), "
                "retenté au prochain lancement.", id_image,
            )
            continue

        cache["S_ent"] = resultat["score"]
        cache["comparaison_entites"] = resultat["comparaison"]
        cache["categories_comparables"] = resultat["categories_comparables"]

        sauvegarder_cache(id_image, cache)
        rumeurs_traitees += 1
        fichiers_complets += 1
        logger.info(
            "[S_ent] id_image=%s complet (S_ent=%s) -> %d/%d fichiers complets (%d restants).",
            id_image, cache["S_ent"], fichiers_complets, total_fichiers, total_fichiers - fichiers_complets,
        )

    logger.info(
        "[S_ent] terminé : %d rumeurs traitées, %d déjà en cache, "
        "%d en attente des pré-requis. Fichiers complets : %d/%d (%d restants).",
        rumeurs_traitees, deja_en_cache, en_attente_prerequis,
        fichiers_complets, total_fichiers, total_fichiers - fichiers_complets,
    )


def afficher_statut() -> None:
    ids = lister_ids_cache()

    total, complets, en_attente_prerequis = 0, 0, 0
    for id_image in ids:
        cache = charger_cache(id_image)
        if cache is None:
            continue
        total += 1
        if s_ent_deja_fait(cache):
            complets += 1
        elif not prerequis_disponibles(cache):
            en_attente_prerequis += 1

    print("\n--- Statut S_ent / comparaison_entites ---")
    print(f"Fichiers complets              : {complets} / {total} ({total - complets} restants)")
    print(f"  dont en attente des pré-requis : {en_attente_prerequis}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calcul de S_ent (comparaison entites_rumeur / entites_dominantes) par rumeur."
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

    etape_s_ent(limit=args.limit)


if __name__ == "__main__":
    main()
