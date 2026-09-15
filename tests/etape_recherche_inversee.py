"""
etape_recherche_inversee.py

Étape 1 du pipeline de test : UNIQUEMENT la recherche inversée Google
Lens (Serper), pour chaque cache déjà créé par creation_cache.py
(cache_test/<id_image>.json) qui n'a pas encore de resultats_lens. Le
scraping des pages associées (étape 2) est fait par un script séparé.

Cette étape ne consomme QUE le quota de l'endpoint /lens de Serper — le
scraping (endpoint séparé, coût différent) n'est pas déclenché ici.

GESTION DU QUOTA SERPER (free tier) :

1. Chaque recherche inversée (appel à recherche_inversee(), endpoint
   /lens) coûte 3 crédits Serper. Un compteur cumulé est tenu dans
   etat_credits_serper.json et incrémenté de 3 à chaque recherche
   effectuée — PERSISTANT d'une exécution à l'autre : relancer le
   script continue le décompte là où il s'était arrêté, pas depuis 0.

2. Dès que le compteur atteint --limite-credits (2000 par défaut), le
   script s'arrête proprement : c'est un budget de SÉCURITÉ interne,
   volontairement choisi en dessous du quota réel du compte pour ne
   jamais s'en approcher.

3. En complément, si Serper répond un véritable épuisement de crédits
   ou une clé invalide, sous_module_ooc.py lève QuotaScrapingEpuiseError
   — attrapée ici pour un arrêt tout aussi propre, y compris si
   --limite-credits est réglée plus haut que ce qu'il reste réellement
   sur le compte.

Dans les deux cas : chaque cache est sauvegardé immédiatement après son
traitement (jamais de perte), et relancer la même commande reprend
exactement où on s'est arrêté (les caches ayant déjà des resultats_lens
sont sautés).

NOTE IMPORTANTE sur le décompte : recherche_inversee() dans
sous_module_ooc.py retente un appel Lens supplémentaire lorsque le
premier revient sans aucun résultat (flakiness connue de Google Lens),
avant de conclure à "aucune correspondance". Dans ce cas précis un seul
appel à recherche_inversee() coûte donc 2 requêtes Lens réelles (~6
crédits), pas 3 — le compteur ci-dessous suit l'hypothèse "3 crédits par
recherche inversée" telle que donnée, à ajuster si vous voulez un
décompte exact incluant ce cas de figure.

Quand votre quota Serper se renouvelle (nouveau mois, nouvelle clé...),
remettez le compteur à zéro avec --reset-compteur.

Utilisation :
    python etape_recherche_inversee.py
    python etape_recherche_inversee.py --limite-credits 500
    python etape_recherche_inversee.py --statut
    python etape_recherche_inversee.py --reset-compteur
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache_utils import charger_cache, lister_ids_cache, sauvegarder_cache

from app.new_graph.node_1.n1_smOOC import (  # noqa: E402
    QuotaScrapingEpuiseError,
    recherche_inversee,
    selection_top_k_lens,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("etape_recherche_inversee")

K_CALIBRE = 5  # voir calibration : K=5 retenu

CREDITS_PAR_RECHERCHE = 3
LIMITE_CREDITS_DEFAUT = 2000

COMPTEUR_PATH = "etat_credits_serper.json"


# ------------------------------------------------------------
# COMPTEUR DE CRÉDITS (persistant entre exécutions)
# ------------------------------------------------------------

def charger_compteur() -> Dict[str, Any]:
    if os.path.exists(COMPTEUR_PATH):
        with open(COMPTEUR_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"credits_consommes": 0, "recherches_effectuees": 0}


def sauvegarder_compteur(etat: Dict[str, Any]) -> None:
    chemin_tmp = COMPTEUR_PATH + ".tmp"
    with open(chemin_tmp, "w", encoding="utf-8") as f:
        json.dump(etat, f, ensure_ascii=False, indent=2)
    os.replace(chemin_tmp, COMPTEUR_PATH)


def reinitialiser_compteur() -> None:
    sauvegarder_compteur({"credits_consommes": 0, "recherches_effectuees": 0})
    logger.info("Compteur de crédits Serper remis à zéro.")


# ------------------------------------------------------------
# ÉTAPE
# ------------------------------------------------------------

def cache_deja_traite(cache: Dict[str, Any]) -> bool:
    return bool(cache.get("resultats_lens"))


def etape_recherche(limite_credits: int) -> None:
    compteur = charger_compteur()
    traites, deja_en_cache, sans_url = 0, 0, 0

    ids = sorted(lister_ids_cache())
    if not ids:
        logger.warning("Aucun cache trouvé dans cache_test/ — lancez d'abord creation_cache.py.")
        return

    for id_image in ids:
        cache = charger_cache(id_image)
        if cache is None:
            continue

        if cache_deja_traite(cache):
            deja_en_cache += 1
            continue

        if not cache.get("url_image"):
            logger.warning("[Recherche] id_image=%s sans url_image — ignoré.", id_image)
            sans_url += 1
            continue

        if compteur["credits_consommes"] + CREDITS_PAR_RECHERCHE > limite_credits:
            logger.warning(
                "LIMITE INTERNE DE CRÉDITS ATTEINTE (%d/%d) — arrêt propre. "
                "%d rumeurs traitées dans cette exécution. Relancez plus tard, "
                "augmentez --limite-credits, ou réinitialisez avec --reset-compteur "
                "si votre quota Serper s'est renouvelé.",
                compteur["credits_consommes"], limite_credits, traites,
            )
            return

        logger.info(
            "[Recherche] id_image=%s (crédits consommés jusqu'ici : %d/%d)",
            id_image, compteur["credits_consommes"], limite_credits,
        )

        try:
            bruts = recherche_inversee(cache["url_image"])
            # Le crédit est consommé dès l'appel /lens, qu'il retourne
            # des résultats ou non — on incrémente donc juste après
            # l'appel, avant même de savoir si des résultats existent.
            compteur["credits_consommes"] += CREDITS_PAR_RECHERCHE
            compteur["recherches_effectuees"] += 1
            sauvegarder_compteur(compteur)

            resultats = selection_top_k_lens(bruts, k=K_CALIBRE)
        except QuotaScrapingEpuiseError as e:
            logger.error("QUOTA SERPER RÉELLEMENT ÉPUISÉ — arrêt propre : %s", e)
            logger.error("%d rumeurs traitées dans cette exécution. Relancez plus tard.", traites)
            return

        cache["resultats_lens"] = resultats
        sauvegarder_cache(id_image, cache)
        traites += 1

    logger.info(
        "[Recherche] terminé : %d traités, %d déjà en cache, %d sans url_image. "
        "Crédits consommés au total : %d.",
        traites, deja_en_cache, sans_url, compteur["credits_consommes"],
    )


def afficher_statut(limite_credits: int) -> None:
    compteur = charger_compteur()
    ids = lister_ids_cache()

    traites = 0
    for id_image in ids:
        cache = charger_cache(id_image)
        if cache and cache_deja_traite(cache):
            traites += 1

    credits_restants = max(limite_credits - compteur["credits_consommes"], 0)
    recherches_possibles = credits_restants // CREDITS_PAR_RECHERCHE

    print("\n--- Statut recherche inversée ---")
    print(f"Crédits consommés     : {compteur['credits_consommes']} / {limite_credits}")
    print(f"Recherches effectuées  : {compteur['recherches_effectuees']}")
    print(f"Caches avec résultats  : {traites} / {len(ids)}")
    print(f"Caches restants        : {len(ids) - traites}")
    print(f"Recherches encore possibles avec cette limite : ~{recherches_possibles}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recherche inversée Serper (Lens uniquement), résumable et bornée en budget de crédits."
    )
    parser.add_argument(
        "--limite-credits", type=int, default=LIMITE_CREDITS_DEFAUT,
        help=f"Budget de crédits Serper pour cette exécution (défaut : {LIMITE_CREDITS_DEFAUT}, {CREDITS_PAR_RECHERCHE} crédits/recherche).",
    )
    parser.add_argument(
        "--reset-compteur", action="store_true",
        help="Remet le compteur de crédits à zéro (ex. après renouvellement du quota), puis quitte.",
    )
    parser.add_argument(
        "--statut", action="store_true",
        help="Affiche l'avancement et la consommation de crédits, sans rien exécuter.",
    )
    args = parser.parse_args()

    if args.reset_compteur:
        reinitialiser_compteur()
        return

    if args.statut:
        afficher_statut(args.limite_credits)
        return

    etape_recherche(args.limite_credits)


if __name__ == "__main__":
    main()