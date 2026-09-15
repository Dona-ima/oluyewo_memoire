"""
etape_scraping.py

Étape 2 du pipeline de test : scraping des pages associées aux résultats
de recherche inversée déjà en cache (remplis par
etape_recherche_inversee.py), pour chaque résultat qui n'a pas encore de
clé "webpage".

QUOTA PARTAGÉ AVEC LA RECHERCHE INVERSÉE :
Lens et Webpage utilisent le même pool de crédits Serper (voir la note
dans sous_module_ooc.py). Ce script réutilise donc le MÊME fichier de
compteur (etat_credits_serper.json) que etape_recherche_inversee.py —
le budget --limite-credits porte sur la consommation cumulée des deux
étapes.

COÛT VARIABLE (2 à 6 crédits par page, contrairement à la recherche
inversée qui coûte un montant fixe) :
Quand Serper indique le coût réel dans sa réponse (champ "credits"), on
l'utilise directement — y compris quand la page cible n'a pas pu être
scrapée (anti-bot, etc.) : Serper facture aussi ces échecs et le champ
"credits" reste présent dans le corps de la réponse. Ce n'est que quand
ce chiffre est vraiment indisponible (timeout, erreur réseau, réponse
non-JSON) qu'on applique CREDITS_ESTIMATION_ECHEC (6 par défaut, le haut
de la fourchette 2-6) pour ne jamais sous-estimer la consommation.

DEUX TYPES D'ÉCHEC DISTINCTS :

1. Page réellement non scrapable (bloquée par anti-bot, erreur HTTP
   durable après les tentatives internes de sous_module_ooc.py, etc.)
   -> recuperation_webpage() renvoie None de façon "définitive".
   Retenter indéfiniment ne changerait rien.

2. Bug ponctuel de Serper qui renvoie une réponse vide/inexploitable
   alors qu'un nouvel essai immédiat fonctionne -> indiscernable du cas
   1 à partir de la seule valeur de retour (les deux donnent None). Ce
   script retente donc automatiquement NB_TENTATIVES fois, avec une
   courte pause entre chaque essai, avant de conclure à un échec
   définitif et d'écrire webpage=None dans le cache (ce qui arrête les
   nouvelles tentatives sur cette page lors des exécutions suivantes,
   sauf remise à zéro manuelle du champ). Conséquence assumée : les
   pages réellement bloquées consomment un peu plus de crédits
   (retentées inutilement) pour récupérer les cas de bug transitoire
   sans intervention manuelle.

Chaque page traitée est sauvegardée immédiatement (cache + compteur de
crédits) — aucune perte de travail en cas d'interruption, reprise exacte
au prochain lancement.

Utilisation :
    python etape_scraping.py
    python etape_scraping.py --limite-credits 500
    python etape_scraping.py --statut
    python etape_scraping.py --reset-compteur
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cache_utils import charger_cache, lister_ids_cache, sauvegarder_cache

from app.new_graph.node_1.n1_smOOC import (  # noqa: E402
    QuotaScrapingEpuiseError,
    recuperation_webpage_et_credits,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("etape_scraping")

NB_TENTATIVES = 2            # tentatives par page avant de conclure à un échec définitif
DELAI_ENTRE_TENTATIVES = 5   # secondes — bug transitoire, pas un rate-limit à respecter

CREDITS_ESTIMATION_ECHEC = 6  # estimation prudente (haut de la fourchette 2-6) quand le coût réel est inconnu
LIMITE_CREDITS_DEFAUT = 2000

COMPTEUR_PATH = "etat_credits_serper.json"  # PARTAGÉ avec etape_recherche_inversee.py


# ------------------------------------------------------------
# COMPTEUR DE CRÉDITS (partagé, persistant entre exécutions)
# ------------------------------------------------------------

def charger_compteur() -> Dict[str, Any]:
    if os.path.exists(COMPTEUR_PATH):
        with open(COMPTEUR_PATH, "r", encoding="utf-8") as f:
            etat = json.load(f)
    else:
        etat = {}
    etat.setdefault("credits_consommes", 0)
    etat.setdefault("recherches_effectuees", 0)
    etat.setdefault("pages_scrapees", 0)
    etat.setdefault("pages_echouees", 0)
    return etat


def sauvegarder_compteur(etat: Dict[str, Any]) -> None:
    chemin_tmp = COMPTEUR_PATH + ".tmp"
    with open(chemin_tmp, "w", encoding="utf-8") as f:
        json.dump(etat, f, ensure_ascii=False, indent=2)
    os.replace(chemin_tmp, COMPTEUR_PATH)


def reinitialiser_compteur() -> None:
    sauvegarder_compteur({
        "credits_consommes": 0,
        "recherches_effectuees": 0,
        "pages_scrapees": 0,
        "pages_echouees": 0,
    })
    logger.info("Compteur de crédits Serper remis à zéro.")


# ------------------------------------------------------------
# SCRAPING D'UNE PAGE, AVEC RETRY SUR ÉCHEC "SANS RÉPONSE EXPLOITABLE"
# ------------------------------------------------------------

def scraper_avec_retry(url: str, compteur: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Tente de scraper `url` jusqu'à NB_TENTATIVES fois. Retourne le dict
    webpage dès qu'une tentative réussit, ou None si toutes échouent.
    Met à jour `compteur` (en place) à chaque tentative, réussie ou non.

    QuotaScrapingEpuiseError n'est jamais avalée ici : elle remonte
    telle quelle (épuisement réel du compte, à distinguer d'un échec
    sur cette page précise).
    """
    for tentative in range(1, NB_TENTATIVES + 1):
        # QuotaScrapingEpuiseError se propage. credits_reels est le coût
        # indiqué par Serper (succès OU échec de scraping de la page
        # cible) : on l'utilise dès qu'il est disponible, l'estimation
        # ne sert plus que de repli (timeout, erreur réseau, etc.).
        donnees, credits_reels = recuperation_webpage_et_credits(url)
        compteur["credits_consommes"] += (
            credits_reels if isinstance(credits_reels, int) else CREDITS_ESTIMATION_ECHEC
        )

        if donnees is not None:
            return donnees

        if tentative < NB_TENTATIVES:
            logger.info(
                "[Scraping] Tentative %d/%d sans résultat exploitable pour %s — nouvel essai dans %ds.",
                tentative, NB_TENTATIVES, url, DELAI_ENTRE_TENTATIVES,
            )
            time.sleep(DELAI_ENTRE_TENTATIVES)

    return None


# ------------------------------------------------------------
# ÉTAPE
# ------------------------------------------------------------

def resultat_deja_traite(resultat: Dict[str, Any]) -> bool:
    return "webpage" in resultat


def fichier_complet(cache: Dict[str, Any]) -> bool:
    """Vrai si TOUS les résultats Lens du fichier ont déjà une clé "webpage"."""
    resultats = cache.get("resultats_lens")
    return bool(resultats) and all(resultat_deja_traite(r) for r in resultats)


def etape_scraping(limite_credits: int) -> None:
    compteur = charger_compteur()
    pages_traitees, deja_en_cache, en_attente_recherche = 0, 0, 0

    ids = sorted(lister_ids_cache())
    if not ids:
        logger.warning(
            "Aucun cache trouvé dans cache_test/ — lancez d'abord creation_cache.py "
            "puis etape_recherche_inversee.py."
        )
        return

    total_fichiers = len(ids)
    fichiers_complets = sum(
        1 for i in ids if (c := charger_cache(i)) is not None and fichier_complet(c)
    )
    logger.info(
        "[Scraping] Départ : %d/%d fichiers déjà complets (webpage sur tous les résultats), %d restants.",
        fichiers_complets, total_fichiers, total_fichiers - fichiers_complets,
    )

    for id_image in ids:
        cache = charger_cache(id_image)
        if cache is None or not cache.get("resultats_lens"):
            en_attente_recherche += 1
            continue

        modifie = False

        for resultat in cache["resultats_lens"]:
            if resultat_deja_traite(resultat):
                deja_en_cache += 1
                continue

            lien = resultat.get("lien")
            if not lien:
                resultat["webpage"] = None
                modifie = True
                continue

            if compteur["credits_consommes"] + CREDITS_ESTIMATION_ECHEC > limite_credits:
                if modifie:
                    sauvegarder_cache(id_image, cache)
                sauvegarder_compteur(compteur)
                logger.warning(
                    "LIMITE INTERNE DE CRÉDITS ATTEINTE (%d/%d) — arrêt propre. "
                    "%d pages scrapées dans cette exécution. Relancez plus tard, "
                    "augmentez --limite-credits, ou réinitialisez avec --reset-compteur "
                    "si votre quota Serper s'est renouvelé.",
                    compteur["credits_consommes"], limite_credits, pages_traitees,
                )
                return

            logger.info(
                "[Scraping] id_image=%s rang_lens=%s -> %s (crédits consommés jusqu'ici : %d/%d)",
                id_image, resultat.get("rang_lens"), lien,
                compteur["credits_consommes"], limite_credits,
            )

            try:
                webpage = scraper_avec_retry(lien, compteur)
            except QuotaScrapingEpuiseError as e:
                if modifie:
                    sauvegarder_cache(id_image, cache)
                sauvegarder_compteur(compteur)
                logger.error("QUOTA SERPER RÉELLEMENT ÉPUISÉ — arrêt propre : %s", e)
                logger.error("%d pages scrapées dans cette exécution. Relancez plus tard.", pages_traitees)
                return

            resultat["webpage"] = webpage
            if webpage is None:
                compteur["pages_echouees"] += 1
                logger.info(
                    "[Scraping] Échec définitif (après %d tentative(s)) : %s", NB_TENTATIVES, lien
                )
            else:
                compteur["pages_scrapees"] += 1

            modifie = True
            sauvegarder_compteur(compteur)  # sauvegarde à chaque page, pas seulement à chaque rumeur
            pages_traitees += 1

        if modifie:
            sauvegarder_cache(id_image, cache)
            if fichier_complet(cache):
                fichiers_complets += 1
                logger.info(
                    "[Scraping] id_image=%s complet -> %d/%d fichiers complets (%d restants).",
                    id_image, fichiers_complets, total_fichiers, total_fichiers - fichiers_complets,
                )

    logger.info(
        "[Scraping] terminé : %d pages traitées, %d déjà en cache, "
        "%d rumeurs en attente de la recherche inversée. Crédits consommés au total : %d. "
        "Fichiers complets : %d/%d (%d restants).",
        pages_traitees, deja_en_cache, en_attente_recherche, compteur["credits_consommes"],
        fichiers_complets, total_fichiers, total_fichiers - fichiers_complets,
    )


def afficher_statut(limite_credits: int) -> None:
    compteur = charger_compteur()
    ids = lister_ids_cache()

    pages_totales, pages_traitees = 0, 0
    fichiers_avec_resultats, fichiers_complets = 0, 0
    for id_image in ids:
        cache = charger_cache(id_image)
        if not cache or not cache.get("resultats_lens"):
            continue
        fichiers_avec_resultats += 1
        if fichier_complet(cache):
            fichiers_complets += 1
        for resultat in cache["resultats_lens"]:
            pages_totales += 1
            if resultat_deja_traite(resultat):
                pages_traitees += 1

    credits_restants = max(limite_credits - compteur["credits_consommes"], 0)

    print("\n--- Statut scraping ---")
    print(f"Crédits consommés (partagés avec la recherche inversée) : {compteur['credits_consommes']} / {limite_credits}")
    print(f"Pages scrapées avec succès : {compteur['pages_scrapees']}")
    print(f"Pages en échec définitif    : {compteur['pages_echouees']}")
    print(f"Pages traitées              : {pages_traitees} / {pages_totales}")
    print(f"Fichiers complets           : {fichiers_complets} / {fichiers_avec_resultats} ({fichiers_avec_resultats - fichiers_complets} restants)")
    print(f"Crédits restants (budget)   : {credits_restants}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scraping des pages des résultats de recherche inversée, résumable et bornée en budget de crédits."
    )
    parser.add_argument(
        "--limite-credits", type=int, default=LIMITE_CREDITS_DEFAUT,
        help=f"Budget de crédits Serper, partagé avec la recherche inversée (défaut : {LIMITE_CREDITS_DEFAUT}).",
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

    etape_scraping(args.limite_credits)


if __name__ == "__main__":
    main()