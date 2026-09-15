"""
Construit, pour chaque ligne du dataset de calibration, un fichier JSON
unifié (nommé par id de la rumeur) contenant tout ce qui est nécessaire à
la calibration de K, seuil_dominance, w_sem/w_ent, méthode d'agrégation et
SEUIL_OOC en phase 2 — sans refaire aucun appel réseau/LLM à ce stade-là.

============================================================
À ADAPTER AVANT DE LANCER (section CONFIG ci-dessous) :
- Le chemin d'import du module (adapte selon l'emplacement réel de ton
  fichier corrigé — ici : app.new_graph.node_1.n1_OOC_calibration)
- DATASET_PATH et le format de chargement (CSV supposé par défaut)
- Les noms de colonnes réels de ton dataset (COL_*)
- Le pattern de nommage des fichiers dans SERPAPI_CACHE_DIR
============================================================

Reprise automatique : les fichiers déjà présents dans OUTPUT_DIR sont
sautés. Le script s'arrête PROPREMENT (sans écrire de cache partiel pour
la ligne en cours) dès qu'un vrai épuisement de quota Gemini ou
Serper.dev est détecté — relance-le simplement le lendemain (ou une fois
le quota reconstitué) pour continuer.
"""

import os
import sys
import json
import time
import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.new_graph.node_1.n1_smOOC import (  # noqa: E402
    logger,
    convertir_resultats_serpapi,
    selection_top_k_lens,
    recuperer_contenu_pages,
    construction_texte_contexte_et_entites,
    calcule_scores_e5,
    construction_entites_rumeur,
    normaliser_et_regrouper_entites,
    schema_entites_vide,
    QuotaLLMEpuiseError,
    QuotaScrapingEpuiseError,
)

# ============================================================
# CONFIG — À ADAPTER
# ============================================================

DATASET_PATH = "data/MFC-OOC/calibration_mfc-ooc_serpapi.csv"
SERPAPI_CACHE_DIR = "data/MFC-OOC/serpapi_cache"
OUTPUT_DIR = "data/MFC-OOC/cache_calibration"

# Les fichiers de cache SerpApi sont nommés par id de la RUMEUR (confirmé) —
# pas par id d'image.
def chemin_cache_serpapi(id_rumeur) -> str:
    return os.path.join(SERPAPI_CACHE_DIR, f"{id_rumeur}.json")

COL_ID_RUMEUR = "id"
COL_ID_IMAGE = "image_id"
COL_CAPTION = "caption"
COL_MATCHED = "matched"
# Pas de COL_URL_IMAGE : l'URL de l'image vient du JSON SerpApi en cache
# (clé "url_image"), pas du dataset — confirmé.

K_MAX = 5

# Pause entre chaque ligne pour rester sous la limite RPM de Gemini
# (~10-15 req/min en tier gratuit pour gemini-2.5-flash).
PAUSE_ENTRE_LIGNES_SECONDES = 15

# ============================================================
# FIN CONFIG
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


def deja_traite(id_rumeur) -> bool:
    return os.path.exists(os.path.join(OUTPUT_DIR, f"{id_rumeur}.json"))


def construire_entree_cache(row) -> dict:
    """
    Construit le dict de cache complet pour une ligne du dataset.
    Ne fait AUCUN appel Serper Lens (recherche_inversee) : réutilise le
    cache SerpApi déjà présent sur disque, converti au format du pipeline.

    Peut lever QuotaLLMEpuiseError ou QuotaScrapingEpuiseError — ces deux
    exceptions ne sont PAS attrapées ici, volontairement : elles doivent
    remonter jusqu'à main() pour déclencher un arrêt propre du batch.
    """

    # ------------------------------------------------------------
    # 0. Identifiants et champs du dataset — extraits AVANT toute
    #    utilisation (évite le bug de scoping vu précédemment : une
    #    variable utilisée avant sa propre ligne d'assignation dans
    #    la même fonction devient un UnboundLocalError en Python).
    # ------------------------------------------------------------
    id_rumeur = row[COL_ID_RUMEUR]
    image_id = row[COL_ID_IMAGE]
    caption = row[COL_CAPTION]
    matched = bool(row[COL_MATCHED])

    # ------------------------------------------------------------
    # 1. Recherche inversée : depuis le cache SerpApi, PAS d'appel Serper.
    #    Le cache est nommé par id_rumeur (confirmé).
    # ------------------------------------------------------------
    chemin_serpapi = chemin_cache_serpapi(id_rumeur)
    if not os.path.exists(chemin_serpapi):
        logger.error(
            "[Cache] Fichier SerpApi introuvable pour id_rumeur=%s (%s) — ligne sautée.",
            id_rumeur, chemin_serpapi
        )
        return None

    with open(chemin_serpapi, "r", encoding="utf-8") as f:
        donnees_serpapi = json.load(f)

    # L'URL de l'image vient du JSON en cache, pas du dataset (confirmé).
    url_image = donnees_serpapi.get("url_image", "")

    resultats = convertir_resultats_serpapi(donnees_serpapi)
    if not resultats:
        logger.warning(
            "[Cache] Aucun visual_match pour id_rumeur=%s — entrée vide construite quand même.",
            id_rumeur
        )
        resultats = []

    # ------------------------------------------------------------
    # 2. Top K_MAX (on cache large, on tronque en phase 2)
    # ------------------------------------------------------------
    resultats = selection_top_k_lens(resultats, k=K_MAX)

    # ------------------------------------------------------------
    # 3. Scraping des pages — peut lever QuotaScrapingEpuiseError,
    #    volontairement non catché ici.
    # ------------------------------------------------------------
    resultats = recuperer_contenu_pages(resultats)

    # ------------------------------------------------------------
    # 4. Contexte + entités par résultat — peut lever QuotaLLMEpuiseError.
    # ------------------------------------------------------------
    resultats = construction_texte_contexte_et_entites(resultats)

    # ------------------------------------------------------------
    # 5. Scores E5 (HF, pas de détection de quota dédiée pour l'instant)
    # ------------------------------------------------------------
    resultats = calcule_scores_e5(caption, resultats)

    # ------------------------------------------------------------
    # 6. Entités de la rumeur — peut lever QuotaLLMEpuiseError.
    # ------------------------------------------------------------
    entites_rumeur = construction_entites_rumeur(caption)

    # ------------------------------------------------------------
    # 7. Normalisation des entités des résultats — peut lever
    #    QuotaLLMEpuiseError. On cache le résultat BRUT (liste
    #    {categorie, entite, rangs}), pas une dominance déjà calculée
    #    à un seuil fixé (voir calcule_dominance_depuis_normalisation
    #    pour recalculer gratuitement en phase 2 pour tout seuil).
    # ------------------------------------------------------------
    entites_pour_llm = [
        {"rang_lens": r["rang_lens"], **r.get("entites", schema_entites_vide())}
        for r in resultats
    ]
    entites_regroupees = normaliser_et_regrouper_entites(entites_pour_llm)
    if entites_regroupees is None:
        entites_regroupees = []

    # ------------------------------------------------------------
    # Assemblage final
    # ------------------------------------------------------------
    return {
        "id_rumeur": id_rumeur,
        "id_image": image_id,
        "caption": caption,
        "url_image": url_image,
        "matched": matched,
        "resultats_lens": resultats,
        "entites_rumeur": entites_rumeur,
        "entites_regroupees": entites_regroupees,
        "meta": {
            "k_max": K_MAX,
            "cree_le": datetime.datetime.now().isoformat(),
        },
    }


def main():
    print("\n" + "=" * 70)
    print("DEBUT DE LA CONSTRUCTION DU CACHE DE CALIBRATION")
    print("=" * 70)
    df = pd.read_csv(DATASET_PATH)

    total = len(df)
    deja_faits = 0
    nouveaux = 0
    erreurs = 0

    for i, row in df.iterrows():
        id_rumeur = row[COL_ID_RUMEUR]

        if deja_traite(id_rumeur):
            deja_faits += 1
            continue

        logger.info("[Cache] Traitement %d/%d — id_rumeur=%s", i + 1, total, id_rumeur)

        try:
            entree = construire_entree_cache(row)

        except QuotaLLMEpuiseError as e:
            logger.error(
                "[Cache] QUOTA GEMINI ÉPUISÉ (%s) — arrêt propre. "
                "%d nouvelles lignes traitées cette session (%d au total avec "
                "celles déjà en cache). Relance plus tard pour continuer : "
                "la ligne en cours n'a PAS été écrite, elle sera retraitée "
                "depuis le début à la prochaine exécution.",
                e, nouveaux, deja_faits + nouveaux
            )
            break

        except QuotaScrapingEpuiseError as e:
            logger.error(
                "[Cache] QUOTA SERPER.DEV ÉPUISÉ (%s) — arrêt propre. "
                "%d nouvelles lignes traitées cette session (%d au total). "
                "Relance plus tard pour continuer.",
                e, nouveaux, deja_faits + nouveaux
            )
            break

        except Exception as e:
            logger.exception("[Cache] Erreur inattendue sur id_rumeur=%s : %s", id_rumeur, e)
            erreurs += 1
            continue

        if entree is None:
            erreurs += 1
            continue

        chemin_sortie = os.path.join(OUTPUT_DIR, f"{id_rumeur}.json")
        with open(chemin_sortie, "w", encoding="utf-8") as f:
            json.dump(entree, f, ensure_ascii=False, indent=2)

        nouveaux += 1

        time.sleep(PAUSE_ENTRE_LIGNES_SECONDES)

    logger.info(
        "[Cache] Terminé pour cette session — %d déjà présents, %d nouveaux, %d erreurs, %d/%d au total.",
        deja_faits, nouveaux, erreurs, deja_faits + nouveaux, total
    )


if __name__ == "__main__":
    main()