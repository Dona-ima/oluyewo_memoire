"""
etape_s_ooc.py

Étape 7 (finale) du pipeline de test : combine S_sem (etape_contexte_e5.py)
et S_ent (etape_s_ent.py) en un score unique S_OOC, puis classifie la
rumeur ("ooc" / "nooc" / "indetermine").

Reproduit exactement la logique de combinaison de
n1_OOC_calibration.py::sous_module_ooc (lignes ~1720-1740) :

    S_OOC = (alpha/(alpha+beta)) * S_sem + (beta/(alpha+beta)) * S_ent

    - si un seul des deux scores est disponible, S_OOC = ce score seul
    - si aucun n'est disponible, S_OOC = None -> classe "indetermine"

Valeurs calibrées utilisées (voir calibrage/) :
    alpha = beta = 0.5  (poids égal entre S_sem et S_ent)
    SEUIL_OOC = 0.40    (classifier_ooc : "ooc" si S_OOC < 0.40, sinon "nooc")

Étape purement locale : AUCUN appel réseau/LLM/HF — juste une lecture
des scores déjà en cache et un calcul arithmétique. Pas de gestion de
quota ni de limite de débit nécessaire ici.

Marqueur "déjà traité" : cache["classe_predite"] non None (la valeur
initiale posée par creation_cache_mfc-ooc.py). classifier_ooc()
renvoie toujours une chaîne non nulle ("ooc", "nooc" ou "indetermine"),
donc aucune ambiguïté possible ici (contrairement à S_ent/S_OOC qui
peuvent légitimement valoir None dans le résultat final).

Chaque rumeur traitée est sauvegardée immédiatement.

Utilisation :
    python etape_s_ooc.py
    python etape_s_ooc.py --statut
    python etape_s_ooc.py --force   # recalcule tout, même déjà fait
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cache_utils import charger_cache, lister_ids_cache, sauvegarder_cache

from app.new_graph.node_1.n1_smOOC import classifier_ooc  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("etape_s_ooc")

# Valeurs calibrées (voir calibrage/) — poids égal entre S_sem et S_ent.
ALPHA = 0.5
BETA = 0.5


def calcule_s_ooc(s_sem: Optional[float], s_ent: Optional[float]) -> Optional[float]:
    if s_sem is not None and s_ent is not None:
        somme = ALPHA + BETA
        return (ALPHA / somme) * s_sem + (BETA / somme) * s_ent
    if s_sem is not None:
        return s_sem
    if s_ent is not None:
        return s_ent
    return None


def s_ooc_deja_fait(cache: Dict[str, Any]) -> bool:
    return cache.get("classe_predite") is not None


def etape_s_ooc(force: bool = False) -> None:
    rumeurs_traitees, deja_en_cache = 0, 0

    ids = sorted(lister_ids_cache())
    if not ids:
        logger.warning("Aucun cache trouvé dans cache_test/.")
        return

    for id_image in ids:
        cache = charger_cache(id_image)
        if cache is None:
            continue

        if not force and s_ooc_deja_fait(cache):
            deja_en_cache += 1
            continue

        s_sem = cache.get("S_sem")
        s_ent = cache.get("S_ent")

        s_ooc = calcule_s_ooc(s_sem, s_ent)
        classe = classifier_ooc(s_ooc)

        cache["S_OOC"] = s_ooc
        cache["classe_predite"] = classe

        sauvegarder_cache(id_image, cache)
        rumeurs_traitees += 1
        logger.info(
            "[S_OOC] id_image=%s -> S_sem=%s, S_ent=%s, S_OOC=%s, classe=%s",
            id_image, s_sem, s_ent, s_ooc, classe,
        )

    logger.info(
        "[S_OOC] terminé : %d rumeurs traitées, %d déjà en cache.",
        rumeurs_traitees, deja_en_cache,
    )


def afficher_statut() -> None:
    ids = lister_ids_cache()

    total, complets = 0, 0
    compte_classes: Dict[str, int] = {}
    for id_image in ids:
        cache = charger_cache(id_image)
        if cache is None:
            continue
        total += 1
        if s_ooc_deja_fait(cache):
            complets += 1
            classe = cache.get("classe_predite")
            compte_classes[classe] = compte_classes.get(classe, 0) + 1

    print("\n--- Statut S_OOC / classe_predite ---")
    print(f"Fichiers complets : {complets} / {total} ({total - complets} restants)")
    for classe, n in sorted(compte_classes.items()):
        print(f"  {classe} : {n}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine S_sem et S_ent en S_OOC puis classifie chaque rumeur (ooc/nooc/indetermine)."
    )
    parser.add_argument(
        "--statut", action="store_true",
        help="Affiche l'avancement et la répartition des classes, sans rien exécuter.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recalcule S_OOC/classe_predite même pour les rumeurs déjà traitées.",
    )
    args = parser.parse_args()

    if args.statut:
        afficher_statut()
        return

    etape_s_ooc(force=args.force)


if __name__ == "__main__":
    main()
