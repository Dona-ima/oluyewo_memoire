"""
evaluer_performances.py

Évalue les performances du pipeline OOC sur cache_test/ : compare
classe_predite (calculée par etape_s_ooc.py) à la vérité terrain
déduite de "matched" (matched=False -> "ooc", matched=True -> "nooc"),
puis écrit un rapport dans calibration_resultats.md.

Métriques calculées (sklearn.metrics) :
    - Accuracy
    - F1 macro (moyenne non pondérée des F1 par classe)
    - F1(OOC) (F1 de la classe positive "ooc")
    - F1(NOOC) (fourni en complément, pour lire le F1 macro en contexte)
    - Matrice de confusion (2x2, classes ooc/nooc)

Les rumeurs dont classe_predite="indetermine" (S_OOC non calculable —
ni S_sem ni S_ent disponibles) sont EXCLUES des métriques (aucune
vérité terrain à comparer dans ce référentiel à 2 classes) mais
comptées à part et signalées dans le rapport.

Utilisation :
    python evaluer_performances.py
    python evaluer_performances.py --sortie autre_rapport.md
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cache_utils import charger_cache, lister_ids_cache  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("evaluer_performances")

LABELS = ["ooc", "nooc"]

# Paramètres calibrés utilisés pour produire ces résultats (voir calibrage/
# et n1_OOC_calibration.py) — informatif seulement, non recalculé ici.
PARAMETRES_CALIBRES = {
    "K": 5,
    "Méthode d'agrégation (S_sem)": "max",
    "alpha (poids de S_sem)": 0.5,
    "SEUIL_OOC": 0.40,
    "seuil_dominance": 0.4,
}


def verite_terrain(matched: Any) -> str:
    return "nooc" if matched is True else "ooc"


def collecter_predictions() -> Tuple[List[str], List[str], int, int]:
    """
    Renvoie (y_true, y_pred, nb_indetermine, nb_ignores) sur toutes les
    rumeurs de cache_test/ ayant une classe_predite ooc/nooc exploitable.
    nb_ignores compte les caches sans classe_predite du tout (étape
    etape_s_ooc.py pas encore lancée) ou sans champ "matched".
    """
    y_true, y_pred = [], []
    nb_indetermine, nb_ignores = 0, 0

    for id_image in sorted(lister_ids_cache()):
        cache = charger_cache(id_image)
        if cache is None:
            nb_ignores += 1
            continue

        classe = cache.get("classe_predite")
        matched = cache.get("matched")

        if classe is None or matched is None:
            nb_ignores += 1
            continue
        if classe == "indetermine":
            nb_indetermine += 1
            continue
        if classe not in LABELS:
            nb_ignores += 1
            continue

        y_true.append(verite_terrain(matched))
        y_pred.append(classe)

    return y_true, y_pred, nb_indetermine, nb_ignores


def formater_matrice_confusion(y_true: List[str], y_pred: List[str]) -> str:
    matrice = confusion_matrix(y_true, y_pred, labels=LABELS)
    lignes = [
        "| Réel \\ Prédit | ooc | nooc |",
        "|---|---|---|",
    ]
    for label, ligne in zip(LABELS, matrice):
        lignes.append(f"| **{label}** | {ligne[0]} | {ligne[1]} |")
    return "\n".join(lignes)


def generer_rapport(chemin_sortie: str) -> None:
    y_true, y_pred, nb_indetermine, nb_ignores = collecter_predictions()

    if not y_true:
        logger.error(
            "Aucune prédiction exploitable trouvée dans cache_test/ — "
            "lancez d'abord etape_s_ooc.py."
        )
        return

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, labels=LABELS, average="macro")
    f1_ooc = f1_score(y_true, y_pred, labels=LABELS, pos_label="ooc", average="binary")
    f1_nooc = f1_score(y_true, y_pred, labels=LABELS, pos_label="nooc", average="binary")
    rapport_detaille = classification_report(y_true, y_pred, labels=LABELS, digits=3)

    contenu = f"""# Résultats de calibration — pipeline OOC

Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} à partir de `cache_test/`
({len(y_true)} rumeurs évaluées sur {len(y_true) + nb_indetermine + nb_ignores} au total).

## Paramètres calibrés utilisés

| Paramètre | Valeur |
|---|---|
""" + "\n".join(f"| {k} | {v} |" for k, v in PARAMETRES_CALIBRES.items()) + f"""

## Métriques

| Métrique | Valeur |
|---|---|
| Accuracy | {accuracy:.3f} |
| F1 macro | {f1_macro:.3f} |
| F1 (OOC) | {f1_ooc:.3f} |
| F1 (NOOC) | {f1_nooc:.3f} |

## Matrice de confusion

{formater_matrice_confusion(y_true, y_pred)}

## Rapport détaillé (précision / rappel / F1 par classe)

```
{rapport_detaille}
```

## Notes

- Vérité terrain déduite du champ `matched` du jeu de test : `matched=False` -> `ooc`, `matched=True` -> `nooc`.
- {nb_indetermine} rumeur(s) classée(s) `indetermine` (S_OOC non calculable) exclue(s) des métriques ci-dessus.
- {nb_ignores} rumeur(s) ignorée(s) (étape `etape_s_ooc.py` pas encore lancée sur ce fichier, ou champ `matched` absent).
"""

    with open(chemin_sortie, "w", encoding="utf-8") as f:
        f.write(contenu)

    logger.info("Rapport écrit dans %s", chemin_sortie)
    logger.info(
        "Accuracy=%.3f | F1 macro=%.3f | F1(OOC)=%.3f | F1(NOOC)=%.3f",
        accuracy, f1_macro, f1_ooc, f1_nooc,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Évalue les performances du pipeline OOC sur cache_test/ et écrit un rapport Markdown."
    )
    parser.add_argument(
        "--sortie", type=str, default="calibration_resultats.md",
        help="Chemin du fichier Markdown de sortie (défaut : calibration_resultats.md, dans tests/).",
    )
    args = parser.parse_args()

    generer_rapport(args.sortie)


if __name__ == "__main__":
    main()
