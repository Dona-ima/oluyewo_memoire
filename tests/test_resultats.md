# Résultats de calibration — pipeline OOC

Généré le 2026-09-12 00:17:20 à partir de `cache_test/`
(196 rumeurs évaluées sur 196 au total).

## Paramètres calibrés utilisés

| Paramètre | Valeur |
|---|---|
| K | 5 |
| Méthode d'agrégation (S_sem) | max |
| alpha (poids de S_sem) | 0.5 |
| SEUIL_OOC | 0.4 |
| seuil_dominance | 0.4 |

## Métriques

| Métrique | Valeur |
|---|---|
| Accuracy | 0.816 |
| F1 macro | 0.816 |
| F1 (OOC) | 0.824 |
| F1 (NOOC) | 0.809 |

## Matrice de confusion

| Réel \ Prédit | ooc | nooc |
|---|---|---|
| **ooc** | 84 | 14 |
| **nooc** | 22 | 76 |

## Rapport détaillé (précision / rappel / F1 par classe)

```
              precision    recall  f1-score   support

         ooc      0.792     0.857     0.824        98
        nooc      0.844     0.776     0.809        98

    accuracy                          0.816       196
   macro avg      0.818     0.816     0.816       196
weighted avg      0.818     0.816     0.816       196

```

## Notes

- Vérité terrain déduite du champ `matched` du jeu de test : `matched=False` -> `ooc`, `matched=True` -> `nooc`.
- 0 rumeur(s) classée(s) `indetermine` (S_OOC non calculable) exclue(s) des métriques ci-dessus.
- 0 rumeur(s) ignorée(s) (étape `etape_s_ooc.py` pas encore lancée sur ce fichier, ou champ `matched` absent).
