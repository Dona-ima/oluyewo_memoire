"""
ÉTAPE C — Combinaison S_sem + S_ent et calibration finale.

Ne fait AUCUN appel réseau/LLM : S_sem est recalculé depuis le cache
principal (score_e5 par résultat), S_ent est lu depuis le cache construit
à l'étape B. Teste toutes les combinaisons (K, méthode d'agrégation,
alpha) et calibre SEUIL_OOC par validation croisée stratifiée à 5 plis.

IMPORTANT méthodologique : à chaque pli, le seuil optimal est choisi sur
le pli d'ENTRAÎNEMENT uniquement, puis évalué (F1) sur le pli de TEST —
jamais l'inverse. Le seuil n'est donc jamais optimisé sur les données qui
servent à le juger, ce qui évite le surapprentissage.

Métrique de classification : trois métriques calculées pour chaque
config — F1 macro (moyenne du F1 sur OOC et NOOC), accuracy, et F1(OOC)
(la classe d'intérêt du point de vue fact-checking, matched=False). Le
seuil optimal par pli reste choisi via F1 macro sur le pli
d'ENTRAÎNEMENT — les trois métriques sont ensuite rapportées sur le pli
de TEST à ce même seuil, pour rester comparables entre elles.
"""

import json
from pathlib import Path
from statistics import median as stat_median

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

# ============================================================
# CONFIG — À ADAPTER
# ============================================================

CACHE_DIR = "data/MFC-OOC/cache_calibration"
S_ENT_DIR = "data/MFC-OOC/cache_s_ent"

K_VALUES = [3, 5]
METHODES = ["mean", "median", "max", "top2_mean"]
ALPHA_VALUES = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]  # poids de S_sem (beta = 1 - alpha)
SEUILS_OOC = [round(x, 2) for x in np.arange(0.30, 0.71, 0.05)]
N_FOLDS = 5
GRAINE_ALEATOIRE = 42

# ============================================================


def charger_dossier(dossier: str) -> dict:
    """Charge tous les JSON d'un dossier, indexés par id_rumeur."""
    resultat = {}
    for chemin in Path(dossier).glob("*.json"):
        with open(chemin, "r", encoding="utf-8") as f:
            donnees = json.load(f)
        resultat[donnees["id_rumeur"]] = donnees
    return resultat


def aggrege(scores: list, methode: str):
    if not scores:
        return None
    if methode == "mean":
        return sum(scores) / len(scores)
    if methode == "median":
        return stat_median(scores)
    if methode == "max":
        return max(scores)
    if methode == "top2_mean":
        top2 = sorted(scores, reverse=True)[:2]
        return sum(top2) / len(top2)
    raise ValueError(f"Méthode inconnue : {methode}")


def s_sem_pour_ligne(ligne: dict, k: int, methode: str):
    resultats = sorted(ligne.get("resultats_lens", []), key=lambda r: r["rang_lens"])[:k]
    scores = [r["score_e5"] for r in resultats if r.get("score_e5") is not None]
    return aggrege(scores, methode)


def combine(s_sem_val, s_ent_val, alpha: float):
    """
    Combine S_sem et S_ent selon alpha (poids de S_sem). Si l'un des deux
    est indisponible, retombe sur l'autre seul plutôt que d'exclure la
    ligne — cohérent avec la logique de sous_module_ooc().
    """
    if s_sem_val is None and s_ent_val is None:
        return None
    if s_sem_val is None:
        return s_ent_val
    if s_ent_val is None:
        return s_sem_val
    return alpha * s_sem_val + (1 - alpha) * s_ent_val


def evalue_config(scores: np.ndarray, y: np.ndarray) -> dict:
    """
    Validation croisée 5-fold : seuil optimal choisi sur le pli
    d'entraînement (F1 macro), puis F1 macro / accuracy / F1(OOC) et les
    prédictions hors-échantillon (out-of-fold) rapportées sur le pli de
    test à ce même seuil. Les prédictions hors-échantillon (concaténées
    sur les 5 plis) servent ensuite à construire une matrice de confusion
    honnête — chaque ligne y est prédite avec un seuil jamais ajusté sur
    elle-même.

    Rappel de l'encodage : y=1 -> NOOC (matched=True), y=0 -> OOC
    (matched=False, la classe d'intérêt en fact-checking). f1_score avec
    pos_label=0 mesure donc F1 pour la détection d'OOC spécifiquement.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=GRAINE_ALEATOIRE)
    f1_macro_par_pli = []
    f1_ooc_par_pli = []
    accuracy_par_pli = []
    seuils_choisis = []
    y_true_oof = []
    y_pred_oof = []

    for idx_train, idx_test in skf.split(scores.reshape(-1, 1), y):
        if len(set(y[idx_train])) < 2 or len(set(y[idx_test])) < 2:
            continue

        meilleur_f1_train = -1.0
        meilleur_seuil = SEUILS_OOC[len(SEUILS_OOC) // 2]

        for seuil in SEUILS_OOC:
            pred_train = (scores[idx_train] >= seuil).astype(int)
            f1_train = f1_score(y[idx_train], pred_train, average="macro", zero_division=0)
            if f1_train > meilleur_f1_train:
                meilleur_f1_train = f1_train
                meilleur_seuil = seuil

        pred_test = (scores[idx_test] >= meilleur_seuil).astype(int)

        f1_macro_par_pli.append(f1_score(y[idx_test], pred_test, average="macro", zero_division=0))
        f1_ooc_par_pli.append(f1_score(y[idx_test], pred_test, pos_label=0, zero_division=0))
        accuracy_par_pli.append(accuracy_score(y[idx_test], pred_test))
        seuils_choisis.append(meilleur_seuil)

        y_true_oof.extend(y[idx_test].tolist())
        y_pred_oof.extend(pred_test.tolist())

    if not f1_macro_par_pli:
        return {
            "f1_moyen": None, "f1_ecart_type": None, "seuil_moyen": None,
            "accuracy_moyen": None, "f1_ooc_moyen": None,
            "y_true_oof": [], "y_pred_oof": [],
        }

    return {
        "f1_moyen": float(np.mean(f1_macro_par_pli)),
        "f1_ecart_type": float(np.std(f1_macro_par_pli)),
        "accuracy_moyen": float(np.mean(accuracy_par_pli)),
        "f1_ooc_moyen": float(np.mean(f1_ooc_par_pli)),
        "seuil_moyen": float(np.mean(seuils_choisis)),
        "y_true_oof": y_true_oof,
        "y_pred_oof": y_pred_oof,
    }


def main():
    lignes = charger_dossier(CACHE_DIR)
    s_ent_data = charger_dossier(S_ENT_DIR)

    ids_communs = sorted(set(lignes) & set(s_ent_data))
    print(f"{len(ids_communs)} lignes disposant à la fois de S_sem et S_ent.")
    if len(ids_communs) < len(lignes):
        print(f"({len(lignes) - len(ids_communs)} lignes n'ont pas encore de S_ent — ignorées pour l'instant.)\n")

    if not ids_communs:
        print("Aucune ligne exploitable — lance d'abord l'étape B.")
        return

    y_toutes = np.array([1 if lignes[i]["matched"] else 0 for i in ids_communs])

    resultats_grid = []

    for k in K_VALUES:
        s_ent_scores = [
            s_ent_data[i]["s_ent_par_k"].get(str(k), {}).get("score")
            for i in ids_communs
        ]

        for methode in METHODES:
            s_sem_scores = [s_sem_pour_ligne(lignes[i], k, methode) for i in ids_communs]

            for alpha in ALPHA_VALUES:
                scores, y_valides = [], []
                for s_sem_val, s_ent_val, label in zip(s_sem_scores, s_ent_scores, y_toutes):
                    score = combine(s_sem_val, s_ent_val, alpha)
                    if score is not None:
                        scores.append(score)
                        y_valides.append(label)

                if len(scores) < N_FOLDS * 2 or len(set(y_valides)) < 2:
                    continue

                metriques = evalue_config(np.array(scores), np.array(y_valides))
                if metriques["f1_moyen"] is None:
                    continue

                resultats_grid.append({
                    "k": k, "methode": methode, "alpha": alpha,
                    **metriques, "n": len(scores),
                })

    resultats_grid.sort(key=lambda r: r["f1_moyen"], reverse=True)

    print(f"\n{'K':<4}{'Méthode':<12}{'alpha':<8}{'F1 macro':<11}{'Accuracy':<11}{'F1(OOC)':<11}{'seuil~':<9}{'n':<6}")
    print("-" * 75)
    for r in resultats_grid[:20]:
        print(
            f"{r['k']:<4}{r['methode']:<12}{r['alpha']:<8}{r['f1_moyen']:<11.4f}"
            f"{r['accuracy_moyen']:<11.4f}{r['f1_ooc_moyen']:<11.4f}{r['seuil_moyen']:<9.3f}{r['n']:<6}"
        )

    if resultats_grid:
        m = resultats_grid[0]
        print(
            f"\n-> Meilleure config (triée par F1 macro) : K={m['k']}, méthode='{m['methode']}', "
            f"alpha={m['alpha']}, SEUIL_OOC≈{m['seuil_moyen']:.2f}"
        )
        print(
            f"   F1 macro={m['f1_moyen']:.4f}  |  Accuracy={m['accuracy_moyen']:.4f}  |  "
            f"F1(OOC)={m['f1_ooc_moyen']:.4f}"
        )

        # Matrice de confusion construite à partir des prédictions
        # hors-échantillon (out-of-fold) agrégées sur les 5 plis — chaque
        # ligne est prédite avec un seuil choisi sans jamais voir cette
        # ligne, donc la matrice reflète une performance honnête, pas un
        # ajustement sur les données d'évaluation elles-mêmes.
        y_true = np.array(m["y_true_oof"])
        y_pred = np.array(m["y_pred_oof"])
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        print("\nMatrice de confusion (0=OOC, 1=NOOC), sur prédictions hors-échantillon :")
        print(f"{'':12}{'Prédit OOC':<14}{'Prédit NOOC':<14}")
        print(f"{'Vrai OOC':<12}{cm[0][0]:<14}{cm[0][1]:<14}")
        print(f"{'Vrai NOOC':<12}{cm[1][0]:<14}{cm[1][1]:<14}")

        vn, fp, fn, vp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        print(
            f"\n({vn} vrais OOC bien détectés, {fp} OOC classés à tort NOOC [faux négatifs OOC], "
            f"{fn} NOOC classés à tort OOC [fausses alertes], {vp} vrais NOOC bien détectés)"
        )

        print(
            "\nNOTE : seuil_dominance est resté fixé à 0.4 dans cette passe "
            "(voir calibration_etape_b_s_ent.py). Pour le calibrer aussi, il "
            "faudrait relancer l'étape B avec d'autres valeurs de seuil_dominance "
            "au K retenu ici — nouveaux appels LLM nécessaires."
        )


if __name__ == "__main__":
    main()