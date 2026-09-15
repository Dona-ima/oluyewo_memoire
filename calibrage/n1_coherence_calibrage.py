"""
Calibration du sous-module SigLIP2
==================================

Dataset attendu :
filename,image_path,premise,hypothesis,label

Classes utilisées :
    entailment    -> cohérente (1)
    contradiction -> incohérente (0)

La classe neutral est exclue du calibrage.

Objectif :
    trouver le seuil optimal de S_visual permettant de distinguer
    les paires cohérentes des paires incohérentes.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


DATASET_PATH = Path("data/SNLI-VE/calibration/calibration_snli-ve.csv")
IMAGE_BASE_DIR = DATASET_PATH.parent

SEUILS = [round(x / 100, 2) for x in range(20, 81, 1)]

LABELS_AUTORISES = {
    "entailment": 1,
    "contradiction": 0,
}

# ============================================================
# IMPORT DU SOUS-MODULE SIGLIP
# ============================================================

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.new_graph.node_1.n1_smCoherence import score_siglip


# ============================================================
# CHARGEMENT DATASET
# ============================================================

def charger_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset introuvable : {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    colonnes_requises = {"image_path", "hypothesis", "label"}
    colonnes_manquantes = colonnes_requises - set(df.columns)

    if colonnes_manquantes:
        raise ValueError(f"Colonnes manquantes : {colonnes_manquantes}")

    # Suppression de neutral & conversion des labels
    df = df[df["label"].isin(LABELS_AUTORISES.keys())].copy()
    df["label_binaire"] = df["label"].map(LABELS_AUTORISES)
    return df.reset_index(drop=True)


# ============================================================
# CALCUL DES SCORES
# ============================================================

def calculer_scores(df: pd.DataFrame) -> pd.DataFrame:
    resultats = []
    total = len(df)

    for index, row in df.iterrows():
        image_path = str(row["image_path"])
        hypothesis = str(row["hypothesis"])
        chemin_image = IMAGE_BASE_DIR / image_path

        print(f"\n[{index + 1}/{total}] {image_path}")

        if not chemin_image.exists():
            print(f"[WARNING] Image introuvable : {chemin_image}")
            continue

        try:
            from PIL import Image
            import torch
            from app.new_graph.node_1.n1_smCoherence import chargement_siglip2

            processor, model = chargement_siglip2()
            image = Image.open(chemin_image).convert("RGB")

            inputs = processor(
                text=[hypothesis],
                images=image,
                padding="max_length",
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = model(**inputs)

            logit = outputs.logits_per_image[0][0]
            score = torch.sigmoid(logit)
            s_visual = float(score.item())

            print(f"Label : {row['label']}")
            print(f"S_visual : {s_visual:.4f}")

            resultats.append({
                "filename": row.get("filename", ""),
                "image_path": image_path,
                "hypothesis": hypothesis,
                "label": row["label"],
                "label_binaire": int(row["label_binaire"]),
                "s_visual": s_visual,
            })

        except Exception as e:
            print(f"[ERROR] {image_path} : {e}")

    return pd.DataFrame(resultats)


# ============================================================
# EVALUATION POUR UN SEUIL
# ============================================================

def evaluer_seuil(y_true, scores, seuil: float) -> dict:
    # score >= seuil -> cohérent (1) | score < seuil -> incohérent (0)
    y_pred = [1 if score >= seuil else 0 for score in scores]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    matrice = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "seuil": seuil,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": int(matrice[0, 0]),
        "fp": int(matrice[0, 1]),
        "fn": int(matrice[1, 0]),
        "tp": int(matrice[1, 1]),
    }


# ============================================================
# CALIBRATION
# ============================================================

def calibrer(df_scores: pd.DataFrame) -> pd.DataFrame:
    y_true = df_scores["label_binaire"].tolist()
    scores = df_scores["s_visual"].tolist()

    resultats = [evaluer_seuil(y_true, scores, seuil) for seuil in SEUILS]
    return pd.DataFrame(resultats)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("CALIBRATION SIGLIP2")
    print("=" * 70)

    # Dataset
    df = charger_dataset()
    print(f"\nNombre total de lignes : {len(df)}")
    print(f"Entailment             : {(df['label'] == 'entailment').sum()}")
    print(f"Contradiction          : {(df['label'] == 'contradiction').sum()}")
    print("Neutral                : exclu")

    # Calcul des scores
    df_scores = calculer_scores(df)

    if df_scores.empty:
        raise RuntimeError("Aucun score SigLIP n'a pu être calculé.")

    # Sauvegarde des scores
    fichier_scores = DATASET_PATH.parent / "scores_siglip_calibration.csv"
    df_scores.to_csv(fichier_scores, index=False, encoding="utf-8")
    print(f"\nScores sauvegardés : {fichier_scores}")

    # Calibration
    df_metriques = calibrer(df_scores)
    meilleur = df_metriques.loc[df_metriques["f1"].idxmax()]

    # Résultats
    print("\n" + "=" * 70)
    print("RESULTATS DE CALIBRATION")
    print("=" * 70)
    print(f"Meilleur seuil : {meilleur['seuil']:.2f}")
    print(f"Accuracy       : {meilleur['accuracy']:.4f}")
    print(f"Precision      : {meilleur['precision']:.4f}")
    print(f"Recall         : {meilleur['recall']:.4f}")
    print(f"F1             : {meilleur['f1']:.4f}")

    print("\nMatrice de confusion :")
    print("                  Pred 0    Pred 1")
    print(f"Réel 0            {int(meilleur['tn']):8d} {int(meilleur['fp']):8d}")
    print(f"Réel 1            {int(meilleur['fn']):8d} {int(meilleur['tp']):8d}")

    # Sauvegarde des métriques
    fichier_metriques = DATASET_PATH.parent / "metriques_siglip_calibration.csv"
    df_metriques.to_csv(fichier_metriques, index=False, encoding="utf-8")
    print(f"\nMétriques sauvegardées : {fichier_metriques}")

    # Affichage des meilleurs seuils
    print("\n" + "=" * 70)
    print("TOP 10 DES SEUILS")
    print("=" * 70)
    print(df_metriques.sort_values("f1", ascending=False).head(10).to_string(index=False))

    print("\n" + "=" * 70)
    print("CALIBRATION TERMINEE")
    print("=" * 70)


if __name__ == "__main__":
    main()