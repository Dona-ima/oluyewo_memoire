import json
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from app.new_graph.node_1.n1_smOOC import (
    filtre_resultats_exploitables,
    selection_top_k_lens,
    get_hf_embedding,
    normaliser_embedding,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("data")

DATASET_PATH = (
    DATASET_DIR
    / "MFC-OOC"
    / "calibration_mfc-ooc_serpapi.csv"
)

CACHE_DIR = (
    DATASET_DIR
    / "MFC-OOC"
    / "serpapi_cache"
)

OUTPUT_INDIVIDUELS = (
    DATASET_DIR
    / "MFC-OOC"
    / "resultats_individuels_ooc.csv"
)

OUTPUT_BENCHMARK = (
    DATASET_DIR
    / "MFC-OOC"
    / "resultats_benchmark_ooc.csv"
)

# ------------------------------------------------------------
# K
# ------------------------------------------------------------

K_VALUES = list(range(1, 11))

# ------------------------------------------------------------
# AGRÉGATIONS
# ------------------------------------------------------------

METHODES = [
    "moyenne",
    "mediane",
    "min",
    "max",
]

# ------------------------------------------------------------
# SEUILS
# ------------------------------------------------------------

SEUILS = [
    round(x / 100, 2)
    for x in range(20, 81, 5)
]


# ============================================================
# AGRÉGATION
# ============================================================

def agreger_scores(scores, methode):
    """
    Agrège les scores E5 correspondant au K sélectionné.
    """

    if not scores:
        return None

    if methode == "moyenne":
        return float(sum(scores) / len(scores))

    if methode == "mediane":
        return float(pd.Series(scores).median())

    if methode == "min":
        return float(min(scores))

    if methode == "max":
        return float(max(scores))

    raise ValueError(
        f"Méthode d'agrégation inconnue : {methode}"
    )


# ============================================================
# CHARGEMENT DU CACHE SERPAPI
# ============================================================

def charger_cache(exemple_id):
    chemin = CACHE_DIR / f"{exemple_id}.json"

    if not chemin.exists():
        return None

    try:
        with open(
            chemin,
            "r",
            encoding="utf-8"
        ) as fichier:
            return json.load(fichier)

    except Exception as e:
        print(
            f"[ERROR] Lecture cache "
            f"{exemple_id}: {e}"
        )
        return None


# ============================================================
# EXTRACTION DES RÉSULTATS
# ============================================================

def extraire_resultats_cache(cache_data):
    """
    Reproduit la logique de recherche_inversee()
    à partir des résultats déjà sauvegardés.
    """

    if not cache_data:
        return []

    donnees = cache_data.get(
        "resultats_serpapi",
        {}
    )

    resultats_par_lien = {}

    # ========================================================
    # VISUAL MATCHES
    # ========================================================

    visual_matches = donnees.get(
        "visual_matches",
        []
    )

    for rang, match in enumerate(
        visual_matches,
        start=1
    ):

        lien = match.get("link")

        if not lien:
            continue

        resultats_par_lien[lien] = {
            "titre": match.get("title"),
            "snippet": None,
            "source": match.get("source"),
            "lien": lien,
            "date": None,
            "rang_lens": rang,
            "brut": match,
        }

    # ========================================================
    # ABOUT THIS IMAGE
    # ========================================================

    about_this_image = donnees.get(
        "about_this_image",
        {}
    )

    for section in about_this_image.get(
        "sections",
        []
    ):

        page_results = section.get(
            "page_results",
            []
        )

        for page_rank, page_result in enumerate(
            page_results,
            start=1
        ):

            lien = page_result.get("link")

            if not lien:
                continue

            # ------------------------------------------------
            # Résultat déjà présent
            # ------------------------------------------------

            if lien in resultats_par_lien:

                existant = resultats_par_lien[lien]

                existant["titre"] = (
                    page_result.get("title")
                    or existant.get("titre")
                )

                existant["snippet"] = (
                    page_result.get("snippet")
                    or existant.get("snippet")
                )

                existant["source"] = (
                    page_result.get("source")
                    or existant.get("source")
                )

                existant["date"] = (
                    page_result.get("date")
                    or existant.get("date")
                )

                existant["brut_about"] = page_result

            # ------------------------------------------------
            # Nouveau résultat
            # ------------------------------------------------

            else:

                resultats_par_lien[lien] = {
                    "titre": page_result.get("title"),
                    "snippet": page_result.get("snippet"),
                    "source": page_result.get("source"),
                    "lien": lien,
                    "date": page_result.get("date"),
                    "rang_lens": page_rank,
                    "brut": page_result,
                }

    # ========================================================
    # TRI PAR RANG LENS
    # ========================================================

    return sorted(
        resultats_par_lien.values(),
        key=lambda x: x.get(
            "rang_lens",
            float("inf")
        ),
    )


# ============================================================
# CALCUL DES SCORES E5
# ============================================================

def calculer_scores_e5(
    rumor_text,
    resultats_top_10,
):
    """
    Calcule les scores E5 UNE SEULE FOIS pour les
    résultats disponibles parmi les 10 premiers.

    Ces scores seront ensuite réutilisés pour K=1...10.
    """

    if not resultats_top_10:
        return []

    resultats_valides = [
        resultat
        for resultat in resultats_top_10
        if resultat.get("texte_compare")
    ]

    if not resultats_valides:
        return []

    # --------------------------------------------------------
    # Construction des textes E5
    # --------------------------------------------------------

    textes = [
        f"query: {rumor_text}"
    ]

    textes.extend(
        f"passage: {resultat['texte_compare']}"
        for resultat in resultats_valides
    )

    try:
        embeddings = get_hf_embedding(textes)

    except Exception as e:

        print(
            f"[ERROR] Calcul E5 : {e}"
        )

        return []

    # --------------------------------------------------------
    # Normalisation
    # --------------------------------------------------------

    embedding_rumeur = normaliser_embedding(
        embeddings[0]
    )

    embeddings_contextes = [
        normaliser_embedding(embedding)
        for embedding in embeddings[1:]
    ]

    # --------------------------------------------------------
    # Similarité cosinus
    # --------------------------------------------------------

    resultats_scores = []

    for resultat, embedding_contexte in zip(
        resultats_valides,
        embeddings_contextes,
    ):

        score = sum(
            a * b
            for a, b in zip(
                embedding_rumeur,
                embedding_contexte,
            )
        )

        resultat_copie = dict(
            resultat
        )

        resultat_copie["e5_score"] = float(
            score
        )

        resultats_scores.append(
            resultat_copie
        )

    return resultats_scores


# ============================================================
# PRÉPARATION D'UNE IMAGE
# ============================================================

def preparer_exemple(row):
    """
    Pipeline effectué UNE SEULE FOIS par image :

        cache
          ↓
        extraction
          ↓
        filtrage
          ↓
        sélection TOP 10
          ↓
        calcul E5

    Ensuite les différents K utilisent ces mêmes scores.
    """

    exemple_id = str(
        row["id"]
    )

    rumor_text = str(
        row["caption"]
    )

    matched = bool(
        row["matched"]
    )

    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    cache_data = charger_cache(
        exemple_id
    )

    if cache_data is None:

        return {
            "id": exemple_id,
            "matched": matched,
            "classe": (
                "NOOC"
                if matched
                else "OOC"
            ),
            "nb_resultats_serpapi": 0,
            "nb_resultats_exploitables": 0,
            "resultats_top_10": [],
            "scores_e5": [],
        }

    # --------------------------------------------------------
    # EXTRACTION
    # --------------------------------------------------------

    resultats = extraire_resultats_cache(
        cache_data
    )

    nb_resultats_serpapi = len(
        resultats
    )

    # --------------------------------------------------------
    # FILTRAGE
    # --------------------------------------------------------

    resultats_exploitables, _ = (
        filtre_resultats_exploitables(
            resultats
        )
    )

    nb_resultats_exploitables = len(
        resultats_exploitables
    )

    if not resultats_exploitables:

        return {
            "id": exemple_id,
            "matched": matched,
            "classe": (
                "NOOC"
                if matched
                else "OOC"
            ),
            "nb_resultats_serpapi": (
                nb_resultats_serpapi
            ),
            "nb_resultats_exploitables": 0,
            "resultats_top_10": [],
            "scores_e5": [],
        }

    # --------------------------------------------------------
    # TOP 10
    # --------------------------------------------------------

    resultats_top_10 = selection_top_k_lens(
        resultats_exploitables,
        k=10,
    )

    # --------------------------------------------------------
    # E5 UNE SEULE FOIS
    # --------------------------------------------------------

    resultats_scores = calculer_scores_e5(
        rumor_text,
        resultats_top_10,
    )

    return {
        "id": exemple_id,
        "matched": matched,
        "classe": (
            "NOOC"
            if matched
            else "OOC"
        ),
        "nb_resultats_serpapi": (
            nb_resultats_serpapi
        ),
        "nb_resultats_exploitables": (
            nb_resultats_exploitables
        ),
        "resultats_top_10": (
            resultats_scores
        ),
        "scores_e5": [
            resultat["e5_score"]
            for resultat in resultats_scores
            if resultat.get("e5_score") is not None
        ],
    }


# ============================================================
# CHARGEMENT DU DATASET
# ============================================================

df = pd.read_csv(
    DATASET_PATH
)

df["id"] = df["id"].astype(str)

# ------------------------------------------------------------
# Garder uniquement les exemples ayant un cache
# ------------------------------------------------------------

ids_avec_cache = {
    chemin.stem
    for chemin in CACHE_DIR.glob("*.json")
}

df = df[
    df["id"].isin(ids_avec_cache)
].copy()

df.reset_index(
    drop=True,
    inplace=True
)

if df.empty:

    raise RuntimeError(
        "Aucun exemple du dataset ne possède "
        "de cache SerpApi."
    )


# ============================================================
# INFORMATIONS
# ============================================================

print()
print("=" * 70)
print("CALIBRATION OOC")
print("=" * 70)

print(
    f"Exemples avec cache : {len(df)}"
)

print(
    f"True  / NOOC        : "
    f"{df['matched'].astype(bool).sum()}"
)

print(
    f"False / OOC         : "
    f"{(~df['matched'].astype(bool)).sum()}"
)

print(
    f"K testés            : "
    f"{K_VALUES}"
)

print(
    f"Agrégations         : "
    f"{METHODES}"
)

print(
    f"Seuils              : "
    f"{SEUILS}"
)

print("=" * 70)


# ============================================================
# PHASE 1
# CALCUL E5 UNE SEULE FOIS PAR IMAGE
# ============================================================

print()
print("=" * 70)
print("PHASE 1 — CACHE + FILTRAGE + TOP 10 + E5")
print("=" * 70)

preparations = []

for index, row in df.iterrows():

    print(
        f"[{index + 1}/{len(df)}]"
    )

    preparation = preparer_exemple(
        row
    )

    preparations.append(
        preparation
    )


# ============================================================
# PHASE 2
# CONSTRUCTION DU CSV INDIVIDUEL
# ============================================================

print()
print("=" * 70)
print("PHASE 2 — CONSTRUCTION DES SCORES PAR K")
print("=" * 70)

resultats_individuels = []


for preparation in preparations:

    scores = preparation[
        "scores_e5"
    ]

    for k in K_VALUES:

        # ----------------------------------------------------
        # On prend uniquement les K premiers scores
        # ----------------------------------------------------

        scores_k = scores[:k]

        # ----------------------------------------------------
        # Agrégations
        # ----------------------------------------------------

        s_moyenne = agreger_scores(
            scores_k,
            "moyenne",
        )

        s_mediane = agreger_scores(
            scores_k,
            "mediane",
        )

        s_min = agreger_scores(
            scores_k,
            "min",
        )

        s_max = agreger_scores(
            scores_k,
            "max",
        )

        # ----------------------------------------------------
        # Ligne CSV
        # ----------------------------------------------------

        resultats_individuels.append({

            "id": preparation["id"],

            "matched": preparation["matched"],

            "classe": preparation["classe"],

            "K": k,

            "nb_resultats_serpapi": (
                preparation[
                    "nb_resultats_serpapi"
                ]
            ),

            "nb_resultats_exploitables": (
                preparation[
                    "nb_resultats_exploitables"
                ]
            ),

            "nb_scores_e5": len(
                scores_k
            ),

            # Liste des scores E5 utilisés
            "scores_e5": json.dumps(
                scores_k,
                ensure_ascii=False,
            ),

            # Scores agrégés
            "s_reverse_moyenne": (
                s_moyenne
            ),

            "s_reverse_mediane": (
                s_mediane
            ),

            "s_reverse_min": (
                s_min
            ),

            "s_reverse_max": (
                s_max
            ),
        })


# ============================================================
# SAUVEGARDE CSV INDIVIDUEL
# ============================================================

df_individuels = pd.DataFrame(
    resultats_individuels
)

df_individuels.to_csv(
    OUTPUT_INDIVIDUELS,
    index=False,
    encoding="utf-8-sig",
)

print()
print(
    f"CSV individuel : "
    f"{OUTPUT_INDIVIDUELS}"
)


# ============================================================
# PHASE 3
# CALIBRATION DES SEUILS
# ============================================================

print()
print("=" * 70)
print("PHASE 3 — CALIBRATION DES SEUILS")
print("=" * 70)

resultats_benchmark = []


for k in K_VALUES:

    df_k = df_individuels[
        df_individuels["K"] == k
    ].copy()

    # --------------------------------------------------------
    # VRAIS LABELS
    # True  = NOOC = 1
    # False = OOC  = 0
    # --------------------------------------------------------

    y_true = (
        df_k["matched"]
        .astype(bool)
        .astype(int)
        .tolist()
    )

    for methode in METHODES:

        colonne_score = (
            f"s_reverse_{methode}"
        )

        scores = df_k[
            colonne_score
        ].tolist()

        for seuil in SEUILS:

            y_pred = []

            for score in scores:

                if pd.isna(score):

                    y_pred.append(None)

                elif score < seuil:

                    # Score faible
                    # → OOC
                    y_pred.append(0)

                else:

                    # Score élevé
                    # → NOOC
                    y_pred.append(1)

            # ------------------------------------------------
            # Retirer les indéterminés
            # ------------------------------------------------

            indices_valides = [
                i
                for i, prediction
                in enumerate(y_pred)
                if prediction is not None
            ]

            if not indices_valides:
                continue

            y_true_valid = [
                y_true[i]
                for i in indices_valides
            ]

            y_pred_valid = [
                y_pred[i]
                for i in indices_valides
            ]

            # ------------------------------------------------
            # MÉTRIQUES
            # ------------------------------------------------

            accuracy = accuracy_score(
                y_true_valid,
                y_pred_valid,
            )

            precision = precision_score(
                y_true_valid,
                y_pred_valid,
                zero_division=0,
            )

            recall = recall_score(
                y_true_valid,
                y_pred_valid,
                zero_division=0,
            )

            f1 = f1_score(
                y_true_valid,
                y_pred_valid,
                zero_division=0,
            )

            tn, fp, fn, tp = (
                confusion_matrix(
                    y_true_valid,
                    y_pred_valid,
                    labels=[0, 1],
                ).ravel()
            )

            # ------------------------------------------------
            # INDETERMINÉS
            # ------------------------------------------------

            nombre_indetermines = (
                len(y_pred)
                - len(indices_valides)
            )

            taux_indetermines = (
                nombre_indetermines
                / len(y_pred)
            )

            # ------------------------------------------------
            # STATISTIQUES DES SCORES
            # ------------------------------------------------

            scores_valides = [
                score
                for score in scores
                if not pd.isna(score)
            ]

            scores_ooc = [
                scores[i]
                for i in range(len(scores))
                if (
                    not pd.isna(scores[i])
                    and y_true[i] == 0
                )
            ]

            scores_nooc = [
                scores[i]
                for i in range(len(scores))
                if (
                    not pd.isna(scores[i])
                    and y_true[i] == 1
                )
            ]

            resultats_benchmark.append({

                "K": k,

                "agregation": methode,

                "seuil": seuil,

                # ----------------------------
                # Taille
                # ----------------------------

                "nombre_exemples_evalues": (
                    len(indices_valides)
                ),

                "nombre_indetermines": (
                    nombre_indetermines
                ),

                "taux_indetermines": (
                    taux_indetermines
                ),

                # ----------------------------
                # Métriques
                # ----------------------------

                "accuracy": accuracy,

                "precision": precision,

                "recall": recall,

                "f1": f1,

                # ----------------------------
                # Matrice confusion
                # ----------------------------

                "TN": tn,

                "FP": fp,

                "FN": fn,

                "TP": tp,

                # ----------------------------
                # Statistiques globales
                # ----------------------------

                "score_moyen_global": (
                    sum(scores_valides)
                    / len(scores_valides)
                    if scores_valides
                    else None
                ),

                # ----------------------------
                # OOC
                # ----------------------------

                "score_moyen_OOC": (
                    sum(scores_ooc)
                    / len(scores_ooc)
                    if scores_ooc
                    else None
                ),

                "score_median_OOC": (
                    float(
                        pd.Series(
                            scores_ooc
                        ).median()
                    )
                    if scores_ooc
                    else None
                ),

                "score_min_OOC": (
                    min(scores_ooc)
                    if scores_ooc
                    else None
                ),

                "score_max_OOC": (
                    max(scores_ooc)
                    if scores_ooc
                    else None
                ),

                # ----------------------------
                # NOOC
                # ----------------------------

                "score_moyen_NOOC": (
                    sum(scores_nooc)
                    / len(scores_nooc)
                    if scores_nooc
                    else None
                ),

                "score_median_NOOC": (
                    float(
                        pd.Series(
                            scores_nooc
                        ).median()
                    )
                    if scores_nooc
                    else None
                ),

                "score_min_NOOC": (
                    min(scores_nooc)
                    if scores_nooc
                    else None
                ),

                "score_max_NOOC": (
                    max(scores_nooc)
                    if scores_nooc
                    else None
                ),
            })


# ============================================================
# DATAFRAME FINAL
# ============================================================

df_benchmark = pd.DataFrame(
    resultats_benchmark
)

if df_benchmark.empty:

    raise RuntimeError(
        "Aucun résultat de benchmark."
    )


# ============================================================
# TRI
# ============================================================

df_benchmark = (
    df_benchmark
    .sort_values(
        by="f1",
        ascending=False,
    )
    .reset_index(drop=True)
)


# ============================================================
# TOP 20
# ============================================================

print()
print("=" * 70)
print("TOP 20 CONFIGURATIONS")
print("=" * 70)

print(
    df_benchmark
    .head(20)
    .to_string(index=False)
)


# ============================================================
# SAUVEGARDE
# ============================================================

df_benchmark.to_csv(
    OUTPUT_BENCHMARK,
    index=False,
    encoding="utf-8-sig",
)

print()
print(
    f"CSV benchmark : "
    f"{OUTPUT_BENCHMARK}"
)

print()
print("=" * 70)
print("CALIBRATION TERMINÉE")
print("=" * 70)