"""
Sélectionne 200 lignes équilibrées (100 OOC + 100 NOOC) parmi les 726
lignes du dataset de test complet, pour limiter le coût en quota
Serper/Gemini pendant la phase de test.

============================================================
À ADAPTER :
- DATASET_PATH / OUTPUT_PATH
- COL_MATCHED si le nom de colonne diffère
============================================================
"""

import pandas as pd

DATASET_PATH = "data/MFC-OOC/test_mfc-ooc.csv"
OUTPUT_PATH = "data/MFC-OOC/test_mfc-ooc_200.csv"

COL_MATCHED = "matched"
N_PAR_CLASSE = 100
GRAINE_ALEATOIRE = 42


def main():
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset de test complet : {len(df)} lignes.")

    # matched=True -> NOOC, matched=False -> OOC (même convention que le
    # dataset de calibration).
    ooc = df[df[COL_MATCHED] == False]  # noqa: E712
    nooc = df[df[COL_MATCHED] == True]  # noqa: E712

    print(f"  OOC disponibles  : {len(ooc)}")
    print(f"  NOOC disponibles : {len(nooc)}")

    if len(ooc) < N_PAR_CLASSE or len(nooc) < N_PAR_CLASSE:
        raise ValueError(
            f"Pas assez de lignes pour {N_PAR_CLASSE} par classe "
            f"(OOC={len(ooc)}, NOOC={len(nooc)})."
        )

    ooc_echantillon = ooc.sample(n=N_PAR_CLASSE, random_state=GRAINE_ALEATOIRE)
    nooc_echantillon = nooc.sample(n=N_PAR_CLASSE, random_state=GRAINE_ALEATOIRE)

    echantillon = (
        pd.concat([ooc_echantillon, nooc_echantillon])
        .sample(frac=1, random_state=GRAINE_ALEATOIRE)  # mélange l'ordre
        .reset_index(drop=True)
    )

    echantillon.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{len(echantillon)} lignes sauvegardées dans {OUTPUT_PATH}")
    print(f"  OOC  : {(echantillon[COL_MATCHED] == False).sum()}")  # noqa: E712
    print(f"  NOOC : {(echantillon[COL_MATCHED] == True).sum()}")  # noqa: E712


if __name__ == "__main__":
    main()