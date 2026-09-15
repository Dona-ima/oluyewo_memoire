"""
Identifie les fichiers de data/MFC-OOC/cache_s_ent dont le score S_ent est
`null` pour K=5 et/ou K=3 — signe qu'ils ont été écrits pendant un
épuisement de quota (avant le correctif de calcule_s_ent), donc avec des
données dégradées plutôt qu'un vrai résultat LLM.

Par défaut : liste seulement les fichiers concernés (rien n'est supprimé).
Avec --delete : les supprime, pour que calibration_etape_b_s_ent.py les
retraite automatiquement au prochain lancement (aucune perte : ils seront
recréés avec un vrai score).
"""

import argparse
import json
from pathlib import Path

S_ENT_DIR = "data/MFC-OOC/cache_s_ent"


def est_corrompu(donnees: dict) -> bool:
    s_ent_par_k = donnees.get("s_ent_par_k", {})
    if not s_ent_par_k:
        return True
    return any(v.get("score") is None for v in s_ent_par_k.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--delete", action="store_true",
        help="Supprime les fichiers corrompus trouvés (pour reprise automatique par l'étape B)."
    )
    args = parser.parse_args()

    fichiers_corrompus = []

    for chemin in sorted(Path(S_ENT_DIR).glob("*.json")):
        with open(chemin, "r", encoding="utf-8") as f:
            donnees = json.load(f)
        if est_corrompu(donnees):
            fichiers_corrompus.append((chemin, donnees.get("id_rumeur")))

    print(f"{len(fichiers_corrompus)} fichier(s) corrompu(s) trouvé(s) (score S_ent manquant) :\n")
    for chemin, id_rumeur in fichiers_corrompus:
        print(f"  - id_rumeur={id_rumeur} ({chemin.name})")

    if not fichiers_corrompus:
        print("\nRien à nettoyer.")
        return

    if args.delete:
        for chemin, _ in fichiers_corrompus:
            chemin.unlink()
        print(
            f"\n{len(fichiers_corrompus)} fichier(s) supprimé(s) — relance "
            "calibration_etape_b_s_ent.py pour les retraiter automatiquement."
        )
    else:
        print(
            "\nAucune suppression effectuée (mode lecture seule). "
            "Relance avec --delete pour les supprimer et forcer leur reprise."
        )


if __name__ == "__main__":
    main()