"""
ÉTAPE B — Calcul de S_ent pour K=5 puis K=3, pour chaque ligne du cache.

Pour chaque ligne :
1. K=5 : recalcule les entités dominantes à partir de `entites_regroupees`
   (déjà normalisé sur les 5 résultats), au seuil SEUIL_DOMINANCE fixé —
   calcul PUR, aucun appel réseau. Puis appelle calcule_s_ent (1 appel
   LLM comparer_entites) pour obtenir S_ent à K=5.
2. K=3 : même chose, mais les entités dominantes sont recalculées en ne
   gardant que les rangs <= 3 (dénominateur = 3 au lieu de 5). Nouvel
   appel LLM pour ce K.

Écrit dans un cache SÉPARÉ (S_ENT_DIR), un fichier par id_rumeur, pour ne
jamais modifier/écraser le cache principal déjà construit. Reprend
automatiquement (fichiers déjà présents = sautés) et s'arrête proprement
sur un vrai épuisement de quota Gemini (QuotaLLMEpuiseError).

============================================================
À ADAPTER AVANT DE LANCER :
- Le chemin d'import du module
- CACHE_DIR / S_ENT_DIR
============================================================
"""

import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.new_graph.node_1.n1_smOOC import (  # noqa: E402
    schema_entites_vide,
    calcule_s_ent,
    QuotaLLMEpuiseError,
    logger,
)

#  ============================================================
# CONFIG — À ADAPTER
# ============================================================
 
CACHE_DIR = "data/MFC-OOC/cache_calibration"
S_ENT_DIR = "data/MFC-OOC/cache_s_ent"
 
K_VALUES = [5, 3]  # K=5 d'abord (denominateur natif), puis K=3
SEUIL_DOMINANCE = 0.4  # fixé pour cette passe — calibrable gratuitement à l'étape C
PAUSE_ENTRE_APPELS_SECONDES = 5  # entre chaque appel LLM (pas seulement chaque ligne)
 
# ============================================================
 
os.makedirs(S_ENT_DIR, exist_ok=True)
 
 
def dominance_pour_k(
    entites_regroupees: list,
    k: int,
    seuil: float = SEUIL_DOMINANCE
) -> dict:
    """
    Calcul PUR (aucun appel réseau) des entités dominantes pour un K donné,
    à partir de `entites_regroupees` (normalisé une seule fois à K_MAX=5
    lors de la construction du cache). On filtre les rangs > k et on
    recalcule la fréquence avec k comme dénominateur — équivalent à
    refaire la normalisation à ce K, sans le coût LLM.
    """
    dominantes = schema_entites_vide()
    if not entites_regroupees or k <= 0:
        return dominantes
 
    for item in entites_regroupees:
        rangs_dans_k = [r for r in item.get("rangs", []) if r <= k]
        if not rangs_dans_k:
            continue
        frequence = len(rangs_dans_k) / k
        if frequence >= seuil:
            dominantes[item["categorie"]].append(item["entite"])
 
    return dominantes
 
 
def deja_traite(id_rumeur) -> bool:
    return os.path.exists(os.path.join(S_ENT_DIR, f"{id_rumeur}.json"))
 
 
def traiter_ligne(ligne: dict) -> dict:
    """
    Calcule S_ent pour chaque K de K_VALUES. Si un vrai épuisement de
    quota survient en cours de route (ex. après K=5 mais pendant K=3),
    rien n'est écrit pour cette ligne — elle sera intégralement retraitée
    (K=5 ET K=3) à la prochaine exécution. Léger gaspillage possible d'un
    appel déjà réussi, mais garde la logique de reprise simple et fiable.
    """
    entites_rumeur = ligne.get("entites_rumeur", schema_entites_vide())
    entites_regroupees = ligne.get("entites_regroupees", [])
 
    s_ent_par_k = {}
 
    for k in K_VALUES:
        entites_dominantes_k = dominance_pour_k(entites_regroupees, k)
        s_ent = calcule_s_ent(entites_rumeur, entites_dominantes_k)
 
        s_ent_par_k[str(k)] = s_ent if s_ent is not None else {
            "score": None,
            "comparaison": {},
            "categories_comparables": []
        }
 
        time.sleep(PAUSE_ENTRE_APPELS_SECONDES)
 
    return {
        "id_rumeur": ligne["id_rumeur"],
        "id_image": ligne.get("id_image"),
        "s_ent_par_k": s_ent_par_k,
        "seuil_dominance_utilise": SEUIL_DOMINANCE,
    }
 
 
def main():
    fichiers = sorted(Path(CACHE_DIR).glob("*.json"))
    total = len(fichiers)
    deja_faits = nouveaux = erreurs = 0
 
    for i, chemin in enumerate(fichiers):
        with open(chemin, "r", encoding="utf-8") as f:
            ligne = json.load(f)
 
        id_rumeur = ligne["id_rumeur"]
 
        if deja_traite(id_rumeur):
            deja_faits += 1
            continue
 
        logger.info("[S_ent] Traitement %d/%d — id_rumeur=%s", i + 1, total, id_rumeur)
 
        try:
            resultat = traiter_ligne(ligne)
        except QuotaLLMEpuiseError as e:
            logger.error(
                "[S_ent] QUOTA GEMINI ÉPUISÉ (%s) — arrêt propre. "
                "%d nouvelles lignes traitées cette session (%d au total avec "
                "celles déjà en cache). Relance plus tard pour continuer.",
                e, nouveaux, deja_faits + nouveaux
            )
            break
        except Exception as e:
            logger.exception("[S_ent] Erreur inattendue sur id_rumeur=%s : %s", id_rumeur, e)
            erreurs += 1
            continue
 
        chemin_sortie = os.path.join(S_ENT_DIR, f"{id_rumeur}.json")
        with open(chemin_sortie, "w", encoding="utf-8") as f:
            json.dump(resultat, f, ensure_ascii=False, indent=2)
 
        nouveaux += 1
 
    logger.info(
        "[S_ent] Terminé pour cette session — %d déjà présents, %d nouveaux, %d erreurs, %d/%d au total.",
        deja_faits, nouveaux, erreurs, deja_faits + nouveaux, total
    )
 
 
if __name__ == "__main__":
    main()
 