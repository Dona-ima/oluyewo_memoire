"""
creation_cache.py

Étape 0 du pipeline de test : initialise cache_test/<id_image>.json pour
chaque ligne du jeu de test qui n'a pas encore de cache.

Pour chaque ligne :
1. Si cache_test/<id_image>.json existe déjà -> ignoré (jamais écrasé,
   pour ne jamais remettre à zéro un cache déjà avancé dans le pipeline
   par les étapes suivantes).
2. Sinon :
   a. Retrouve le fichier image local dont le nom (sans extension)
      correspond à id_image, dans --dossier-images.
   b. Upload cette image sur Cloudinary pour obtenir une url_image
      publique. L'upload est IDEMPOTENT : si un asset avec ce
      public_id existe déjà côté Cloudinary (ex. relance après un
      crash), son URL est réutilisée sans re-uploader — pas de
      gaspillage de quota Cloudinary.
   c. Écrit le fichier cache avec la structure de départ (listes/dicts
      vides, à remplir par les étapes suivantes du pipeline).

Utilisation :
    python creation_cache.py --jeu-test donnees_test.json --dossier-images ./images
    python creation_cache.py --jeu-test donnees_test.csv --dossier-images ./images --limit 50

Format attendu du jeu de test (.csv ou .json — liste d'objets), clés
minimales : id, image_id, caption, matched.

Configuration Cloudinary (variables d'environnement, dans un .env ou
l'environnement du shell) :
    CLOUDINARY_URL="cloudinary://<api_key>:<api_secret>@<cloud_name>"
    (ou séparément : CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET)
    CLOUDINARY_FOLDER (optionnel, dossier Cloudinary de destination — ex. "oluyewo_benchmark")

Dépendances : pip install cloudinary python-dotenv
"""

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cloudinary
import cloudinary.api
import cloudinary.uploader
from cloudinary.exceptions import NotFound
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("creation_cache")

CACHE_DIR = "cache_test"
EXTENSIONS_IMAGE_VALIDES = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")


# ------------------------------------------------------------
# CONFIGURATION CLOUDINARY
# ------------------------------------------------------------

if os.getenv("CLOUDINARY_URL"):
    # cloudinary.config() lit automatiquement CLOUDINARY_URL depuis
    # l'environnement — rien à faire de plus ici.
    cloudinary.config(secure=True)
else:
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET"),
        secure=True,
    )
    if not all([
        os.getenv("CLOUDINARY_CLOUD_NAME"),
        os.getenv("CLOUDINARY_API_KEY"),
        os.getenv("CLOUDINARY_API_SECRET"),
    ]):
        raise RuntimeError(
            "Configuration Cloudinary manquante : définissez CLOUDINARY_URL, "
            "ou CLOUDINARY_CLOUD_NAME + CLOUDINARY_API_KEY + CLOUDINARY_API_SECRET "
            "(dans un fichier .env ou l'environnement)."
        )

CLOUDINARY_FOLDER = os.getenv("CLOUDINARY_FOLDER", "").strip()


# ------------------------------------------------------------
# STRUCTURE DE CACHE INITIALE
# ------------------------------------------------------------

def chemin_cache(id_image: Any) -> str:
    return os.path.join(CACHE_DIR, f"{id_image}.json")


def structure_cache_vide(
    id_rumeur: Any,
    id_image: Any,
    caption: str,
    url_image: str,
    matched: Any,
) -> Dict[str, Any]:
    """
    Structure de départ d'un cache de test. `resultats_lens` est une
    liste VIDE ici : le schéma détaillé de chaque élément (titre,
    source, lien, image_url, thumbnail_url, rang_lens, webpage,
    texte_contexte, niveau_contexte, entites, score_e5) est rempli par
    l'étape de recherche inversée / scraping, pas par ce script.
    """
    return {
        "id_rumeur": id_rumeur,
        "id_image": id_image,
        "caption": caption,
        "url_image": url_image,
        "matched": matched,
        "resultats_lens": [],
        "entites_rumeur": {},
        "entites_dominantes": [],
        "comparaison_entites": {},
        "S_sem": None,
        "S_ent": None,
        "S_OOC": None,
        "classe_predite": None,
    }


def sauvegarder_cache_atomique(id_image: Any, donnees: Dict[str, Any]) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    chemin = chemin_cache(id_image)
    chemin_tmp = chemin + ".tmp"
    with open(chemin_tmp, "w", encoding="utf-8") as f:
        json.dump(donnees, f, ensure_ascii=False, indent=2)
    os.replace(chemin_tmp, chemin)


# ------------------------------------------------------------
# RECHERCHE DE L'IMAGE LOCALE
# ------------------------------------------------------------

def trouver_image_locale(dossier_images: str, id_image: Any) -> Optional[Path]:
    dossier = Path(dossier_images)

    # Recherche rapide par extensions connues.
    for ext in EXTENSIONS_IMAGE_VALIDES:
        candidat = dossier / f"{id_image}{ext}"
        if candidat.exists():
            return candidat

    # Repli : extension inconnue mais nom de fichier (sans extension)
    # identique à id_image.
    if dossier.is_dir():
        for fichier in dossier.iterdir():
            if fichier.is_file() and fichier.stem == str(id_image):
                return fichier

    return None


# ------------------------------------------------------------
# UPLOAD CLOUDINARY (IDEMPOTENT)
# ------------------------------------------------------------

def _public_id(id_image: Any) -> str:
    base = str(id_image)
    return f"{CLOUDINARY_FOLDER}/{base}" if CLOUDINARY_FOLDER else base


def obtenir_url_cloudinary(chemin_image: Path, id_image: Any) -> str:
    public_id = _public_id(id_image)

    # Idempotent : si l'asset existe déjà côté Cloudinary (ex. relance
    # après un crash survenu juste après un upload précédent), on
    # réutilise son URL plutôt que de re-uploader — évite de gaspiller
    # du quota Cloudinary.
    try:
        ressource = cloudinary.api.resource(public_id, resource_type="image", type="upload")
        logger.info("[Cloudinary] Asset déjà présent pour %s — réutilisation.", public_id)
        return ressource["secure_url"]
    except NotFound:
        pass

    logger.info("[Cloudinary] Upload de %s (public_id=%s)", chemin_image, public_id)
    resultat = cloudinary.uploader.upload(
        str(chemin_image),
        public_id=public_id,
        overwrite=False,
        unique_filename=False,
    )
    return resultat["secure_url"]


# ------------------------------------------------------------
# CHARGEMENT DU JEU DE TEST (.csv OU .json)
# ------------------------------------------------------------

def charger_jeu_test(chemin: str) -> List[Dict[str, Any]]:
    chemin_p = Path(chemin)

    if chemin_p.suffix.lower() == ".json":
        with open(chemin_p, "r", encoding="utf-8") as f:
            donnees = json.load(f)
        if not isinstance(donnees, list):
            raise ValueError("Le fichier JSON du jeu de test doit contenir une liste d'objets.")
        lignes = donnees
    elif chemin_p.suffix.lower() == ".csv":
        with open(chemin_p, "r", encoding="utf-8") as f:
            lignes = list(csv.DictReader(f))
    else:
        raise ValueError(f"Format non supporté : {chemin_p.suffix} (attendu : .csv ou .json)")

    if not lignes:
        raise ValueError(f"Jeu de test vide : {chemin}")

    manquantes = {"id", "image_id", "caption"} - set(lignes[0].keys())
    if manquantes:
        raise ValueError(f"Colonnes/clés manquantes dans le jeu de test : {manquantes}")

    return lignes


def _normaliser_matched(valeur: Any) -> Any:
    if isinstance(valeur, str):
        return valeur.strip().lower() in ("true", "1", "vrai", "oui")
    return valeur


# ------------------------------------------------------------
# ORCHESTRATION
# ------------------------------------------------------------

def creer_caches(
    jeu_test: List[Dict[str, Any]],
    dossier_images: str,
    limite: Optional[int] = None,
) -> None:
    lignes = jeu_test if not limite else jeu_test[:limite]

    crees, deja_en_cache, images_introuvables, echecs_upload = 0, 0, 0, 0

    for ligne in lignes:
        id_image = ligne["image_id"]

        if os.path.exists(chemin_cache(id_image)):
            deja_en_cache += 1
            continue

        image_locale = trouver_image_locale(dossier_images, id_image)
        if image_locale is None:
            logger.warning(
                "[Image] Aucun fichier local trouvé pour id_image=%s dans %s — ignoré.",
                id_image, dossier_images,
            )
            images_introuvables += 1
            continue

        try:
            url_image = obtenir_url_cloudinary(image_locale, id_image)
        except Exception as e:
            logger.error("[Cloudinary] Échec de l'upload pour id_image=%s : %s", id_image, e)
            echecs_upload += 1
            continue

        cache = structure_cache_vide(
            id_rumeur=ligne["id"],
            id_image=id_image,
            caption=ligne["caption"],
            url_image=url_image,
            matched=_normaliser_matched(ligne.get("matched")),
        )

        sauvegarder_cache_atomique(id_image, cache)
        logger.info("[Cache] Créé : %s", chemin_cache(id_image))
        crees += 1

    logger.info(
        "[Bilan] %d caches créés, %d déjà présents, %d images locales introuvables, %d échecs d'upload.",
        crees, deja_en_cache, images_introuvables, echecs_upload,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialise le cache de test (upload Cloudinary inclus pour les images locales)."
    )
    parser.add_argument("--jeu-test", required=True, help="Chemin du jeu de test (.csv ou .json).")
    parser.add_argument(
        "--dossier-images", required=True,
        help="Dossier contenant les images locales, nommées <id_image>.<extension>.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limite le nombre de lignes traitées dans cette exécution (pour doser les uploads Cloudinary).",
    )
    args = parser.parse_args()

    jeu_test = charger_jeu_test(args.jeu_test)
    creer_caches(jeu_test, args.dossier_images, limite=args.limit)


if __name__ == "__main__":
    main()