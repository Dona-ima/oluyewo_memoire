import os
import shutil
import json
import logging
import time
import requests
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError

from app.services.storage_service import uploader_image
from app.graph.build import compiler_graphe
from app.auth import (
    hasher_password,
    obtenir_utilisateur_courant,
    obtenir_utilisateur_optionnel,
    verifier_password,
    creer_token,
)
# Constante ré-utilisée uniquement pour documenter la méthode d'agrégation
# réellement appliquée au score sémantique dans la réponse API (voir plus bas).
from app.graph.node_1.n1_smOOC import METHODE_AGREGATION_DEFAUT

load_dotenv()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.getenv("DATABASE_URL"):
        from app.database import initialiser_bdd
        try:
            initialiser_bdd()
        except OperationalError:
            logger.exception(
                "Connexion à PostgreSQL impossible au démarrage. "
                "Les comptes et l'historique resteront indisponibles."
            )
    yield


app = FastAPI(
    title="Oluyewo Backend API",
    description="Moteur de fact-checking multimodal basé sur une architecture multi-agents (LangGraph & Gemini)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(SQLAlchemyError)
async def gerer_erreur_base_de_donnees(_: Request, erreur: SQLAlchemyError):
    logger.error("Erreur lors de l'accès à PostgreSQL.", exc_info=erreur)
    return JSONResponse(
        status_code=503,
        content={"detail": "La base de données est indisponible. Réessayez plus tard."}
    )


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Alignée sur la limite annoncée côté frontend ("PNG ou JPG jusqu'à 10MB").
TAILLE_MAX_IMAGE_OCTETS = 10 * 1024 * 1024

graphe_oluyewo = compiler_graphe()

# Libellés affichés côté frontend pour chaque nœud du graphe — tenus en un
# seul endroit pour rester synchronisés avec les AGENTS définis dans
# verification.tsx.
LIBELLES_ETAPES = {
    "node_recherche_origine": "Recherche d'origine de l'image",
    "node_comparaison_visuelle": "Comparaison visuelle image/texte",
    "node_2": "Rédaction du rapport final",
}


class InscriptionRequest(BaseModel):
    nom: str = Field(min_length=1, max_length=100)
    prenom: str = Field(min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


def telecharger_image_depuis_url(url: str) -> str:
    """
    Télécharge côté serveur une image dont l'utilisateur a collé l'URL
    (plutôt qu'un fichier téléversé), la valide (type MIME, taille) et
    l'enregistre localement — même point d'entrée que l'upload de fichier
    ensuite (voir verify_rumor_stream), pour que la suite du traitement
    (upload Cloudinary) soit identique quel que soit le mode de soumission.

    Le téléchargement se fait côté serveur et non depuis le navigateur car la
    plupart des hébergeurs d'images bloquent les requêtes cross-origin
    (CORS) déclenchées depuis une page web tierce.

    Lève ValueError (→ HTTP 400, erreur imputable à l'utilisateur) plutôt que
    de laisser remonter l'exception réseau brute.
    """
    if not url or not url.strip():
        raise ValueError("URL d'image vide.")

    try:
        reponse = requests.get(url.strip(), stream=True, timeout=15)
    except requests.RequestException as e:
        raise ValueError(f"Impossible de récupérer l'image à cette URL : {e}")

    if not reponse.ok:
        raise ValueError(f"L'URL a répondu avec le statut {reponse.status_code}.")

    content_type = reponse.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        raise ValueError("L'URL fournie ne pointe pas vers une image.")

    extension = content_type.split("/")[-1].split(";")[0].strip() or "jpg"
    nom_unique = f"photo_url_{time.time()}.{extension}"
    chemin_local = os.path.join(UPLOAD_DIR, nom_unique)

    taille = 0
    try:
        with open(chemin_local, "wb") as f:
            for chunk in reponse.iter_content(chunk_size=8192):
                taille += len(chunk)
                if taille > TAILLE_MAX_IMAGE_OCTETS:
                    raise ValueError("L'image dépasse la taille maximale autorisée (10MB).")
                f.write(chunk)
    except ValueError:
        if os.path.exists(chemin_local):
            os.remove(chemin_local)
        raise

    return chemin_local


def construire_donnees_techniques_detail(resultat_graphe: dict) -> dict:
    """
    Construit un bloc de données techniques SOURCÉ DIRECTEMENT depuis l'état
    du graphe (résultats bruts de node_1), sans passer par la reconstruction
    JSON du LLM dans node_2. Objectif : donner au frontend des valeurs
    numériques fiables pour construire une démonstration de calcul fidèle
    (score par source pour S_sem, comparaison catégorie par catégorie pour
    S_ent), au lieu de l'approximation que Gemini reconstruit dans
    "donnees_techniques".
    """
    resultats_bruts = resultat_graphe.get("resultats", []) or []

    resultats_semantiques = [
        {
            "rang": r.get("rang_lens"),
            "source": r.get("source") or r.get("titre") or "Source inconnue",
            "lien": r.get("lien"),
            "contexte": r.get("texte_contexte"),
            "score_e5": r.get("score_e5"),
        }
        for r in resultats_bruts
    ]

    return {
        "classe_ooc": resultat_graphe.get("classe_ooc"),
        "score_semantique": resultat_graphe.get("s_sem_ooc"),
        "methode_agregation": METHODE_AGREGATION_DEFAUT,
        "resultats_semantiques": resultats_semantiques,

        "score_entites": resultat_graphe.get("s_ent_ooc"),
        "entites_rumeur": resultat_graphe.get("entites_rumeur_ooc", {}),
        "entites_dominantes": resultat_graphe.get("entites_dominantes_ooc", {}),
        "comparaison_categories": resultat_graphe.get("comparaison_entites_ooc", {}),

        "classe_visual": resultat_graphe.get("classe_visual"),
        "score_visual": resultat_graphe.get("s_visual"),
    }


def construire_reponse_finale(etat_final: dict, rumor_text: str, url_image: str) -> dict:
    """
    Assemble la réponse finale complète à partir de l'état final du graphe —
    utilisée aussi bien en fin de streaming qu'en cas de fallback.
    """
    rapport_llm = etat_final.get(
        "justification_json",
        {
            "resultat_utilisateur": {
                "titre": "Analyse indisponible",
                "resume": "Le système n'a pas pu générer le rapport d'analyse.",
                "analyse_contexte": {},
                "analyse_image_rumeur": {},
                "mise_en_relation": "",
                "conclusion": "",
                "avertissement": ""
            },
            "donnees_techniques": {}
        }
    )

    metrics = etat_final.get("performance_metrics", {})

    return {
        **rapport_llm,
        "classe_ooc": etat_final.get("classe_ooc"),
        "rumor_image": url_image,
        "rumor_text": rumor_text,
        "performance_metrics": metrics,
        "donnees_techniques_detail": construire_donnees_techniques_detail(etat_final),
    }


async def executer_graphe_stream(
    etat_initial: dict,
    rumor_text: str,
    url_image: str,
    user_id: int | None = None
):
    """
    Générateur asynchrone : exécute le graphe via astream(stream_mode="updates")
    et streame un événement NDJSON (une ligne = un objet JSON) à chaque nœud
    terminé, puis un événement final contenant le résultat complet.

    stream_mode="updates" fait remonter, à chaque super-step, un dict
    {nom_du_noeud: maj_partielle_du_state}. Comme node_recherche_origine et
    node_comparaison_visuelle tournent dans le même super-step (fan-out
    natif, voir build.py), un même événement peut contenir les deux — d'où
    la boucle sur event.items() plutôt que de supposer un seul nœud par
    événement.
    """
    etat_courant = dict(etat_initial)

    yield json.dumps({
        "type": "start",
        "etapes": list(LIBELLES_ETAPES.keys()),
    }) + "\n"

    try:
        config = {"configurable": {"thread_id": f"session_{time.time()}"}}
        async for event in graphe_oluyewo.astream(
            etat_initial, config=config, stream_mode="updates"
        ):
            for nom_noeud, maj in event.items():
                etat_courant.update(maj)
                yield json.dumps({
                    "type": "etape_terminee",
                    "etape": nom_noeud,
                    "libelle": LIBELLES_ETAPES.get(nom_noeud, nom_noeud),
                }) + "\n"

        reponse_finale = construire_reponse_finale(etat_courant, rumor_text, url_image)
        if user_id is not None:
            from app.database import sauvegarder_verification
            reponse_finale["verification_id"] = sauvegarder_verification(
                rumor_text=rumor_text,
                image_path=url_image,
                resultat_graphe=etat_courant,
                reponse_finale_json=reponse_finale,
                user_id=user_id,
            )
        yield json.dumps({"type": "resultat", "donnees": reponse_finale}) + "\n"

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield json.dumps({
            "type": "erreur",
            "detail": f"Erreur interne : {str(e)}",
        }) + "\n"


TYPES_IMAGE_AUTORISES = {"image/jpeg", "image/png", "image/webp"}


def fichier_fourni(image_file: UploadFile | None) -> bool:
    """Vrai seulement si un vrai fichier a été envoyé (pas un champ vide)."""
    return bool(image_file and image_file.filename)


def valider_fichier_image(image_file: UploadFile) -> None:
    if image_file.content_type not in TYPES_IMAGE_AUTORISES:
        raise ValueError("Le fichier doit être une image PNG, JPG ou WEBP.")
    image_file.file.seek(0, os.SEEK_END)
    taille = image_file.file.tell()
    image_file.file.seek(0)
    if taille == 0:
        raise ValueError("Le fichier image est vide.")
    if taille > TAILLE_MAX_IMAGE_OCTETS:
        raise ValueError("L'image dépasse la taille maximale autorisée (10MB).")


# --- Routes santé ---
@app.get("/")
def read_root():
    return {"status": "online", "application": "Oluyewo Fact-Checking Engine"}


@app.post("/auth/register", status_code=201)
def register_user(donnees: InscriptionRequest):
    from app.database import creer_utilisateur, obtenir_utilisateur_par_email

    nom = donnees.nom.strip()
    prenom = donnees.prenom.strip()
    if not nom or not prenom:
        raise HTTPException(status_code=422, detail="Le nom et le prénom sont obligatoires.")
    if len(donnees.password.encode("utf-8")) > 72:
        raise HTTPException(status_code=422, detail="Le mot de passe ne peut pas dépasser 72 octets.")

    email = str(donnees.email).lower()
    try:
        utilisateur_existant = obtenir_utilisateur_par_email(email)
    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail="La base de données n'est pas configurée."
        ) from error
    if utilisateur_existant:
        raise HTTPException(status_code=409, detail="Cette adresse e-mail est déjà utilisée.")

    try:
        user_id = creer_utilisateur(
            nom=nom,
            prenom=prenom,
            email=email,
            hashed_password=hasher_password(donnees.password),
        )
    except IntegrityError as error:
        raise HTTPException(status_code=409, detail="Cette adresse e-mail est déjà utilisée.") from error
    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail="La base de données n'est pas configurée."
        ) from error

    return {
        "message": "Compte créé avec succès.",
        "user": {
            "id": user_id,
            "nom": nom,
            "prenom": prenom,
            "email": email,
        }
    }


class ConnexionRequest(BaseModel):
    email: EmailStr
    password: str


@app.post("/auth/login")
def login_user(donnees: ConnexionRequest):
    from app.database import obtenir_utilisateur_par_email

    email = str(donnees.email).strip().lower()
    password = donnees.password

    try:
        user = obtenir_utilisateur_par_email(email)
    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail="La base de données n'est pas configurée."
        ) from error

    if not user or not verifier_password(password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="E-mail ou mot de passe incorrect.")

    try:
        token = creer_token({"sub": str(user["id"])})
    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail="L'authentification n'est pas configurée."
        ) from error

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "nom": user["nom"],
            "prenom": user["prenom"],
            "email": user["email"],
        },
    }

@app.get("/auth/me")
@app.get("/me")
def read_current_user(user: dict = Depends(obtenir_utilisateur_courant)):
    return user


@app.get("/history")
def read_verification_history(user: dict = Depends(obtenir_utilisateur_courant)):
    from app.database import obtenir_historique_utilisateur
    try:
        return obtenir_historique_utilisateur(user["id"])
    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail="La base de données n'est pas configurée."
        ) from error


@app.get("/history/{verification_id}")
def read_verification_detail(
    verification_id: int,
    user: dict = Depends(obtenir_utilisateur_courant)
):
    from app.database import obtenir_detail_verification
    try:
        verification = obtenir_detail_verification(verification_id, user["id"])
    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail="La base de données n'est pas configurée."
        ) from error
    if verification is None:
        raise HTTPException(status_code=404, detail="Vérification introuvable.")
    return verification


# --- Route principale (streaming) ---
@app.post("/verify-rumor/stream")
async def verify_rumor_stream(
    rumor_text: str = Form(...),
    image_file: UploadFile = File(None),
    image_url: str = Form(None),
    user: dict | None = Depends(obtenir_utilisateur_optionnel),
):
    if not rumor_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Le texte de la rumeur ne peut pas être vide."
        )

    a_fichier = fichier_fourni(image_file)
    a_url = bool(image_url and image_url.strip())

    if not a_fichier and not a_url:
        raise HTTPException(
            status_code=400,
            detail="Une image est requise : téléversez un fichier ou collez un lien."
        )
    if a_fichier and a_url:
        raise HTTPException(
            status_code=400,
            detail="Fournissez soit un fichier, soit un lien, pas les deux."
        )

    chemin_image_temporaire = ""
    url_image_cloudinary = ""
    actuel_date = datetime.now().strftime("%d %B %Y")

    # --- Gestion de l'image : on la fait AVANT de streamer, pour pouvoir
    # encore renvoyer une erreur HTTP classique si ça échoue (une fois le
    # streaming démarré, les en-têtes HTTP sont déjà envoyés et on ne peut
    # plus lever d'HTTPException proprement — voir le bloc except dans
    # executer_graphe_stream pour les erreurs survenant après ce point).
    try:
        if a_fichier:
            valider_fichier_image(image_file)
            nom_unique = f"photo_{time.time()}_{os.path.basename(image_file.filename)}"
            chemin_image_temporaire = os.path.join(UPLOAD_DIR, nom_unique)
            with open(chemin_image_temporaire, "wb") as buffer:
                shutil.copyfileobj(image_file.file, buffer)
        else:
            # L'utilisateur a collé une URL plutôt que téléversé un fichier.
            chemin_image_temporaire = telecharger_image_depuis_url(image_url)

        url_image_cloudinary = uploader_image(chemin_image_temporaire)
        os.remove(chemin_image_temporaire)
        chemin_image_temporaire = ""
    except ValueError as e:
        # Erreur imputable à l'utilisateur (URL invalide, pas une image, trop
        # lourde...) — distincte d'une erreur serveur.
        if chemin_image_temporaire and os.path.exists(chemin_image_temporaire):
            os.remove(chemin_image_temporaire)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        if chemin_image_temporaire and os.path.exists(chemin_image_temporaire):
            os.remove(chemin_image_temporaire)
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement de l'image : {str(e)}")

    etat_initial = {
        "rumor_text": rumor_text,
        "rumor_image": url_image_cloudinary,
        "current_date": actuel_date,
        "justification_json": {},
        "performance_metrics": {}
    }

    return StreamingResponse(
        executer_graphe_stream(
            etat_initial,
            rumor_text,
            url_image_cloudinary,
            user_id=user["id"] if user else None,
        ),
        media_type="application/x-ndjson",
    )