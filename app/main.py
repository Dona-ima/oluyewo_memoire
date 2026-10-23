import os
import shutil
import json
import time
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from datetime import datetime

from app.services.storage_service import uploader_image
from app.graph.build import compiler_graphe
# Constante ré-utilisée uniquement pour documenter la méthode d'agrégation
# réellement appliquée au score sémantique dans la réponse API (voir plus bas).
from app.graph.node_1.n1_smOOC import METHODE_AGREGATION_DEFAUT

load_dotenv()

app = FastAPI(
    title="Oluyewo Backend API",
    description="Moteur de fact-checking multimodal basé sur une architecture multi-agents (LangGraph & Gemini)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


async def executer_graphe_stream(etat_initial: dict, rumor_text: str, url_image: str):
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
        yield json.dumps({"type": "resultat", "donnees": reponse_finale}) + "\n"

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield json.dumps({
            "type": "erreur",
            "detail": f"Erreur interne : {str(e)}",
        }) + "\n"


# --- Routes santé ---
@app.get("/")
def read_root():
    return {"status": "online", "application": "Oluyewo Fact-Checking Engine"}


# --- Route principale (streaming) ---
@app.post("/verify-rumor/stream")
async def verify_rumor_stream(
    rumor_text: str = Form(...),
    image_file: UploadFile = File(None),
    image_url: str = Form(None),
):
    if not rumor_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Le texte de la rumeur ne peut pas être vide."
        )

    image_url_fournie = bool(image_url and image_url.strip())
    if not image_file and not image_url_fournie:
        raise HTTPException(
            status_code=400,
            detail="Une image est requise (fichier ou URL)."
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
        if image_file:
            nom_unique = f"photo_{time.time()}_{image_file.filename}"
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
        executer_graphe_stream(etat_initial, rumor_text, url_image_cloudinary),
        media_type="application/x-ndjson",
    )