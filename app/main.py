import os
import shutil
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
from datetime import datetime

from app.database import (
    initialiser_bdd, sauvegarder_verification,
    creer_utilisateur, obtenir_utilisateur_par_email,
    obtenir_historique_utilisateur, obtenir_detail_verification
)
from app.auth import (
    hasher_password, verifier_password, creer_token,
    obtenir_utilisateur_courant
)
from app.services.storage_service import uploader_image
from app.graph.build import compiler_graphe
from jose import JWTError, jwt

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

initialiser_bdd()
graphe_oluyewo = compiler_graphe()

# --- Modèles Pydantic ---
class InscriptionSchema(BaseModel):
    nom: str
    prenom: str
    email: EmailStr
    password: str

class ConnexionSchema(BaseModel):
    email: EmailStr
    password: str

# --- Utilitaire : user optionnel depuis token ---
def obtenir_user_id_optionnel(request: Request):
    SECRET_KEY = os.getenv("SECRET_KEY", "changeme")
    ALGORITHM = "HS256"
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return int(payload.get("sub"))
        except:
            return None
    return None

# --- Routes santé ---
@app.get("/")
def read_root():
    return {"status": "online", "application": "Oluyewo Fact-Checking Engine"}

# --- Routes auth ---
@app.post("/auth/register")
def inscription(data: InscriptionSchema):
    if obtenir_utilisateur_par_email(data.email):
        raise HTTPException(status_code=400, detail="Cet email est déjà utilisé.")
    hashed = hasher_password(data.password)
    user_id = creer_utilisateur(data.nom, data.prenom, data.email, hashed)
    return {"message": "Compte créé avec succès.", "user_id": user_id}

@app.post("/auth/login")
def connexion(data: ConnexionSchema):
    user = obtenir_utilisateur_par_email(data.email)
    if not user or not verifier_password(data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect.")
    token = creer_token({"sub": str(user["id"])})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/auth/logout")
def deconnexion(current_user: dict = Depends(obtenir_utilisateur_courant)):
    return {"message": "Déconnexion effectuée avec succès."}

# --- Routes protégées ---
@app.get("/me")
def mon_profil(current_user: dict = Depends(obtenir_utilisateur_courant)):
    return current_user

@app.get("/history")
def historique(current_user: dict = Depends(obtenir_utilisateur_courant)):
    return obtenir_historique_utilisateur(current_user["id"])

@app.get("/history/{verification_id}")
def detail_verification(verification_id: int, current_user: dict = Depends(obtenir_utilisateur_courant)):
    detail = obtenir_detail_verification(verification_id, current_user["id"])
    if not detail:
        raise HTTPException(status_code=404, detail="Analyse introuvable.")
    return detail

# --- Route principale ---
@app.post("/verify-rumor")
async def verify_rumor(
    request: Request,
    rumor_text: str = Form(...),
    image_file: UploadFile = File(None)
):
    if not rumor_text.strip():
        raise HTTPException(status_code=400, detail="Le texte de la rumeur ne peut pas être vide.")

    user_id = obtenir_user_id_optionnel(request)
    chemin_image_temporaire = ""
    url_image_cloudinary = ""
    actuel_date = datetime.now().strftime("%d %B %Y")

    try:
        
        if image_file:
            nom_unique = f"photo_{time.time()}_{image_file.filename}"
            chemin_image_temporaire = os.path.join(UPLOAD_DIR, nom_unique)
            with open(chemin_image_temporaire, "wb") as buffer:
                shutil.copyfileobj(image_file.file, buffer)
            
            
            url_image_cloudinary = uploader_image(chemin_image_temporaire)
            os.remove(chemin_image_temporaire)  
        else:
            url_image_cloudinary = ""


        etat_initial = {
            "rumor_text": rumor_text,
            "rumor_image": url_image_cloudinary,
            "current_date": actuel_date,
            "image_description": "",
            "lens_results": "",
            "sub_questions": [],
            "search_history": [],
            "evidence_gathered": [],
            "confidence_score": 0.0,
            "iterations": 0,
            "verdict": "",
            "justification_json": {},
            "performance_metrics": {
                "total_latency": 0.0,
                "total_api_calls": 0,
                "path_taken": []
            }
        }

        config = {"configurable": {"thread_id": f"session_{time.time()}"}}
        resultat_graphe = await graphe_oluyewo.ainvoke(etat_initial, config=config)

        reponse_finale = resultat_graphe.get("justification_json", {
            "statut": "Unvérifiable",
            "score_certitude": 0.0,
            "analyse_narrative": "Le système n'a pas pu générer de rapport.",
            "metadonnees": {}
        })

        if chemin_image_temporaire and os.path.exists(chemin_image_temporaire):
            url_image_cloudinary = uploader_image(chemin_image_temporaire)
            os.remove(chemin_image_temporaire)

        sauvegarder_verification(
            rumor_text,
            url_image_cloudinary,
            resultat_graphe,
            reponse_finale,
            user_id
        )

        return reponse_finale

    except Exception as e:
        import traceback
        traceback.print_exc()
        if chemin_image_temporaire and os.path.exists(chemin_image_temporaire):
            os.remove(chemin_image_temporaire)
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")