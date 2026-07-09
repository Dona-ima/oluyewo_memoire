import os
import json
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,        
    pool_recycle=300,          
    pool_size=5,
    max_overflow=10
)

def initialiser_bdd():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                nom TEXT NOT NULL,
                prenom TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                hashed_password TEXT NOT NULL,
                date_creation TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS verifications (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                rumor_text TEXT NOT NULL,
                image_path TEXT,
                verdict TEXT,
                score_certitude REAL,
                analyse_narrative TEXT,
                image_description TEXT,
                evidence_gathered TEXT,
                recherches_effectuees TEXT,
                date_creation TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.commit()
    print("[BDD] Tables initialisées avec succès.")

def creer_utilisateur(nom: str, prenom: str, email: str, hashed_password: str):
    with engine.connect() as conn:
        result = conn.execute(text("""
            INSERT INTO users (nom, prenom, email, hashed_password)
            VALUES (:nom, :prenom, :email, :hashed_password)
            RETURNING id
        """), {"nom": nom, "prenom": prenom, "email": email, "hashed_password": hashed_password})
        conn.commit()
        return result.fetchone()[0]

def obtenir_utilisateur_par_email(email: str):
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT id, nom, prenom, email, hashed_password, date_creation FROM users WHERE email = :email"
        ), {"email": email})
        row = result.fetchone()
        if row:
            return {"id": row[0], "nom": row[1], "prenom": row[2], "email": row[3], "hashed_password": row[4], "date_creation": str(row[5])}
        return None

def obtenir_utilisateur_par_id(user_id: int):
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT id, nom, prenom, email, date_creation FROM users WHERE id = :id"
        ), {"id": user_id})
        row = result.fetchone()
        if row:
            return {"id": row[0], "nom": row[1], "prenom": row[2], "email": row[3], "date_creation": str(row[4])}
        return None

def sauvegarder_verification(rumor_text: str, image_path: str, resultat_graphe: dict, reponse_finale_json: dict, user_id: int = None):
    recherches = json.dumps(resultat_graphe.get("search_history", []))
    preuves = json.dumps(resultat_graphe.get("evidence_gathered", []))
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO verifications 
            (user_id, rumor_text, image_path, verdict, score_certitude, analyse_narrative, image_description, evidence_gathered, recherches_effectuees)
            VALUES (:user_id, :rumor_text, :image_path, :verdict, :score_certitude, :analyse_narrative, :image_description, :evidence_gathered, :recherches_effectuees)
        """), {
            "user_id": user_id,
            "rumor_text": rumor_text,
            "image_path": image_path,
            "verdict": reponse_finale_json.get("statut"),
            "score_certitude": reponse_finale_json.get("score_certitude"),
            "analyse_narrative": reponse_finale_json.get("analyse_narrative"),
            "image_description": resultat_graphe.get("image_description", ""),
            "evidence_gathered": preuves,
            "recherches_effectuees": recherches
        })
        conn.commit()
    print("[BDD] Vérification sauvegardée avec succès.")

def obtenir_historique_utilisateur(user_id: int):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT id, rumor_text, verdict, score_certitude, date_creation
            FROM verifications WHERE user_id = :user_id
            ORDER BY date_creation DESC
        """), {"user_id": user_id})
        rows = result.fetchall()
        return [{"id": r[0], "rumor_text": r[1], "verdict": r[2], "score_certitude": r[3], "date_creation": str(r[4])} for r in rows]

def obtenir_detail_verification(verification_id: int, user_id: int):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT id, rumor_text, image_path, verdict, score_certitude, analyse_narrative,
                   image_description, evidence_gathered, recherches_effectuees, date_creation
            FROM verifications WHERE id = :id AND user_id = :user_id
        """), {"id": verification_id, "user_id": user_id})
        row = result.fetchone()
        if row:
            return {
                "id": row[0], "rumor_text": row[1], "image_path": row[2],
                "verdict": row[3], "score_certitude": row[4], "analyse_narrative": row[5],
                "image_description": row[6],
                "evidence_gathered": json.loads(row[7] or "[]"),
                "recherches_effectuees": json.loads(row[8] or "[]"),
                "date_creation": str(row[9])
            }
        return None