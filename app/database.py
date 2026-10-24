import os
import json
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

engine = None


def obtenir_engine():
    global engine
    if engine is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError("La variable d'environnement DATABASE_URL est absente.")
        if database_url.startswith("postgres://"):
            database_url = "postgresql://" + database_url[len("postgres://"):]
        engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=5,
            max_overflow=10
        )
    return engine

def initialiser_bdd():
    with obtenir_engine().connect() as conn:
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
        conn.execute(text("""
            ALTER TABLE verifications
            ADD COLUMN IF NOT EXISTS resultat_complet TEXT
        """))
        conn.commit()
    print("[BDD] Tables initialisées avec succès.")

def creer_utilisateur(nom: str, prenom: str, email: str, hashed_password: str):
    with obtenir_engine().connect() as conn:
        result = conn.execute(text("""
            INSERT INTO users (nom, prenom, email, hashed_password)
            VALUES (:nom, :prenom, :email, :hashed_password)
            RETURNING id
        """), {"nom": nom, "prenom": prenom, "email": email, "hashed_password": hashed_password})
        user_id = result.fetchone()[0]
        conn.commit()
        return user_id

def obtenir_utilisateur_par_email(email: str):
    with obtenir_engine().connect() as conn:
        result = conn.execute(text(
            "SELECT id, nom, prenom, email, hashed_password, date_creation FROM users WHERE LOWER(email) = :email"
        ), {"email": email})
        row = result.fetchone()
        if row:
            return {"id": row[0], "nom": row[1], "prenom": row[2], "email": row[3], "hashed_password": row[4], "date_creation": str(row[5])}
        return None

def obtenir_utilisateur_par_id(user_id: int):
    with obtenir_engine().connect() as conn:
        result = conn.execute(text(
            "SELECT id, nom, prenom, email, date_creation FROM users WHERE id = :id"
        ), {"id": user_id})
        row = result.fetchone()
        if row:
            return {"id": row[0], "nom": row[1], "prenom": row[2], "email": row[3], "date_creation": str(row[4])}
        return None

def sauvegarder_verification(
    rumor_text: str,
    image_path: str,
    resultat_graphe: dict,
    reponse_finale_json: dict,
    user_id: int
) -> int:
    resultat_utilisateur = reponse_finale_json.get("resultat_utilisateur", {})
    if not isinstance(resultat_utilisateur, dict):
        resultat_utilisateur = {}
    analyse_contexte = resultat_utilisateur.get("analyse_contexte")
    if not isinstance(analyse_contexte, dict):
        analyse_contexte = {}
    donnees_techniques = reponse_finale_json.get("donnees_techniques", {})
    if not isinstance(donnees_techniques, dict):
        donnees_techniques = {}
    analyse_visuelle = resultat_utilisateur.get("analyse_visuelle_image_rumeur")
    if not isinstance(analyse_visuelle, dict):
        analyse_visuelle = {}
    preuves = analyse_contexte.get("preuves", [])
    recherches = resultat_graphe.get("resultats", [])

    with obtenir_engine().connect() as conn:
        result = conn.execute(text("""
            INSERT INTO verifications 
            (user_id, rumor_text, image_path, verdict, score_certitude, analyse_narrative,
             image_description, evidence_gathered, recherches_effectuees, resultat_complet)
            VALUES (:user_id, :rumor_text, :image_path, :verdict, :score_certitude, :analyse_narrative, :image_description, :evidence_gathered, :recherches_effectuees, :resultat_complet)
            RETURNING id
        """), {
            "user_id": user_id,
            "rumor_text": rumor_text,
            "image_path": image_path,
            "verdict": resultat_utilisateur.get("verdict"),
            "score_certitude": donnees_techniques.get("score_certitude"),
            "analyse_narrative": resultat_utilisateur.get("conclusion") or resultat_utilisateur.get("resume", ""),
            "image_description": analyse_visuelle.get("explication", ""),
            "evidence_gathered": json.dumps(preuves, ensure_ascii=False),
            "recherches_effectuees": json.dumps(recherches, ensure_ascii=False),
            "resultat_complet": json.dumps(reponse_finale_json, ensure_ascii=False)
        })
        verification_id = result.fetchone()[0]
        conn.commit()
        return verification_id

def obtenir_historique_utilisateur(user_id: int):
    with obtenir_engine().connect() as conn:
        result = conn.execute(text("""
            SELECT id, rumor_text, image_path, verdict, score_certitude, date_creation
            FROM verifications WHERE user_id = :user_id
            ORDER BY date_creation DESC
        """), {"user_id": user_id})
        rows = result.fetchall()
        return [
            {
                "id": row[0],
                "rumor_text": row[1],
                "image_path": row[2],
                "verdict": row[3],
                "score_certitude": row[4],
                "date_creation": str(row[5])
            }
            for row in rows
        ]

def obtenir_detail_verification(verification_id: int, user_id: int):
    with obtenir_engine().connect() as conn:
        result = conn.execute(text("""
            SELECT id, rumor_text, image_path, verdict, score_certitude, analyse_narrative,
                   image_description, evidence_gathered, recherches_effectuees, date_creation,
                   resultat_complet
            FROM verifications WHERE id = :id AND user_id = :user_id
        """), {"id": verification_id, "user_id": user_id})
        row = result.fetchone()
        if row:
            detail = {
                "id": row[0], "rumor_text": row[1], "image_path": row[2],
                "verdict": row[3], "score_certitude": row[4], "analyse_narrative": row[5],
                "image_description": row[6],
                "evidence_gathered": json.loads(row[7] or "[]"),
                "recherches_effectuees": json.loads(row[8] or "[]"),
                "date_creation": str(row[9])
            }
            detail["resultat"] = json.loads(row[10]) if row[10] else None
            return detail
        return None