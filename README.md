<div align="center">

# 🔍 Oluyewo
### Système Multi-Agent Multimodal pour le Fact-Checking Automatisé

*Oluyewo (ọlùṣàyẹ̀wò) — mot Yoruba signifiant « Vérificateur »*

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-blue?style=flat)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/Google-Gemini%202.5%20Flash-4285F4?style=flat&logo=google)](https://deepmind.google/technologies/gemini)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Render-336791?style=flat&logo=postgresql)](https://render.com)
[![Cloudinary](https://img.shields.io/badge/Cloudinary-Image%20Storage-3448C5?style=flat)](https://cloudinary.com)

**Projet de fin d'études — Licence 3 Intelligence Artificielle**  
Institut de Formation et de Recherche en Informatique (IFRI) — Université d'Abomey-Calavi  
Étudiante : **Ariane Aliénore Dona AGBOTON**  
Encadreurs : **Ing. Mélène TONOU** & **Dr Ing. (MC) Ratheil HOUNDJI**  
Année académique : 2025-2026

[Démo API](https://oluyewo.onrender.com/docs) · [Signaler un bug](https://github.com/ton-username/oluyewo-backend/issues)

</div>

---

## 📌 À propos

La désinformation numérique se propage en quelques heures alors que la vérification humaine prend plusieurs jours. En Afrique de l'Ouest, ce problème est aggravé par la rareté des cellules de fact-checking professionnalisées et la complexité des contenus multilingues.

**Oluyewo** est un système de fact-checking automatisé, **multi-agent** et **multimodal**, capable de vérifier conjointement le texte et l'image d'une rumeur. Il détecte notamment les **détournements de contexte** — cas où une image authentique est décontextualisée pour tromper le public — un phénomène particulièrement difficile à détecter pour l'utilisateur ordinaire.

### Ce qu'Oluyewo peut faire

- ✅ Vérifier une rumeur textuelle pure
- ✅ Analyser une image accompagnant une rumeur (recherche inversée Google Lens)
- ✅ Détecter les images détournées de leur contexte original
- ✅ Fournir un verdict structuré avec score de confiance et justification narrative
- ✅ Effectuer des recherches web itératives pour collecter des preuves

### Verdicts possibles

| Verdict | Signification |
|---|---|
| `Vrai` | L'information est confirmée par les preuves collectées |
| `Faux` | L'information est contredite par les preuves collectées |
| `Contexte détourné` | L'image vient d'ailleurs ou le contexte est manipulé |
| `Unvérifiable` | Les preuves disponibles sont insuffisantes pour trancher |

---

## 🏗️ Architecture

Oluyewo repose sur une architecture **multi-agent orchestrée par LangGraph**, composée de trois agents spécialisés et d'un routeur conditionnel appelé **Routeur FIRE** (inspiré du framework FIRE — Fact-checking via Iterative Retrieval and Evaluation).

```
Rumeur (texte + image)
        │
        ▼
┌─────────────────────────────┐
│  Agent Vérificateur         │  ← Analyse image (Google Lens) + texte
│  Principal                  │  ← Calcule score de confiance
└─────────────┬───────────────┘
              │
              ▼
        [Routeur FIRE]
       score >= 0.85 ?
      /               \
    OUI               NON
     │                 │
     ▼                 ▼
┌─────────┐   ┌─────────────────┐
│Synthéti-│   │ Agent Chercheur │  ← Recherche web itérative
│seur     │   │ Web             │  ← Max 3 itérations
└─────────┘   └────────┬────────┘
     │                 │
     │         retour au vérificateur
     │
     ▼
Verdict JSON structuré
```

### Schéma du graphe LangGraph

![Graphe Oluyewo](docs/graph_oluyewo.png)

---

## 🛠️ Stack technique

| Composant | Technologie |
|---|---|
| Backend API | FastAPI |
| Orchestration multi-agent | LangGraph |
| Modèle de langage | Google Gemini 2.5 Flash |
| Recherche inversée d'image | SerpApi Google Lens |
| Recherche web | SerpApi |
| Base de données | PostgreSQL (Render) |
| Stockage images | Cloudinary |
| Authentification | JWT (python-jose + passlib) |
| Déploiement | Render |

---

## 📸 Aperçu

### Interface API (Swagger UI)

![Swagger UI](docs/swagger_ui.png)

### Exemple — Verdict "Contexte détourné"

Rumeur soumise : *"Des affrontements entre le niger et le bénin ont éclaté à la frontière des 2 pays. On compte actuellement 3 morts et 20 blessés"* + image de militaire.

![Verdict Contexte détourné](docs/militaire.jpg)

![Verdict Contexte détourné](docs/verdict_contexte_detourne.png)

> Oluyewo a identifié que l'image montrant des soldats et un véhicule militaire, est en réalité liée au coup d'État survenu en Guinée en 2021 , concluant à un détournement de contexte avec un score de certitude de 0.95.

### Exemple — Verdict "Faux"

Rumeur soumise : *"La tentative de coup d'état du 7 Décembre 2025 a été un franc succès. Le Président Patrice TALON a été déchu."*

![Verdict Faux](docs/verdict_faux.png)

> Oluyewo a détecté qu'il est vrai qu'une tentative de coup d'État a bien eu lieu le 7 décembre 2025 au Bénin, cependant, cette tentative a été rapidement déjouée. Patrice Talon lui-même est apparu à la télévision nationale le soir même pour confirmer l'échec du putsch et rassurer la population. Score de certitude : 0.95.

---

## 🚀 Installation & Lancement

### Prérequis

- Python 3.11+
- Un compte [Google AI Studio](https://aistudio.google.com) (clé API Gemini)
- Un compte [Tavily](https://www.tavily.com/) (clé API)
- Un compte [SerpApi](https://serpapi.com) (clé API)
- Un compte [Cloudinary](https://cloudinary.com)
- Une base de données PostgreSQL (ex: Render free tier)

### 1. Cloner le dépôt

```bash
git clone https://github.com/ton-username/oluyewo-backend.git
cd oluyewo-backend
```

### 2. Créer l'environnement virtuel

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement

Crée un fichier `.env` à la racine :

```env
# Google Gemini
GEMINI_API_KEY=ta_cle_gemini

# Tavily
TAVILY_API_KEY=ta_cle_tavily

# SerpApi
SERPAPI_API_KEY=ta_cle_serpapi

# Base de données PostgreSQL
DATABASE_URL=postgresql://user:password@host/dbname

# Cloudinary
CLOUDINARY_CLOUD_NAME=ton_cloud_name
CLOUDINARY_API_KEY=ta_cle_api
CLOUDINARY_API_SECRET=ton_secret

# JWT
SECRET_KEY=ta_cle_secrete_jwt
```

### 5. Lancer le serveur

```bash
uvicorn app.main:app --reload
```

L'API est disponible sur `http://127.0.0.1:8000`  
La documentation interactive Swagger est sur `http://127.0.0.1:8000/docs`

---

## 📡 Routes API

### Authentification

| Méthode | Route | Description |
|---|---|---|
| `POST` | `/auth/register` | Créer un compte |
| `POST` | `/auth/login` | Se connecter (retourne JWT) |
| `POST` | `/auth/logout` | Se déconnecter |

### Fact-checking

| Méthode | Route | Description | Auth requise |
|---|---|---|---|
| `POST` | `/verify-rumor` | Analyser une rumeur (texte + image optionnelle) | Non |

### Utilisateur

| Méthode | Route | Description | Auth requise |
|---|---|---|---|
| `GET` | `/me` | Profil de l'utilisateur connecté | Oui |
| `GET` | `/history` | Historique des analyses | Oui |
| `GET` | `/history/{id}` | Détails d'une analyse | Oui |

### Exemple de requête

```bash
curl -X POST "https://oluyewo.onrender.com/verify-rumor" \
  -F "rumor_text=L'échangeur de Godomey s'est effondré ce matin" \
  -F "image_file=@image.jpg"
```

### Exemple de réponse

```json
{
  "statut": "Contexte détourné",
  "score_certitude": 0.95,
  "analyse_narrative": "L'image qui accompagne cette rumeur provient en réalité du Carrefour Solibra à Abidjan, en Côte d'Ivoire, et non de l'échangeur de Godomey au Bénin...",
  "metadonnees": {
    "recherches_effectuees": [],
    "image_analyse": true,
    "cycles_recherche": 0
  },
  "performance": {
    "latence_totale_secondes": 8.34,
    "total_appels_api": 2,
    "chemin_execution": ["agent_verificateur_principal"]
  }
}
```

---

## 🧪 Cas de test validés

| # | Rumeur | Image | Verdict attendu | Verdict obtenu | Score |
|---|---|---|---|---|---|
| 1 | Effondrement échangeur Godomey | Pont Carrefour Solibra (Abidjan) | Contexte détourné | ✅ Contexte détourné | 0.95 |
| 2 | Séisme magnitude 6.2 à Porto-Novo, 47 morts | — | Faux | ✅ Faux | 0.98 |
| 3 | Démission du président Patrice Talon | — | Faux | ✅ Faux | 0.95 |
| 4 | Bénin rejoint l'Alliance des États du Sahel | — | Faux | ✅ Faux | 0.95 |
| 5 | Inondations dévastatrices à Cotonou | Photo inondation Ghana | Contexte détourné | ✅ Contexte détourné | 0.95 |
| 6 | Baleine échouée sur les côtes de Cotonou | Photo rorqual Belgique 2018 | Contexte détourné | ✅ Contexte détourné | — |

---

## 📁 Structure du projet

```
oluyewo-backend/
├── app/
│   ├── graph/
│   │   ├── build.py        # Compilation du graphe LangGraph
│   │   ├── edges.py        # Routeur FIRE (logique conditionnelle)
│   │   ├── nodes.py        # Les 3 agents + prompts
│   │   └── state.py        # Structure de l'état partagé
│   ├── services/
│   │   └── search_service.py  # Service de recherche web
│   ├── auth.py             # JWT authentication
│   ├── database.py         # ORM SQLAlchemy + PostgreSQL
│   ├── main.py             # Routes FastAPI
│   ├── storage_service.py  # Upload Cloudinary
│   └── utils.py            # Utilitaires
├── docs/
│   ├── graph_oluyewo.png
│   ├── swagger_ui.png
│   ├── verdict_contexte_detourne.png
│   └── verdict_faux.png
├── uploads/                # Dossier temporaire (ignoré par git)
├── .env                    # Variables d'environnement (ignoré par git)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚠️ Limites connues

- **Dépendance aux APIs externes** : SerpApi et Gemini sont des services tiers soumis à des quotas et des pannes
- **Google Lens** : n'identifie pas toujours l'origine exacte d'une image peu indexée sur le web
- **Latence variable** : entre 7 secondes (cas simple) et 160 secondes (3 cycles de recherche)
- **Langues** : optimisé pour le français, support partiel de l'anglais

---

## 🔭 Perspectives

- Intégration d'un frontend responsive
- Implémentation du module d'audit dynamique FACT-AUDIT (métriques JFR et IMR)
- Benchmarks sur le dataset AVerImaTeC
- Support des langues locales (fon, yoruba)
- Détection de deepfakes

---

## 📚 Références

- FIRE : Xie, Z. et al. (2025). *FIRE: Fact-checking via Iterative Retrieval and Evaluation*
- FACT-AUDIT : Lin, H. et al. *FACT-AUDIT: An Adaptive Multi-Agent Framework for Dynamic Fact-Checking Evaluation of LLMs*
- LoCal : Ma, Y. et al. (2025). *LoCal: Logical and Causal Reasoning for Automated Fact-Checking*. ACM Web Conference 2025
- AVerImaTeC : Rui4416. (2024). [Dataset Hugging Face](https://huggingface.co/datasets/Rui4416/AVerImaTeC)
- Africa Check : [africacheck.org](https://africacheck.org)
- Reuters Institute. (2024). *Digital News Report 2024*

---

<div align="center">

IFRI — Université d'Abomey-Calavi — 2025-2026

</div>
