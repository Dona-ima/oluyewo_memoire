# Oluyewo Backend

API de fact-checking multimodal : à partir du texte d'une rumeur et d'une image, Oluyewo analyse leur cohérence et rédige un rapport de vérification.

## Fonctionnement

Le graphe LangGraph exécute en parallèle deux analyses :

1. Recherche de l'origine de l'image avec Google Lens via SerpApi, puis comparaison des résultats avec le texte de la rumeur.
2. Comparaison visuelle entre l'image et le texte avec le Space PaliGemma SNLI-VE hébergé sur Hugging Face.

Un dernier nœud Gemini rassemble les résultats et produit le rapport. L'API transmet les étapes et le rapport au fil de leur exécution au format NDJSON.

## Prérequis

- Python 3.11 ou ultérieur
- Une clé API Gemini
- Une clé API SerpApi
- Un jeton Hugging Face pour accéder au Space PaliGemma
- Un compte Cloudinary et ses identifiants pour l'hébergement des images

## Installation

Clone le dépôt et entre dans son dossier :

```bash
git clone https://github.com/Dona-ima/oluyewo_memoire.git
cd oluyewo_memoire
```

Crée et active un environnement virtuel, puis installe les dépendances :

```bash
python -m venv .venv
```

Sous Windows PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
```

Sous Linux ou macOS :

```bash
source .venv/bin/activate
```

Installe les paquets :

```bash
python -m pip install -r requirements.txt
```

## Configuration

Crée un fichier `.env` à la racine du dépôt :

```env
GEMINI_API_KEY_OOC=your_gemini_api_key
SERPAPI_API_KEY=your_serpapi_api_key
HF_TOKEN=your_huggingface_token

CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_cloudinary_api_key
CLOUDINARY_API_SECRET=your_cloudinary_api_secret
```

`GEMINI_TRANSLATION_MODEL` est facultative ; le modèle par défaut est `gemini-3.5-flash-lite`.

## Lancement

Depuis la racine du dépôt :

```bash
uvicorn app.main:app --reload
```

- API : <http://127.0.0.1:8000>
- Documentation interactive : <http://127.0.0.1:8000/docs>
- Vérification de disponibilité : `GET /`

## Vérifier une rumeur

L'endpoint `POST /verify-rumor/stream` reçoit des données de formulaire. Le texte est obligatoire et une image est requise, soit sous forme de fichier (`image_file`), soit sous forme d'URL (`image_url`).

Exemple avec un fichier image :

```bash
curl -N -X POST "http://127.0.0.1:8000/verify-rumor/stream" \
  -F "rumor_text=Une rumeur à vérifier" \
  -F "image_file=@image.jpg"
```

Exemple avec une URL d'image :

```bash
curl -N -X POST "http://127.0.0.1:8000/verify-rumor/stream" \
  -F "rumor_text=Une rumeur à vérifier" \
  -F "image_url=https://example.com/image.jpg"
```

La réponse est un flux NDJSON, avec un objet JSON par ligne. Les événements comprennent le démarrage (`start`), les étapes terminées (`etape_terminee`), puis le rapport final (`resultat`, dans `donnees`). Si une erreur survient après le démarrage du flux, un événement `erreur` est envoyé.

Le rapport final comprend notamment le texte et l'image soumis, la classe OOC, les données techniques des analyses et les métriques de performance. Le détail du rapport de vérification est généré par Gemini.

## Structure principale

```text
app/
├── graph/
│   ├── node_1/       # Recherche d'origine et comparaison visuelle
│   ├── node_2/       # Synthèse Gemini
│   ├── build.py      # Construction du graphe LangGraph
│   ├── nodes.py      # Nœuds du graphe
│   └── state.py      # État partagé du graphe
├── services/
│   ├── search_service.py
│   └── storage_service.py
└── main.py           # Application FastAPI et endpoint de streaming
```

## Dépendances externes

Le traitement repose sur Gemini, SerpApi, Hugging Face et Cloudinary. La disponibilité, les quotas et les délais de réponse de ces services peuvent affecter l'analyse.
