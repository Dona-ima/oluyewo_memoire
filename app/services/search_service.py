import os
import requests
from typing import List

def effectuer_recherche_web(requete: str) -> str:
    """
    Interroge l'API Tavily (ou Serper) pour obtenir des extraits de textes factuels.
    Nécessite la clé API TAVILY_API_KEY dans votre fichier .env.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        # En mode secours si la clé est manquante pour éviter un crash complet
        return f"[Erreur] Clé API Tavily manquante. Impossible de chercher : '{requete}'"
        
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": requete,
        "search_depth": "advanced", # Recherche poussée pour le fact-checking
        "max_results": 3            # On prend les 3 meilleurs résultats
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # On extrait et formatte les résultats pertinents
        resultats = []
        for res in data.get("results", []):
            resultats.append(f"Source: {res.get('url')}\nTitre: {res.get('title')}\nExtrait: {res.get('content')}\n")
            
        return "\n---\n".join(resultats) if resultats else "Aucun résultat trouvé sur le web."
        
    except Exception as e:
        return f"[Erreur de recherche] : {str(e)}"