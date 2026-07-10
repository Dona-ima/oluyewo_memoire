import os
import requests
from typing import List

def effectuer_recherche_web(requete: str) -> dict:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {"texte": f"[Erreur] Clé API Tavily manquante.", "sources": []}
        
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": requete,
        "search_depth": "advanced",
        "max_results": 3
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        resultats_texte = []
        sources = []
        for res in data.get("results", []):
            # Tavily retourne un score entre 0 et 1
            score = res.get("score", 0)
            
            # Ne garde que les sources avec un score de pertinence >= 0.7
            if score >= 0.7:
                resultats_texte.append(
                    f"Source: {res.get('url')}\nTitre: {res.get('title')}\nExtrait: {res.get('content')}\n"
                )
                sources.append({
                    "url": res.get("url"),
                    "titre": res.get("title"),
                    "score": score
                })
            
        return {
            "texte": "\n---\n".join(resultats_texte) if resultats_texte else "Aucun résultat trouvé.",
            "sources": sources
        }
        
    except Exception as e:
        return {"texte": f"[Erreur de recherche] : {str(e)}", "sources": []}