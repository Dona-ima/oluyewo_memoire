from app.graph.state import AgentState

def routeur_fire(state: AgentState) -> str:
    """
    Fonction de routage conditionnel (Le Carrefour FIRE).
    Évalue l'état actuel pour décider de la trajectoire à suivre.
    
    Returns:
        str: Le nom du prochain nœud à activer ('vers_synthese' ou 'vers_recherche')
    """
    score = state.get("confidence_score", 0.0)
    boucles = state.get("iterations", 0)
    
    print(f"\n--- [ROUTEUR FIRE] Évaluation du flux ---")
    print(f"Score de confiance actuel : {score:.2f}")
    print(f"Nombre d'itérations de recherche effectuées : {boucles}/3")
    
    # Règle formelle : Si la certitude est suffisante (>= 0.85)
    # OU si le garde-fou des 3 itérations maximum est atteint.
    if score >= 0.85 or boucles >= 3:
        if boucles >= 3:
            print("-> Garde-fou activé (3 itérations atteintes). Direction la synthèse.")
        else:
            print("-> Confiance suffisante atteinte (>= 0.85). Direction la synthèse.")
        return "vers_synthese"
        
    # Sinon, on continue l'investigation sur le web
    print("-> Confiance insuffisante et limite non atteinte. Lancement/Poursuite des recherches.")
    return "vers_recherche"