import os 
import time
import base64
import requests  
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from app.graph.state import AgentState
from app.utils import encoder_image_en_base64
from app.services.search_service import effectuer_recherche_web


class VerificationAnalysis(BaseModel):
    image_description: str = Field(description="Description sémantique détaillée de l'image, origine détectée via recherche inversée, et OCR du texte.")
    critique_causale: str = Field(description="Analyse des biais logiques, corrélations douteuses ou faux liens de cause à effet.")
    confidence_score: float = Field(
        description="Score de certitude entre 0.0 et 1.0. "
                    "OBLIGATOIREMENT < 0.85 pour tout événement récent (moins de 2 ans) "
                    "car ta base de connaissance peut être obsolète. "
                    "Ne jamais mettre un score >= 0.85 sans preuves web collectées pour "
                    "les événements politiques, coups d'état, élections ou changements "
                    "de gouvernement récents."
    )
    sub_questions: List[str] = Field(description="Liste de 2 à 4 sous-questions précises à chercher sur le web si le score est bas. Vide si le score est >= 0.85.")

class FinalVerdictStructure(BaseModel):
    verdict: str = Field(description="Le statut final de la rumeur. Doit être STRICTEMENT l'un de ces mots : 'Vrai', 'Faux', 'Contexte détourné' ou 'Unvérifiable'.")
    explanation: str = Field(description="Une explication fluide, naturelle et humanisée en français. Rédige de manière narrative, sans puces complexes ni jargon artificiel. Sois direct et transparent.")
    score_final: float = Field(description="Le score de confiance final arrêté par le système (entre 0.0 et 1.0).")


VERIFICATEUR_PROMPT = """
Tu es l'analyste principal d'Oluyewo, un système expert en fact-checking multimodal.
Ton rôle est d'analyser la rumeur textuelle, l'image fournie et les résultats de la recherche inversée Google Lens.

Instructions cruciales :
1. Examine la rumeur textuelle et croise-la avec la description visuelle et les indices de recherche inversée fournis.
2. Si les données de recherche inversée ('indices_recherche_inversee') montrent que l'image correspond à un tout autre endroit (ex: Carrefour Solibra à Abidjan) que celui de la rumeur (ex: Échangeur de Godomey au Bénin), tu fais face à un 'Contexte détourné' ou une fausse information claire.
3. Évalue ton niveau de confiance (confidence_score) et le statut probable :
   - Image détournée + texte faux → 'Contexte détourné' (l'image vient d'ailleurs ET l'événement n'a pas eu lieu)
   - Image détournée + texte vrai → 'Contexte détourné' (l'image vient d'ailleurs MAIS l'événement a réellement eu lieu)
   - Aucune image détournée + texte faux → 'Faux'
   - Aucune image détournée + texte vrai → 'Vrai'
   - Preuves insuffisantes dans les deux cas → 'Unvérifiable'
   
   Le statut 'Contexte détourné' concerne UNIQUEMENT l'image, pas le texte.
   Le texte doit être vérifié indépendamment via les preuves web collectées.

RÈGLE ABSOLUE sur l'origine de l'image :
- Tu ne peux affirmer l'origine d'une image QUE si elle apparaît explicitement dans les 
  indices SerpApi (titre, source, lien, ou requête suggérée).
- Si les correspondances visuelles montrent des images "similaires" sans lien direct avec 
  la photo soumise, tu dois écrire : "L'origine exacte de l'image n'a pas pu être 
  déterminée par la recherche inversée."
- Il est INTERDIT de déduire ou supposer un pays/lieu d'origine non mentionné dans les indices.

RÈGLE ABSOLUE sur les événements récents :
- La date d'aujourd'hui est : {current_date}.
- Pour tout événement datant de moins de 2 ans par rapport à cette date, tu DOIS 
  générer des sous-questions et mettre un score < 0.85, même si tu crois connaître 
  la réponse.
- Ta base de connaissance peut être obsolète sur des événements récents.
- Ne jamais affirmer qu'un événement politique récent "n'a pas eu lieu" sans preuve 
  issue d'une recherche web.
- Les rumeurs impliquant des dirigeants, des coups d'état, des élections ou des 
  changements de gouvernement DOIVENT toujours déclencher une recherche web.

RÈGLE sur les fautes d'orthographe et les termes mal orthographiés :
- Si la rumeur contient un mot qui semble être une faute d'orthographe d'un terme connu
  (ex: "antavirus" → "hantavirus", "covide" → "Covid", "omicront" → "Omicron"),
  interprète-le comme le terme correct le plus probable et analyse la rumeur en conséquence.
- Mentionne explicitement dans ton analyse que tu as interprété le terme corrigé.
- Ne rejette jamais une rumeur uniquement parce qu'elle contient une faute d'orthographe.

Rumeur textuelle à analyser : {rumor_text}
Indices issus de la recherche inversée de l'image (Google Lens) : {indices_recherche_inversee}
Preuves internet additionnelles déjà collectées : {evidence_gathered}
"""

SYNTHETISEUR_PROMPT = """
Tu es l'expert décisionnel final d'Oluyewo. Ta mission est de clore l'enquête de fact-checking en rédigeant le rapport de synthèse.

Tu disposes des éléments suivants :
- Rumeur initiale : {rumor_text}
- Analyse de l'image et origine : {image_description}
- Historique des questions explorées : {search_history}
- Preuves brutes extraites du web : {evidence_gathered}
- Dernier score de confiance calculé : {confidence_score}

Instructions impératives pour la rédaction de l'explication (field 'explanation') :
1. Adopte un ton naturel, posé et profondément humain. Écris comme un journaliste fact-checker indépendant qui s'adresse au grand public.
2. Sois d'une précision chirurgicale sur l'origine des images et des faits. Ne suppose rien qui ne soit écrit dans les preuves.
3. La date d'aujourd'hui est : {current_date}. Écris ton rapport avec une précision rigoureuse sur le temps présent.
4. Si image_description indique que l'origine de l'image est inconnue ou non déterminée,
   écris UNIQUEMENT que l'image ne correspond pas visuellement au lieu de la rumeur.
   Il est INTERDIT de mentionner un pays ou lieu d'origine non confirmé par les indices SerpApi.

Instructions pour le verdict final :
- Si l'image est détournée, mentionne-le explicitement, puis statue séparément sur la véracité du texte.
- Le champ 'verdict' doit refléter le cas le plus grave : une image détournée pour une rumeur vraie reste 'Contexte détourné'.
- Si après 2+ cycles de recherche aucune preuve directe n'existe dans un sens ou dans l'autre,
  et que la rumeur porte sur un événement privé/secret par nature → 'Unvérifiable', score <= 0.6.
- 'Faux' nécessite une preuve positive que l'événement N'A PAS eu lieu, pas juste une absence de preuve.

Génère ta réponse en suivant scrupuleusement la structure demandée.
"""

def effectuer_recherche_inversee_lens(url_image: str) -> str:
    try:
        if not url_image:
            return "Aucune image fournie."

        print(f"[SerpApi] Envoi de l'URL Cloudinary : {url_image}")

        params = {
            "engine": "google_lens",
            "url": url_image,  
            "api_key": os.getenv("SERPAPI_API_KEY"),
        }

        response = requests.get("https://serpapi.com/search", params=params, timeout=30)
        response.raise_for_status()
        results = response.json()

        matches = results.get("visual_matches", [])
        related = results.get("related_content", [])
        indices = []

        if related:
            indices.append("=== REQUÊTES SUGGÉRÉES PAR GOOGLE LENS (indices clés) ===")
            for item in related:
                indices.append(f"- Requête suggérée: '{item.get('query')}' | Lien: {item.get('link')}")

        if matches:
            indices.append("=== CORRESPONDANCES VISUELLES ===")
            for match in matches[:5]:
                indices.append(
                    f"- Titre: {match.get('title')} | "
                    f"Source: {match.get('source')} | "
                    f"Lien: {match.get('link')}"
                )

        if indices:
            print("[SerpApi] Correspondances visuelles récupérées avec succès.")
            return "\n".join(indices)

        return "Google Lens n'a trouvé aucune correspondance publique."

    except requests.exceptions.HTTPError:
        print(f"[SerpApi] Erreur HTTP : {response.status_code} - {response.text}")
        return f"Erreur serveur SerpApi ({response.status_code})."
    except Exception as e:
        print(f"[SerpApi] Erreur critique : {e}")
        return f"Erreur technique : {str(e)}"
    


def agent_verificateur_principal(state: AgentState) -> dict:
    debut_noeud = time.perf_counter()
    api_calls_count = 0
    
    print("--- AGENT VÉRIFICATEUR PRINCIPAL (AVEC LENS) ---")
    
    chemin_image = state.get("rumor_image") 
    lens_results_text = state.get("lens_results", "")

    if not lens_results_text:
        if chemin_image and chemin_image.startswith("http"):
            lens_results_text = effectuer_recherche_inversee_lens(chemin_image)
            api_calls_count += 1
        else:
            lens_results_text = "Aucune image fournie."
    else:
        print("[SerpApi] Résultats Lens déjà en cache — appel ignoré.")

    
    prompt_injecte = VERIFICATEUR_PROMPT.format(
        rumor_text=state.get("rumor_text", ""),
        indices_recherche_inversee=lens_results_text,
        evidence_gathered="\n".join(state.get("evidence_gathered", [])) if state.get("evidence_gathered") else "Aucune pour l'instant.",
        current_date=state.get("current_date", "")
    )
    
    contenu_message = [{"type": "text", "text": prompt_injecte}]

    if chemin_image and chemin_image.startswith("http"):
        try:
            import requests as req
            img_response = req.get(chemin_image)
            image_base64 = base64.b64encode(img_response.content).decode("utf-8")
            contenu_message.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        except Exception as e:
            print(f"Erreur encodage image : {e}")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    structured_llm = llm.with_structured_output(VerificationAnalysis)
    response = structured_llm.invoke([("user", contenu_message)])
    api_calls_count += 1
    
    fin_noeud = time.perf_counter()
    
    performance_log = state.get("performance_metrics", {"total_latency": 0.0, "total_api_calls": 0, "path_taken": []})
    performance_log["total_latency"] += (fin_noeud - debut_noeud)
    performance_log["total_api_calls"] += api_calls_count
    performance_log["path_taken"].append("Agent Vérificateur Principal")

    journal_agents = list(state.get("journal_agents") or [])
    journal_agents.append({
        "agent": "agent_verificateur_principal",
        "score_confiance": response.confidence_score,
        "iteration": state.get("iterations", 0)
    })

    return {
        "image_description": response.image_description,
        "lens_results": lens_results_text,
        "sub_questions": response.sub_questions,
        "confidence_score": response.confidence_score,
        "performance_metrics": performance_log,
        "journal_agents": journal_agents
    }

def agent_chercheur_web(state: AgentState) -> dict:

    questions_a_chercher = state.get("sub_questions", [])
    iteration_actuelle = state["iterations"]
    journal_agents = list(state.get("journal_agents") or [])
    journal_agents.append({
        "agent": "agent_chercheur_web",
        "iteration": iteration_actuelle,
        "questions": questions_a_chercher
    })
    nouvelles_preuves = []
    sources_collectees = list(state.get("sources_collectees") or [])

    print(f"--- AGENT CHERCHEUR WEB : Déclenchement (Itération : {state['iterations']}) ---")
    
    for question in questions_a_chercher:
        if question.strip():
            print(f"Recherche en cours pour : '{question}'")
            time.sleep(1)
            resultat = effectuer_recherche_web(question)
            nouvelles_preuves.append(
                f"Résultats pour la requête [{question}] :\n{resultat['texte']}"
            )
            sources_collectees.extend(resultat["sources"])
            
    preuves_totales = list(state.get("evidence_gathered") or [])
    if nouvelles_preuves:
        preuves_totales.extend(nouvelles_preuves)
    else:
        preuves_totales.append(f"Itération {state['iterations'] + 1} : Aucune nouvelle preuve trouvée.")

    historique_total = list(state.get("search_history") or [])
    if questions_a_chercher:
        historique_total.extend(questions_a_chercher)

    performance_log = state.get("performance_metrics", {"total_latency": 0.0, "total_api_calls": 0, "path_taken": []})
    performance_log["path_taken"].append("Agent Chercheur Web")

    # Déduplique par URL
    urls_vues = {s["url"] for s in sources_collectees}
    for source in resultat["sources"]:
        if source["url"] not in urls_vues:
            sources_collectees.append(source)
            urls_vues.add(source["url"])

    return {
        "evidence_gathered": preuves_totales,
        "search_history": historique_total,
        "sources_collectees": sources_collectees,
        "iterations": state["iterations"] + 1,
        "sub_questions": [],
        "journal_agents": journal_agents,
        "performance_metrics": performance_log
    }



def agent_synthetiseur_verdict(state: AgentState) -> dict:
    print("--- AGENT SYNTHÉTISEUR : Formulation du verdict final ---")
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    structured_llm = llm.with_structured_output(FinalVerdictStructure)
    
    from datetime import datetime
    date_actuelle = datetime.now().strftime("%Y-%m-%d")
    
    prompt_formate = SYNTHETISEUR_PROMPT.format(
        rumor_text=state.get("rumor_text", ""),
        image_description=state.get("image_description", "Aucune analyse d'image disponible."),
        search_history=", ".join(state.get("search_history", [])) if state.get("search_history") else "Aucune recherche web n'a été nécessaire.",
        evidence_gathered="\n".join(state.get("evidence_gathered", [])) if state.get("evidence_gathered") else "Aucune preuve externe collectée.",
        confidence_score=state.get("confidence_score", 0.0),
        current_date=date_actuelle
    )
    
    # Retry automatique sur erreur 503
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response: FinalVerdictStructure = structured_llm.invoke([("user", prompt_formate)])
            break
        except Exception as e:
            if "503" in str(e) and attempt < max_retries - 1:
                print(f"[Synthétiseur] Gemini surchargé, retry {attempt + 1}/{max_retries} dans 10s...")
                time.sleep(10)
            else:
                raise e
            
    score_final = response.score_final
    if score_final == 0.0:
        # Fallback : utilise le dernier score du vérificateur
        score_final = state.get("confidence_score", 0.0)

    sources_finales = sorted(
        state.get("sources_collectees", []),
        key=lambda x: x.get("score", 0),
        reverse=True
    )[:5]

    performance_log = state.get("performance_metrics", {"total_latency": 0.0, "total_api_calls": 0, "path_taken": []})
    performance_log["path_taken"].append("Agent Synthétiseur Verdict")

    payload_interface = {
        "statut": response.verdict,
        "score_certitude": score_final,
        "analyse_narrative": response.explanation,
        "sources": sources_finales,
        "journal_agents": state.get("journal_agents", []),
        "metadonnees": {
            "recherches_effectuees": state.get("search_history", []),
            "image_analyse": True if state.get("image_description") else False,
            "cycles_recherche": state.get("iterations", 0)
        },
        "performance": {
            "latence_totale_secondes": round(state.get("performance_metrics", {}).get("total_latency", 0.0), 2),
            "total_appels_api": state.get("performance_metrics", {}).get("total_api_calls", 0),
            "chemin_execution": state.get("performance_metrics", {}).get("path_taken", [])
        }
    }
    
    return {
        "verdict": response.verdict,
        "justification_json": payload_interface
    }