"""
Sous-module du noeud 2 : Synthèse des résultats d'analyse

Rôle :
    Le synthétiseur reçoit les résultats produits par le noeud 1 :
        - analyse du hors-contexte de l'image (S_sem et S_ent) ;
        - analyse de la cohérence image-rumeur (classe visuelle et score).

    Il ne réalise aucune nouvelle recherche et ne vérifie pas
    directement la véracité de la rumeur.

    Son rôle est de :
        1. interpréter les résultats déjà obtenus ;
        2. expliquer la signification concrète des métriques ;
        3. présenter ces résultats dans un langage compréhensible
           par un utilisateur non technique ;
        4. conserver séparément les informations techniques.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_OOC")
MODEL_GEMINI = "gemini-3.5-flash-lite"

if not GEMINI_API_KEY:
    raise RuntimeError(
        "La variable d'environnement GEMINI_API_KEY_OOC est absente."
    )

client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# PRÉPARATION DES RÉSULTATS OOC
# ============================================================

def preparation_resultats_ooc(
    resultats: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Prépare les résultats issus de la recherche inversée
    avant leur transmission au LLM.
    """
    resultats_prepares = []

    for resultat in resultats:
        resultats_prepares.append({
            "rang": resultat.get("rang_lens"),
            "titre": resultat.get("titre", ""),
            "source": resultat.get("source", ""),
            "lien": resultat.get("lien", ""),
            "contexte": resultat.get("texte_contexte", ""),
            "score_e5": resultat.get("score_e5"),
            "entites": resultat.get("entites", {})
        })

    return resultats_prepares


# ============================================================
# CONSTRUCTION DU PROMPT
# ============================================================

import json
from typing import Dict, Any, Optional

def construction_prompt_synthese(
    rumor_text: str,
    resultat_ooc: Dict[str, Any],
    classe_visual: Optional[str],
    s_visual: Optional[float],
    explication_visual: Optional[str] = None,
) -> str:
    # 1. Extraction propre des résultats de recherche / preuves.
    #
    # BUGFIX : resultat_ooc (construit dans node_2) porte les résultats
    # bruts de la recherche inversée sous la clé française "resultats"
    # (voir node_1.py : state["resultats"] = resultat_ooc["resultats"]).
    # L'ancienne version cherchait "preuves" / "pages" / "results" (clé
    # anglaise, sans le "at") — aucune de ces clés n'existe jamais dans ce
    # dict, donc preuves_input était TOUJOURS vide, même quand la
    # recherche inversée avait bien trouvé des sources. Résultat : le LLM
    # croyait systématiquement qu'aucune preuve n'avait été trouvée.
    #
    # On utilise maintenant preparation_resultats_ooc() (déjà définie
    # plus haut mais jamais appelée jusqu'ici) pour transformer les
    # résultats bruts (rang_lens, titre, source, lien, texte_contexte,
    # score_e5, entites) dans le format attendu par le prompt.
    preuves_input = preparation_resultats_ooc(resultat_ooc.get("resultats", []))

    donnees_json = json.dumps({
        "rumor_text": rumor_text,
        "resultat_ooc": resultat_ooc,
        "preuves_disponibles": preuves_input,
        "classe_visual": classe_visual,
        "score_visual": s_visual,
        "explication_visual": explication_visual,
    }, ensure_ascii=False, indent=2)


    prompt = f"""
        Tu es le module de synthèse de la plateforme Oluyewo, un système d'analyse de l'association entre une image et un texte.

        Ton rôle est d'interpréter les résultats produits par les deux branches d'analyse, de décider si les éléments disponibles indiquent un détournement de contexte (OOC), puis d'expliquer le résultat au citoyen.

        Tu ne dois jamais déterminer si la rumeur est vraie ou fausse dans l'absolu.

        ============================================================
        1. RÔLES DES MODÈLES
        ============================================================

        Deux types de résultats te sont fournis :

        - Analyse OOC : scores sémantique et entités, entités extraites,
        comparaison des entités et résultats de la recherche inversée.
        - Analyse visuelle : classe, score de confiance et explication retournés
        par le modèle PaliGemma fine-tuné.

        Le champ resultat_ooc.classe_ooc est la décision autoritaire produite
        par le module de classification. Tu ne dois pas refaire ni modifier
        cette classification.

        En revanche, la classification visuelle est déjà effectuée par
        PaliGemma. Tu dois reprendre sa classe et son score tels qu'ils
        figurent dans les données fournies, sans refaire la classification,
        les recalculer ou les modifier.

        Pour la présentation destinée au citoyen, utilise la correspondance
        suivante si la classe reçue est exprimée sous forme de labels SNLI-VE :
        - "entailment" → "coherente"
        - "contradiction" → "incoherente"
        - "neutral" → "neutre"

        Si la classe est déjà exprimée sous la forme "coherente",
        "incoherente" ou "neutre", reprends-la telle quelle.
        Si la classe ou le score visuel est absent, conserve cette absence
        sans inventer de valeur.
        L'explication visuelle fournie est le texte produit par PaliGemma après
        traduction en français. Présente-la comme une explication du modèle,
        pas comme une preuve indépendante.
        Si cette explication est absente ou vide, ne prétends pas que PaliGemma
        a fourni une explication et n'en invente pas une.

        ============================================================
        2. CLASSE OOC FOURNIE
        ============================================================

        Recopie exactement resultat_ooc.classe_ooc dans le champ "verdict"
        de la sortie. Cette valeur a déjà été déterminée par le classificateur
        à partir des scores, entités et résultats de recherche inversée.
        N'infère pas, ne réévalue pas et ne remplace pas cette classe, même si
        tu interprètes différemment les éléments. 
        Ta tâche est de produire une synthèse fidèle aux données. Le signal
        visuel peut être expliqué séparément, mais ne doit jamais modifier le
        verdict fourni.

        ============================================================
        3. RÉSUMÉ DESTINÉ AU CITOYEN
        ============================================================

        RÈGLE ESSENTIELLE : tout ce qui apparaît entre guillemets dans
        cette section (exemples, formulations, phrases types) sert
        uniquement à illustrer le ton et l'idée. Ne les recopie jamais
        telles quelles. Reformule avec tes propres mots, en les adaptant
        aux données réelles de cette vérification : la rumeur, les
        lieux, les dates et les sources concernés. Les consignes
        (ce qui est écrit sans guillemets) décrivent ce que le résumé
        doit faire. Ce ne sont jamais des phrases à écrire.
        Le citoyen ne doit jamais lire de formulation qui ressemble à
        une instruction ou à une consigne.

        Le champ "resume" s'adresse à une personne qui n'a aucune
        connaissance technique et qui lit sur son téléphone. Écris comme
        un journaliste qui explique à un proche, en 4 à 6 phrases courtes.

        ORDRE À SUIVRE :
        1. Commence directement par ce que nous avons constaté sur
            l'image. Ne répète pas et ne résume pas la rumeur : le
            citoyen vient de la saisir.
            Si verdict = "ooc" : l'image semble provenir d'un autre
            événement que celui décrit.
            Si verdict = "nooc" : les publications retrouvées vont dans
            le sens de ce que décrit la rumeur et aucune ne rattache
            l'image à un autre moment ou lieu. Le citoyen doit
            comprendre que cela ne prouve pas que la rumeur est vraie,
            mais tu l'exprimes avec tes propres mots, une seule fois.
        2. Explique pourquoi, avec les faits : où et quand l'image
           est déjà apparue, et ce que disaient ces publications.
           Cite la date de l'événement d'origine et compare-la à ce
           que dit la rumeur.
        3. Termine par une précaution simple si nécessaire (par
           exemple : l'image ressemble à celle de ces articles mais on
           ne peut pas être sûr qu'il s'agisse exactement de la même).

        STYLE :
        - Vocabulaire de tous les jours.
        - Interdit : "recherche inversée", "analyse visuelle",
          "score", "similarité", "entités", "modèle", "classe",
          "neutre", "cohérente", "détournement de contexte".
        - Dis plutôt : "nous avons retrouvé cette image dans...",
          "elle accompagnait un article du...", "elle ne semble pas
          correspondre à...".
        - Ne parle jamais de PaliGemma ni de pourcentages.
        - Ne recopie ni ne résume la rumeur en introduction. Tu peux
          y faire référence brièvement, seulement quand c'est
          nécessaire pour une comparaison (date, lieu, circonstances).

        INTÉGRATION DE CE QUE MONTRE L'IMAGE :
        Le contenu de l'image n'est jamais un paragraphe à part, ni une
        phrase finale. Il s'intègre en UNE phrase, placée juste après
        les preuves d'origine, et il doit renforcer le raisonnement,
        jamais le contredire ni le relativiser.

        Choisis la formulation selon le cas :

        - verdict "ooc" et image qui correspond à la description de la
          rumeur : "L'image montre bien [ce que décrit la rumeur], mais
          c'est un autre événement : elle date de [date d'origine]."
          La ressemblance explique pourquoi l'image trompe, elle ne
          rend pas la rumeur plus crédible.
        - verdict "ooc" et image qui ne correspond pas : "De plus, ce
          que montre l'image ne correspond pas à ce que décrit la
          rumeur."
        - verdict "nooc" et image qui correspond : "Ce que montre
          l'image correspond à la description, et nous n'avons rien
          trouvé qui la rattache à un autre événement."
        - classe visuelle neutre ou absente : ne présente pas cela
          comme un doute. Dis seulement ce que l'image montre, sans
          phrase du type "rien ne prouve" ou "rien ne permet de dire".
        - explication visuelle absente : n'en parle pas.

        INTERDIT : répéter une précaution déjà exprimée, finir par une
        suite de négations ("mais rien ne prouve, rien n'indique..."),
        ou laisser l'analyse visuelle nuancer un verdict déjà énoncé.
        Le verdict ne change jamais à cause de l'image.

        INTÉGRATION DES SOURCES :

        - Lorsqu'une affirmation s'appuie sur une source externe, intègre
        immédiatement dans la phrase le lien Markdown cliquable vers
        la page qui étaye cette affirmation.
        - Utilise uniquement les URL présentes dans les données fournies.
        N'invente, ne complète et ne modifie aucune URL.
        - Associe chaque lien à l'information précise qu'il permet d'étayer.
        - Privilégie les sources pertinentes qui donnent des informations
        sur le contexte de l'image.
        - Si une source concerne une image similaire sans prouver qu'il
        s'agit de la même photographie, indique cette limite.
        - Si aucune source pertinente n'est disponible, explique que les
        recherches n'ont pas permis d'établir le contexte d'origine.
        - N'affirme jamais qu'une information est confirmée si les éléments
        disponibles ne permettent pas de l'établir.

        Ne présente pas l'absence de résultat comme une preuve de
        détournement. Ne fabrique aucun lieu, aucune date, aucun événement
        ni aucun détail visuel. Évite le jargon technique et les
        formulations répétitives.

        ============================================================
        4. INFORMATIONS STRUCTURÉES
        ============================================================

        - "analyse_contexte" : présente les éléments qui permettent de
        comprendre le contexte retrouvé et sa relation avec le texte.
        - "preuves" : conserve les sources disponibles avec leur titre,
        leur site, leur URL exacte et leur contexte pertinent. Si aucune
        source exploitable n'est disponible, retourne [].
        - "analyse_visuelle_image_rumeur.classe" : restitue la classe
        fournie par PaliGemma dans la forme attendue par l'interface.
        - "analyse_visuelle_image_rumeur.conclusion" et "explication" :
        reprennent fidèlement l'explication visuelle fournie et la mettent en
        contexte, sans produire une nouvelle prédiction ni inventer des détails.
        - "donnees_techniques" : restitue la classe_ooc, les scores, classes et entités
        reçus dans les données d'entrée. Ne recalcule pas ces valeurs
        et n'en invente aucune.
        - "mise_en_relation", "conclusion" et "avertissement" :
        restent cohérents avec le verdict et le résumé. Évite les
        répétitions inutiles. Signale les limites lorsque nécessaire.
        - Dans "analyse_visuelle_image_rumeur", "conclusion" tient en une
        phrase qui dit si l'image correspond à la description de la
        rumeur, et "explication" décrit simplement ce qu'on voit.
        Ces champs ne répètent pas le résumé et ne remettent pas le
        verdict en cause. Ne dis jamais "neutre".

        ============================================================
        5. INTERDICTIONS
        ============================================================

        - Ne détermine pas la véracité générale de la rumeur.
        - Ne refais pas la classification visuelle réalisée par PaliGemma.
        - Ne modifie pas son score de confiance.
        - N'invente aucune preuve, URL, entité, date ou information.
        - Ne présente pas un score comme une preuve certaine.
        - Ne confonds pas une absence de preuve avec une preuve de
        détournement.
        - Ne présente pas une origine comme confirmée sans éléments
        suffisamment pertinents.

        ============================================================
        6. FORMAT DE SORTIE
        ============================================================

        Retourne UNIQUEMENT un objet JSON valide, sans Markdown autour,
        en respectant exactement cette structure :

        {{
            "resultat_utilisateur": {{
                "titre": "Rapport d'analyse de l'association image-texte",
                "verdict": "ooc | nooc",
                "resume": "...",
                "analyse_contexte": {{
                    "conclusion": "...",
                    "explication": "...",
                    "comparaison": [
                        {{
                            "element": "Événement / Contexte",
                            "rumeur": "Incendie récent au Port de Cotonou",
                            "resultats_retrouves": "Incendie lié à de l'essence de contrebande (Sèmè-Kpodji)",
                            "interpretation": "L'image ne montre pas le Port de Cotonou."
                        }}
                    ],
                    "preuves": [
                        {{
                            "rang": 1,
                            "titre": "Titre exact de l'article retrouvé",
                            "source": "Nom du site / Média",
                            "lien": "https://...",
                            "contexte": "Bref résumé du contexte de l'article"
                        }}
                    ]
                }},
                "analyse_visuelle_image_rumeur": {{
                    "classe": "coherente",
                    "conclusion": "...",
                    "explication": "..."
                }},
                "mise_en_relation": "...",
                "conclusion": "...",
                "avertissement": "..."
            }},
            "donnees_techniques": {{
                "classe_ooc": "ooc",
                "score_semantique": 0.78,
                "score_entites": 0.0,
                "classe_visual": "coherente",
                "score_visual": 0.92,
                "entites_rumeur": [
                    "Cotonou",
                    "Port"
                ],
                "entites_source": [
                    "Sèmè-Kpodji",
                    "Port",
                    "Bénin"
                ]
            }}
        }}

        Les valeurs ci-dessus illustrent uniquement la structure attendue :
        elles ne constituent pas des résultats à recopier. Remplace-les
        par les valeurs réellement présentes dans les données d'entrée.
        Si une valeur n'est pas disponible, utilise null ou une liste vide
        selon le type du champ.

        ============================================================
        DONNÉES À ANALYSER
        ============================================================

        {donnees_json}
        """

    return prompt


def _erreur_serveur_gemini(exc: BaseException) -> bool:
    """Réessaie seulement sur les erreurs serveur temporaires (5xx)."""
    return isinstance(exc, genai_errors.ServerError)


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=3, min=3, max=20),
    retry=retry_if_exception(_erreur_serveur_gemini),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _generer_synthese(prompt: str):
    return client.models.generate_content(
        model=MODEL_GEMINI,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )



# ============================================================
# APPEL DU LLM
# ============================================================

def appel_llm_synthese(
    prompt: str
) -> Dict[str, Any]:
    """
    Appelle Gemini et récupère la synthèse au format JSON.
    """
    try:
        response = _generer_synthese(prompt)

        texte = response.text.strip()
        resultat = json.loads(texte)
        return resultat

    except json.JSONDecodeError as e:
        logger.error(
            "La réponse du modèle n'est pas un JSON valide : %s",
            e
        )
        raise ValueError(
            "Le modèle de synthèse a retourné un JSON invalide."
        ) from e

    except Exception as e:
        logger.exception(
            "Erreur lors de l'appel au modèle de synthèse."
        )
        raise RuntimeError(
            "Erreur lors de la génération de la synthèse."
        ) from e


# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def sous_module_synthese(
    rumor_text: str,
    resultat_ooc: Dict[str, Any],
    classe_visual: Optional[str],
    s_visual: Optional[float],
    explication_visual: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fonction principale du sous-module de synthèse.
    """
    if not rumor_text or not rumor_text.strip():
        raise ValueError(
            "Le texte de la rumeur est obligatoire."
        )

    if not isinstance(resultat_ooc, dict):
        raise TypeError(
            "resultat_ooc doit être un dictionnaire."
        )
    if resultat_ooc.get("classe_ooc") not in {"ooc", "nooc"}:
        raise ValueError("resultat_ooc.classe_ooc doit être 'ooc' ou 'nooc'.")

    prompt = construction_prompt_synthese(
        rumor_text=rumor_text,
        resultat_ooc=resultat_ooc,
        classe_visual=classe_visual,
        s_visual=s_visual,
        explication_visual=explication_visual,
    )

    justification = appel_llm_synthese(prompt)

    classe_ooc = resultat_ooc.get("classe_ooc")
    resultat_utilisateur = justification.get("resultat_utilisateur")
    if not isinstance(resultat_utilisateur, dict):
        raise ValueError("La synthèse ne contient pas d'objet resultat_utilisateur valide.")
    resultat_utilisateur["verdict"] = classe_ooc
    donnees_techniques = justification.get("donnees_techniques")
    if isinstance(donnees_techniques, dict):
        donnees_techniques["classe_ooc"] = classe_ooc

    return {
        "statut": "succes",
        "justification": justification
    }