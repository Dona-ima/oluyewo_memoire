"""
Sous-module du noeud 1 : Cohérence visuelle image/rumeur

Rôle : Déterminer si le contenu visuel d'une image est cohérent avec le texte de la rumeur.

Étapes :
    1. Envoyer la paire image/rumeur à SigLIP2.
    2. Récupérer le logit image-texte et appliquer sigmoid() pour obtenir S_visual entre 0 et 1.
    3. Classification de la paire image/rumeur_texte comme "Coherente"/"Incoherente" si S_visual est inférieur à un seuil prédéfini.

    Note: Coherente = Cohérence visuelle, Incoherente = Incohérence visuelle
"""

import requests
from typing import Optional, Dict, Any
import torch
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, AutoModel


MODEL_NAME = "google/siglip2-base-patch16-224"
SEUIL_S_VISUAL = 0.5

_processor = None
_model = None


def chargement_siglip2():
    global _processor, _model
    if _model is None:
        _processor = AutoProcessor.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()

    return _processor, _model


def chargement_image(url_image: str) -> Image.Image:
    reponse = requests.get(url_image,timeout=30)
    reponse.raise_for_status()
    image = Image.open(BytesIO(reponse.content)).convert("RGB")
    return image

def score_siglip(url_image: str,texte_rumeur: str) -> Optional[float]:
    if not url_image:
        return None
    if not texte_rumeur or not texte_rumeur.strip():
        return None
    try:
        processor, model = chargement_siglip2()
        image = chargement_image(url_image)
        inputs = processor(
            text=[texte_rumeur],    
            images=image,
            padding="max_length",
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**inputs)
        logit = outputs.logits_per_image[0][0]
        score = torch.sigmoid(logit)

        return float(score.item())

    except Exception as e:
        print(f"[SigLIP2] Erreur lors du calcul : {e}")
        return None


def classifier_visual(s_visual: Optional[float], seuil: float = SEUIL_S_VISUAL) -> str:
    if s_visual is None:
        return "indetermine"
    if s_visual >= seuil:
        return "coherence_visuelle"

    return "incoherence_visuelle"


def sous_module_visual(rumor_text: str,rumor_image: Optional[str],seuil: float = SEUIL_S_VISUAL) -> Dict[str, Any]:
    if (not rumor_image or not rumor_image.startswith("http")):
        return {
            "s_visual": None,
            "statut_visual": None,
            "seuil_s_visual": seuil,
            "modele_visual": MODEL_NAME
        }
    s_visual = score_siglip(rumor_image,rumor_text)
    statut_visual = classifier_visual(s_visual,seuil)

    return {
        "s_visual": s_visual,
        "statut_visual": statut_visual,
        "seuil_s_visual": seuil,
        "modele_visual": MODEL_NAME
    }