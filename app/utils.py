import base64
import os

def encoder_image_en_base64(chemin_image: str) -> str:
    """
    Lit un fichier image local et le convertit en chaîne de caractères Base64 
    prête à être envoyée à l'API Gemini.
    """
    if not chemin_image or not os.path.exists(chemin_image):
        raise FileNotFoundError(f"Le fichier image spécifié est introuvable : {chemin_image}")
        
    with open(chemin_image, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")