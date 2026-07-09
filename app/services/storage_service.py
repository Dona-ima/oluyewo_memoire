import cloudinary
import cloudinary.uploader
import os
from dotenv import load_dotenv

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

def uploader_image(chemin_local: str) -> str:
    """Upload l'image sur Cloudinary et retourne l'URL publique."""
    try:
        result = cloudinary.uploader.upload(
            chemin_local,
            folder="oluyewo",
            resource_type="image"
        )
        print(f"[Cloudinary] Image uploadée : {result['secure_url']}")
        return result["secure_url"]
    except Exception as e:
        print(f"[Cloudinary] Erreur upload : {e}")
        return ""