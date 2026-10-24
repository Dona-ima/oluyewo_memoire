import os
from datetime import datetime, timedelta, timezone
from typing import Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from dotenv import load_dotenv

load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

def hasher_password(password: str) -> str:
    return pwd_context.hash(password)

def verifier_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)

def creer_token(data: dict) -> str:
    if not SECRET_KEY:
        raise RuntimeError("La variable d'environnement SECRET_KEY est absente.")
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def obtenir_utilisateur_optionnel(
    token: str | None = Depends(oauth2_scheme)
) -> dict[str, Any] | None:
    if token is None:
        return None

    from app.database import obtenir_utilisateur_par_id
    try:
        if not SECRET_KEY:
            raise RuntimeError("La variable d'environnement SECRET_KEY est absente.")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not isinstance(user_id, str) or not user_id.isdigit():
            raise HTTPException(status_code=401, detail="Token invalide.")
    except JWTError:
        raise HTTPException(status_code=401, detail="Token invalide ou expiré.")
    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="L'authentification n'est pas configurée."
        ) from error
    try:
        user = obtenir_utilisateur_par_id(int(user_id))
    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="La base de données n'est pas configurée."
        ) from error
    if not user:
        raise HTTPException(status_code=401, detail="Utilisateur introuvable.")
    return user


def obtenir_utilisateur_courant(
    user: dict[str, Any] | None = Depends(obtenir_utilisateur_optionnel)
) -> dict[str, Any]:
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentification requise.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return user