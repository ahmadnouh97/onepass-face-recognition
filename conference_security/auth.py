from datetime import datetime, timedelta, timezone
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from .config import settings
from .db import get_session
from .models import Role, User

bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_token(user: User) -> str:
    expires = datetime.now(timezone.utc) + timedelta(minutes=settings.token_expire_minutes)
    return jwt.encode({"sub": user.id, "role": user.role, "exp": expires}, settings.jwt_secret, algorithm="HS256")


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except JWTError as error:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired session") from error


def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer), session: Session = Depends(get_session)) -> User:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Sign in required")
    payload = decode_token(credentials.credentials)
    user = session.get(User, payload["sub"])
    if user is None:
        raise HTTPException(status_code=401, detail="Account no longer exists")
    return user


def require_role(*roles: Role):
    def checker(user: User = Depends(current_user)) -> User:
        if user.role not in {role.value for role in roles}:
            raise HTTPException(status_code=403, detail="You do not have permission for this action")
        return user
    return checker