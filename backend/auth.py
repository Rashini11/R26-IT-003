import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError
from pwdlib import PasswordHash
from dotenv import load_dotenv


BASE_PATH = Path(__file__).resolve().parent.parent
load_dotenv(BASE_PATH / "backend" / ".env")

MONGO_URI = os.getenv("MONGO_URI", "").strip()
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "marine_ai_db").strip() or "marine_ai_db"

COOKIE_NAME = os.getenv("AUTH_COOKIE_NAME", "oceaniq_session")
COOKIE_SECURE = os.getenv("AUTH_COOKIE_SECURE", "false").lower() == "true"
SESSION_HOURS = max(1, int(os.getenv("AUTH_SESSION_HOURS", "8")))
MAX_FAILED_ATTEMPTS = max(3, int(os.getenv("AUTH_MAX_FAILED_ATTEMPTS", "5")))
LOCK_MINUTES = max(1, int(os.getenv("AUTH_LOCK_MINUTES", "10")))

UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
password_hasher = PasswordHash.recommended()
DUMMY_HASH = password_hasher.hash("OceanIQ-dummy-password-never-used")

router = APIRouter(prefix="/auth", tags=["Authentication"])

_client: Optional[MongoClient] = None
_indexes_ready = False


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=256)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=10, max_length=256)


@dataclass
class AuthContext:
    user_doc: dict
    session_doc: dict


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_username(value: str) -> str:
    return value.strip().lower()


def normalize_email(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip().lower()
    return value or None


def public_user(user: dict) -> dict:
    return {
        "id": str(user.get("_id", "")),
        "username": user.get("username"),
        "email": user.get("email"),
        "role": user.get("role", "user"),
        "is_active": bool(user.get("is_active", False)),
    }


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _auth_collections():
    global _client, _indexes_ready

    if not MONGO_URI:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication database is not configured",
        )

    try:
        if _client is None:
            _client = MongoClient(
                MONGO_URI,
                serverSelectionTimeoutMS=3000,
                connectTimeoutMS=3000,
                tz_aware=True,
            )
            _client.admin.command("ping")

        db = _client[MONGO_DB_NAME]
        users = db["users"]
        sessions = db["auth_sessions"]

        if not _indexes_ready:
            users.create_index("username_normalized", unique=True)
            users.create_index("email_normalized", unique=True, sparse=True)
            sessions.create_index("token_hash", unique=True)
            sessions.create_index("expires_at", expireAfterSeconds=0)
            sessions.create_index("user_id")
            _indexes_ready = True

        return users, sessions
    except HTTPException:
        raise
    except PyMongoError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication database is unavailable",
        ) from exc


def create_user_account(
    username: str,
    password: str,
    email: Optional[str] = None,
    role: str = "user",
) -> dict:
    username = username.strip()
    username_normalized = normalize_username(username)
    email_normalized = normalize_email(email)

    if len(username_normalized) < 3:
        raise ValueError("Username must contain at least 3 characters")
    if len(password) < 10:
        raise ValueError("Password must contain at least 10 characters")
    if role not in {"admin", "user", "viewer"}:
        raise ValueError("Role must be admin, user, or viewer")

    users, _ = _auth_collections()
    now = utc_now()
    doc = {
        "username": username,
        "username_normalized": username_normalized,
        "password_hash": password_hasher.hash(password),
        "role": role,
        "is_active": True,
        "created_at": now,
        "updated_at": now,
        "last_login_at": None,
        "failed_login_attempts": 0,
        "locked_until": None,
    }

    if email_normalized:
        doc["email"] = email.strip()
        doc["email_normalized"] = email_normalized

    try:
        result = users.insert_one(doc)
        doc["_id"] = result.inserted_id
        return public_user(doc)
    except DuplicateKeyError as exc:
        raise ValueError("Username or email already exists") from exc


def _load_auth_context(request: Request) -> AuthContext:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    users, sessions = _auth_collections()
    now = utc_now()

    try:
        session_doc = sessions.find_one(
            {
                "token_hash": _hash_token(token),
                "expires_at": {"$gt": now},
            }
        )
        if not session_doc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired or invalid",
            )

        user_doc = users.find_one(
            {
                "_id": session_doc["user_id"],
                "is_active": True,
            }
        )
        if not user_doc:
            sessions.delete_one({"_id": session_doc["_id"]})
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User is unavailable",
            )

        return AuthContext(user_doc=user_doc, session_doc=session_doc)
    except HTTPException:
        raise
    except PyMongoError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication database is unavailable",
        ) from exc


def get_auth_context(request: Request) -> AuthContext:
    return _load_auth_context(request)


def require_authenticated_request(
    request: Request,
    ctx: AuthContext = Depends(get_auth_context),
) -> dict:
    if request.method.upper() in UNSAFE_METHODS:
        supplied = request.headers.get("X-CSRF-Token", "")
        expected = ctx.session_doc.get("csrf_token", "")
        if not supplied or not expected or not hmac.compare_digest(supplied, expected):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="CSRF validation failed",
            )

    return public_user(ctx.user_doc)


def require_admin(
    user: dict = Depends(require_authenticated_request),
) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access required",
        )
    return user


@router.post("/login")
def login(payload: LoginRequest, response: Response):
    users, sessions = _auth_collections()
    username_normalized = normalize_username(payload.username)
    now = utc_now()

    try:
        user = users.find_one({"username_normalized": username_normalized})

        if not user:
            password_hasher.verify(payload.password, DUMMY_HASH)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            )

        if not user.get("is_active", False):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            )

        locked_until = user.get("locked_until")
        if locked_until and locked_until > now:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed login attempts. Try again later.",
            )

        if not password_hasher.verify(payload.password, user["password_hash"]):
            failures = int(user.get("failed_login_attempts", 0)) + 1
            update = {
                "$set": {"updated_at": now},
                "$inc": {"failed_login_attempts": 1},
            }
            if failures >= MAX_FAILED_ATTEMPTS:
                update["$set"]["locked_until"] = now + timedelta(minutes=LOCK_MINUTES)
            users.update_one({"_id": user["_id"]}, update)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            )

        users.update_one(
            {"_id": user["_id"]},
            {
                "$set": {
                    "last_login_at": now,
                    "updated_at": now,
                    "failed_login_attempts": 0,
                    "locked_until": None,
                }
            },
        )

        # Always create a brand-new session after successful authentication.
        raw_token = secrets.token_urlsafe(48)
        csrf_token = secrets.token_urlsafe(32)
        expires_at = now + timedelta(hours=SESSION_HOURS)

        sessions.insert_one(
            {
                "user_id": user["_id"],
                "token_hash": _hash_token(raw_token),
                "csrf_token": csrf_token,
                "created_at": now,
                "last_seen_at": now,
                "expires_at": expires_at,
            }
        )

        response.set_cookie(
            key=COOKIE_NAME,
            value=raw_token,
            max_age=SESSION_HOURS * 60 * 60,
            expires=expires_at,
            path="/",
            secure=COOKIE_SECURE,
            httponly=True,
            samesite="strict",
        )

        return {
            "authenticated": True,
            "user": public_user(user),
            "csrf_token": csrf_token,
        }

    except HTTPException:
        raise
    except PyMongoError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication database is unavailable",
        ) from exc


@router.get("/me")
def me(ctx: AuthContext = Depends(get_auth_context)):
    return {
        "authenticated": True,
        "user": public_user(ctx.user_doc),
        "csrf_token": ctx.session_doc.get("csrf_token"),
    }


@router.post("/logout")
def logout(
    response: Response,
    ctx: AuthContext = Depends(get_auth_context),
    _: dict = Depends(require_authenticated_request),
):
    _, sessions = _auth_collections()
    try:
        sessions.delete_one({"_id": ctx.session_doc["_id"]})
    except PyMongoError:
        # Clear the browser cookie even if the DB is momentarily unavailable.
        pass

    response.delete_cookie(
        key=COOKIE_NAME,
        path="/",
        secure=COOKIE_SECURE,
        httponly=True,
        samesite="strict",
    )
    return {"authenticated": False}


@router.post("/change-password")
def change_password(
    payload: ChangePasswordRequest,
    response: Response,
    ctx: AuthContext = Depends(get_auth_context),
    _: dict = Depends(require_authenticated_request),
):
    users, sessions = _auth_collections()

    if not password_hasher.verify(
        payload.current_password,
        ctx.user_doc["password_hash"],
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    now = utc_now()
    users.update_one(
        {"_id": ctx.user_doc["_id"]},
        {
            "$set": {
                "password_hash": password_hasher.hash(payload.new_password),
                "updated_at": now,
            }
        },
    )

    # Password changes revoke every active session for the account.
    sessions.delete_many({"user_id": ctx.user_doc["_id"]})
    response.delete_cookie(
        key=COOKIE_NAME,
        path="/",
        secure=COOKIE_SECURE,
        httponly=True,
        samesite="strict",
    )

    return {
        "message": "Password changed. Please sign in again.",
        "authenticated": False,
    }
