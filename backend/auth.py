import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

from bson import ObjectId
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field, field_validator
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError
from pwdlib import PasswordHash


BASE_PATH = Path(__file__).resolve().parent.parent
load_dotenv(BASE_PATH / "backend" / ".env")

MONGO_URI = os.getenv("MONGO_URI", "").strip()
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "marine_ai_db").strip() or "marine_ai_db"

COOKIE_NAME = os.getenv("AUTH_COOKIE_NAME", "oceaniq_session")
COOKIE_SECURE = os.getenv("AUTH_COOKIE_SECURE", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SESSION_HOURS = max(1, int(os.getenv("AUTH_SESSION_HOURS", "8")))
MAX_FAILED_ATTEMPTS = max(3, int(os.getenv("AUTH_MAX_FAILED_ATTEMPTS", "5")))
LOCK_MINUTES = max(1, int(os.getenv("AUTH_LOCK_MINUTES", "10")))

UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
READ_LEVELS = {"read_only", "read_write"}
WRITE_LEVELS = {"read_write"}

password_hasher = PasswordHash.recommended()
DUMMY_HASH = password_hasher.hash("OceanIQ-dummy-password-never-used")

router = APIRouter(prefix="/auth", tags=["Authentication"])

_client: Optional[MongoClient] = None
_indexes_ready = False


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=256)


class SignupRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=120)
    username: str = Field(min_length=3, max_length=80)
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=10, max_length=256)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        value = value.strip().lower()
        if "@" not in value or value.startswith("@") or value.endswith("@"):
            raise ValueError("Enter a valid email address")
        return value

    @field_validator("full_name", "username")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value cannot be empty")
        return value


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=10, max_length=256)


class AdminUserUpdateRequest(BaseModel):
    approval_status: Optional[Literal["pending", "approved", "rejected"]] = None
    access_level: Optional[Literal["none", "read_only", "read_write"]] = None
    is_active: Optional[bool] = None


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


def _iso(value):
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def public_user(user: dict) -> dict:
    role = user.get("role", "user")
    approval_status = user.get(
        "approval_status",
        "approved" if role == "admin" else "pending",
    )
    access_level = user.get(
        "access_level",
        "read_write" if role == "admin" else "none",
    )

    return {
        "id": str(user.get("_id", "")),
        "full_name": user.get("full_name"),
        "username": user.get("username"),
        "email": user.get("email"),
        "role": role,
        "approval_status": approval_status,
        "access_level": access_level,
        "is_active": bool(user.get("is_active", False)),
        "created_at": _iso(user.get("created_at")),
        "last_login_at": _iso(user.get("last_login_at")),
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
            users.create_index("approval_status")
            users.create_index("access_level")
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
    full_name: Optional[str] = None,
    approval_status: Optional[str] = None,
    access_level: Optional[str] = None,
) -> dict:
    username = username.strip()
    username_normalized = normalize_username(username)
    email_normalized = normalize_email(email)

    if len(username_normalized) < 3:
        raise ValueError("Username must contain at least 3 characters")
    if len(password) < 10:
        raise ValueError("Password must contain at least 10 characters")
    if role not in {"admin", "user"}:
        raise ValueError("Role must be admin or user")

    if role == "admin":
        approval_status = "approved"
        access_level = "read_write"
    else:
        approval_status = approval_status or "pending"
        access_level = access_level or "none"

    if approval_status not in {"pending", "approved", "rejected"}:
        raise ValueError("Invalid approval status")
    if access_level not in {"none", "read_only", "read_write"}:
        raise ValueError("Invalid access level")

    users, _ = _auth_collections()
    now = utc_now()
    doc = {
        "full_name": (full_name or "").strip() or None,
        "username": username,
        "username_normalized": username_normalized,
        "password_hash": password_hasher.hash(password),
        "role": role,
        "approval_status": approval_status,
        "access_level": access_level,
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


def _check_csrf(request: Request, ctx: AuthContext) -> None:
    if request.method.upper() not in UNSAFE_METHODS:
        return

    supplied = request.headers.get("X-CSRF-Token", "")
    expected = ctx.session_doc.get("csrf_token", "")
    if not supplied or not expected or not hmac.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF validation failed",
        )


def require_authenticated_request(
    request: Request,
    ctx: AuthContext = Depends(get_auth_context),
) -> dict:
    """
    Main OceanIQ API authorization dependency.

    Admin: full access.
    Read-write user: GET + state-changing requests.
    Read-only user: GET/HEAD/OPTIONS only.
    Pending/rejected user: authentication endpoints only; dashboard APIs are blocked.
    """
    _check_csrf(request, ctx)
    user = public_user(ctx.user_doc)

    if user.get("role") == "admin":
        return user

    approval = user.get("approval_status")
    access = user.get("access_level")

    if approval == "pending":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is awaiting administrator approval",
        )
    if approval == "rejected":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account access request was rejected",
        )
    if approval != "approved" or access not in READ_LEVELS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account does not have OceanIQ access",
        )

    if request.method.upper() in UNSAFE_METHODS and access not in WRITE_LEVELS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Read-only account: this action requires read-write access",
        )

    return user


def require_admin(
    request: Request,
    ctx: AuthContext = Depends(get_auth_context),
) -> dict:
    _check_csrf(request, ctx)
    user = public_user(ctx.user_doc)
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access required",
        )
    return user


@router.post("/signup", status_code=status.HTTP_201_CREATED)
def signup(payload: SignupRequest):
    """Public account creation. Access is always pending admin approval."""
    try:
        user = create_user_account(
            username=payload.username,
            password=payload.password,
            email=payload.email,
            role="user",
            full_name=payload.full_name,
            approval_status="pending",
            access_level="none",
        )
        return {
            "created": True,
            "message": (
                "Profile created successfully. An administrator must approve "
                "read-only or read-write access before the dashboard can be used."
            ),
            "user": user,
        }
    except ValueError as exc:
        message = str(exc)
        code = status.HTTP_409_CONFLICT if "already exists" in message else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=code, detail=message) from exc


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
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is disabled. Contact an administrator.",
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
        user["last_login_at"] = now

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
    request: Request,
    response: Response,
    ctx: AuthContext = Depends(get_auth_context),
):
    _check_csrf(request, ctx)
    _, sessions = _auth_collections()
    try:
        sessions.delete_one({"_id": ctx.session_doc["_id"]})
    except PyMongoError:
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
    request: Request,
    response: Response,
    ctx: AuthContext = Depends(get_auth_context),
):
    _check_csrf(request, ctx)
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


@router.get("/admin/users")
def admin_list_users(_: dict = Depends(require_admin)):
    users, _ = _auth_collections()
    try:
        items = list(users.find({}).sort("created_at", -1))
        return {
            "users": [public_user(user) for user in items],
            "count": len(items),
        }
    except PyMongoError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication database is unavailable",
        ) from exc


@router.patch("/admin/users/{user_id}")
def admin_update_user(
    user_id: str,
    payload: AdminUserUpdateRequest,
    _: dict = Depends(require_admin),
):
    users, sessions = _auth_collections()

    try:
        object_id = ObjectId(user_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user id",
        ) from exc

    try:
        target = users.find_one({"_id": object_id})
        if not target:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )

        if target.get("role") == "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Administrator accounts cannot be changed from this access panel",
            )

        updates = {}
        if payload.approval_status is not None:
            updates["approval_status"] = payload.approval_status
        if payload.access_level is not None:
            updates["access_level"] = payload.access_level
        if payload.is_active is not None:
            updates["is_active"] = payload.is_active

        next_approval = updates.get(
            "approval_status",
            target.get("approval_status", "pending"),
        )
        next_access = updates.get(
            "access_level",
            target.get("access_level", "none"),
        )

        if next_approval == "approved" and next_access not in READ_LEVELS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Approved users must have read-only or read-write access",
            )

        if next_approval in {"pending", "rejected"}:
            updates["access_level"] = "none"

        if not updates:
            return {"user": public_user(target)}

        updates["updated_at"] = utc_now()
        users.update_one({"_id": object_id}, {"$set": updates})

        # Revoke sessions immediately so changed permissions take effect at once.
        sessions.delete_many({"user_id": object_id})

        updated = users.find_one({"_id": object_id})
        return {
            "message": "User access updated. Existing sessions were revoked.",
            "user": public_user(updated),
        }

    except HTTPException:
        raise
    except PyMongoError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication database is unavailable",
        ) from exc
