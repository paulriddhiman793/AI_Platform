"""
api/auth.py - Simple Mongo-backed auth helpers.
"""
from __future__ import annotations

import base64
import hashlib
import os
from typing import Optional

from pymongo import MongoClient
from pymongo.collection import Collection
try:
    import certifi  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    certifi = None


DEFAULT_EMAIL = "mada69@gmail.com"
DEFAULT_PASSWORD = "qwerty"


def _mongo_client() -> MongoClient:
    uri = (os.getenv("MONGO_URI") or "mongodb://localhost:27017").strip()
    kwargs = {}
    if uri.startswith("mongodb+srv://") or "tls=true" in uri or "ssl=true" in uri:
        if certifi:
            kwargs["tlsCAFile"] = certifi.where()
        if "tls=" not in uri and "ssl=" not in uri:
            kwargs["tls"] = True
    return MongoClient(uri, **kwargs)


def _users_collection() -> Collection:
    client = _mongo_client()
    db = client[(os.getenv("MONGO_DB") or "ai_platform").strip()]
    return db["users"]


OWASP_PBKDF2_ITERATIONS = 600_000


def _hash_password(password: str, salt_b64: Optional[str] = None, iterations: int = OWASP_PBKDF2_ITERATIONS) -> tuple[str, str, int]:
    salt = base64.b64decode(salt_b64) if salt_b64 else os.urandom(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return base64.b64encode(salt).decode("utf-8"), base64.b64encode(hashed).decode("utf-8"), iterations


def _verify_password(password: str, salt_b64: str, hash_b64: str, iterations: int = 100_000) -> bool:
    if not salt_b64 or not hash_b64:
        return False
    try:
        salt = base64.b64decode(salt_b64)
        hashed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return base64.b64encode(hashed).decode("utf-8") == hash_b64
    except Exception:
        return False


def validate_password_complexity(password: str) -> tuple[bool, str]:
    if len(password) < 12:
        return False, "Password must be at least 12 characters long."
    has_letter = any(c.isalpha() for c in password)
    has_digit_or_sym = any(not c.isalpha() for c in password)
    if not (has_letter and has_digit_or_sym):
        return False, "Password must contain at least one letter and one number or symbol."
    if password.lower() in ("password123456", "qwertyuiop1234", "abcdefghijklmn"):
        return False, "Password is too common or weak."
    return True, "OK"


def ensure_default_user() -> None:
    users = _users_collection()
    if users.find_one({"email": DEFAULT_EMAIL}):
        return
    salt, hashed, iters = _hash_password(DEFAULT_PASSWORD, iterations=OWASP_PBKDF2_ITERATIONS)
    users.insert_one({"email": DEFAULT_EMAIL, "salt": salt, "hash": hashed, "iterations": iters})


def create_user(email: str, password: str) -> tuple[bool, str]:
    email = (email or "").strip().lower()
    if not email or "@" not in email:
        return False, "Invalid email address."
    ok, msg = validate_password_complexity(password)
    if not ok:
        return False, msg
    users = _users_collection()
    if users.find_one({"email": email}):
        return False, "Email already exists."
    salt, hashed, iters = _hash_password(password, iterations=OWASP_PBKDF2_ITERATIONS)
    users.insert_one({"email": email, "salt": salt, "hash": hashed, "iterations": iters})
    return True, "User created."


def authenticate(email: str, password: str) -> bool:
    users = _users_collection()
    doc = users.find_one({"email": (email or "").strip().lower()})
    if not doc:
        return False
    stored_iters = doc.get("iterations", 100_000)
    is_valid = _verify_password(password, doc.get("salt", ""), doc.get("hash", ""), iterations=stored_iters)
    if not is_valid:
        return False
    # Lazy re-hashing: if user has < 600,000 iterations, upgrade hash transparently
    if stored_iters < OWASP_PBKDF2_ITERATIONS:
        try:
            salt, hashed, iters = _hash_password(password, iterations=OWASP_PBKDF2_ITERATIONS)
            users.update_one({"_id": doc["_id"]}, {"$set": {"salt": salt, "hash": hashed, "iterations": iters}})
        except Exception:
            pass
    return True
