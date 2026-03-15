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


DEFAULT_EMAIL = "mada69@gmail.com"
DEFAULT_PASSWORD = "qwerty"


def _mongo_client() -> MongoClient:
    uri = (os.getenv("MONGO_URI") or "mongodb://localhost:27017").strip()
    return MongoClient(uri)


def _users_collection() -> Collection:
    client = _mongo_client()
    db = client[(os.getenv("MONGO_DB") or "ai_platform").strip()]
    return db["users"]


def _hash_password(password: str, salt_b64: Optional[str] = None) -> tuple[str, str]:
    salt = base64.b64decode(salt_b64) if salt_b64 else os.urandom(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return base64.b64encode(salt).decode("utf-8"), base64.b64encode(hashed).decode("utf-8")


def _verify_password(password: str, salt_b64: str, hash_b64: str) -> bool:
    salt = base64.b64decode(salt_b64)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return base64.b64encode(hashed).decode("utf-8") == hash_b64


def ensure_default_user() -> None:
    users = _users_collection()
    if users.find_one({"email": DEFAULT_EMAIL}):
        return
    salt, hashed = _hash_password(DEFAULT_PASSWORD)
    users.insert_one({"email": DEFAULT_EMAIL, "salt": salt, "hash": hashed})


def create_user(email: str, password: str) -> tuple[bool, str]:
    users = _users_collection()
    if users.find_one({"email": email}):
        return False, "Email already exists."
    salt, hashed = _hash_password(password)
    users.insert_one({"email": email, "salt": salt, "hash": hashed})
    return True, "User created."


def authenticate(email: str, password: str) -> bool:
    users = _users_collection()
    doc = users.find_one({"email": email})
    if not doc:
        return False
    return _verify_password(password, doc.get("salt", ""), doc.get("hash", ""))
