"""
api/schemas.py — Pydantic models for strict API validation & encryption helpers.
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field, field_validator, ConfigDict
from fastapi import HTTPException
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes


# ─── Encryption Helper (At-Rest Protection) ───────────────────────────────────

_fernet_instance: Optional[Fernet] = None


def _get_fernet() -> Fernet:
    global _fernet_instance
    if _fernet_instance is not None:
        return _fernet_instance

    key_env = os.getenv("ENCRYPTION_MASTER_KEY", "").strip()
    if key_env:
        try:
            # Ensure 32 urlsafe base64 bytes
            if len(key_env) == 44 and key_env.endswith("="):
                _fernet_instance = Fernet(key_env.encode("utf-8"))
                return _fernet_instance
        except Exception:
            pass

    # Derive from master key file inside platform root
    platform_root_env = os.getenv("PLATFORM_STORAGE_ROOT")
    if platform_root_env:
        base = Path(platform_root_env)
    else:
        base = Path(__file__).resolve().parent.parent / "platform_projects"
    base.mkdir(parents=True, exist_ok=True)
    key_path = base / ".master.key"

    if not key_path.exists():
        raw_key = Fernet.generate_key()
        try:
            key_path.write_bytes(raw_key)
            try:
                os.chmod(key_path, 0o600)
            except Exception:
                pass
        except Exception:
            pass
    else:
        raw_key = key_path.read_bytes().strip()

    try:
        _fernet_instance = Fernet(raw_key)
    except Exception:
        # Fallback derivation
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"ai_platform_static_salt_v1",
            iterations=100_000,
        )
        derived = base64.urlsafe_b64encode(kdf.derive(raw_key))
        _fernet_instance = Fernet(derived)

    return _fernet_instance


def encrypt_data(text: str) -> str:
    """Encrypt plain text or JSON string to Fernet token string."""
    if not text:
        return ""
    f = _get_fernet()
    token = f.encrypt(text.encode("utf-8"))
    return token.decode("utf-8")


def decrypt_data(token_or_text: str) -> str:
    """Decrypt Fernet token to text; gracefully falls back if raw text/JSON."""
    if not token_or_text:
        return ""
    # If it looks like plain JSON or text and not Fernet (Fernet tokens start with gAAAAA)
    token_str = token_or_text.strip()
    if not token_str.startswith("gAAAAA") or len(token_str) < 40:
        return token_or_text
    try:
        f = _get_fernet()
        dec = f.decrypt(token_str.encode("utf-8"))
        return dec.decode("utf-8")
    except Exception:
        # If decryption fails (e.g., rotated key or raw json), return original
        return token_or_text


# ─── Pydantic Request Schemas ─────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=150, description="User email address")
    password: str = Field(..., min_length=12, max_length=128, description="User password (min 12 chars)")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v):
            raise ValueError("Invalid email format.")
        return v

    @field_validator("password")
    @classmethod
    def validate_password_complexity(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 12:
            raise ValueError("Password must be at least 12 characters long.")
        has_letter = any(c.isalpha() for c in v)
        has_digit_or_sym = any(not c.isalpha() for c in v)
        if not (has_letter and has_digit_or_sym):
            raise ValueError("Password must contain at least one letter and one number or symbol.")
        if v.lower() in ("password123456", "qwertyuiop1234", "abcdefghijklmn"):
            raise ValueError("Password is too common or weak.")
        return v


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=150)
    password: str = Field(..., min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def clean_email(cls, v: str) -> str:
        return v.strip().lower()


class TokenRequest(BaseModel):
    auth_token: str = Field(..., min_length=10, max_length=128)


class WorkerExecRequest(BaseModel):
    auth_token: str = Field(..., min_length=10, max_length=128)
    command: str = Field(..., min_length=1, max_length=2000)
    cwd: Optional[str] = None
    detach: bool = False
    timeout_s: float = Field(default=300.0, ge=1.0, le=1800.0)

    @field_validator("command")
    @classmethod
    def validate_command_allowlist(cls, v: str) -> str:
        cmd = v.strip()
        if not cmd:
            raise ValueError("Command cannot be empty.")
        # Check command base executable against safe allowlist
        parts = cmd.split()
        base_cmd = parts[0].replace("\\", "/").split("/")[-1].lower()
        if base_cmd.endswith(".exe"):
            base_cmd = base_cmd[:-4]

        allowlist = {
            "python", "pytest", "git", "pip", "node", "npm", "npx",
            "uv", "echo", "cat", "ls", "dir", "mkdir", "make", "vitest", "tsc", "vite"
        }
        if base_cmd not in allowlist:
            raise ValueError(f"Command execution of '{base_cmd}' is denied by server security policy allowlist.")

        # Block dangerous chained operators or destructive commands
        if any(op in cmd for op in ["; rm -rf", "&& rm -rf", "| sh", "| bash", "rm -rf /", "mkfs"]):
            raise ValueError("Destructive commands are forbidden.")
        return cmd


class WorkerWriteFileRequest(BaseModel):
    auth_token: str = Field(..., min_length=10, max_length=128)
    path: str = Field(..., min_length=1, max_length=1024)
    content_b64: str = Field(..., min_length=1)
    cwd: Optional[str] = None
    timeout_s: float = Field(default=120.0, ge=1.0, le=600.0)

    @field_validator("content_b64")
    @classmethod
    def validate_size(cls, v: str) -> str:
        v = v.strip()
        # Max size 10MB decoded ~ 13.3MB base64
        if len(v) > 14_000_000:
            raise ValueError("File size exceeds maximum allowed worker limit (10MB).")
        return v


class FileReadRequest(BaseModel):
    auth_token: str = Field(..., min_length=10, max_length=128)
    path: str = Field(..., min_length=1, max_length=1024)


class ProjectSelectRequest(BaseModel):
    auth_token: str = Field(..., min_length=10, max_length=128)
    project_id: str = Field(..., min_length=1, max_length=256)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
