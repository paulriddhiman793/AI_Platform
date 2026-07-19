"""Integration tests for API endpoints."""
import pytest
from httpx import AsyncClient, ASGITransport
from api.server import app


@pytest.fixture
async def client():
    """Create test client."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_healthz(client):
    """Test liveness endpoint."""
    response = await client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "ai-platform-api"


@pytest.mark.asyncio
async def test_readyz(client):
    """Test readiness endpoint."""
    response = await client.get("/readyz")
    # May be 200 or 503 depending on dependencies
    assert response.status_code in (200, 503)
    data = response.json()
    assert "status" in data
    assert "checks" in data


@pytest.mark.asyncio
async def test_cors_headers(client):
    """Test CORS headers are present."""
    response = await client.options("/healthz", headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "GET",
    })
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers