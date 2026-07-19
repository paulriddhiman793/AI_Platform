"""
Configuration module using pydantic-settings.
"""
from functools import lru_cache
from pathlib import Path
from typing import Optional, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    public_base_url: Optional[str] = Field(
        default=None, description="Public URL for accurate startup logs"
    )
    frontend_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000",
        description="Comma-separated CORS origins",
    )
    platform_storage_root: str = Field(
        default="./platform_projects", description="Root directory for project storage"
    )

    # MongoDB
    mongo_uri: str = Field(
        default="mongodb://localhost:27017", description="MongoDB connection URI"
    )
    mongo_db: str = Field(default="ai_platform", description="MongoDB database name")

    # GitHub
    github_repo_url: Optional[str] = Field(
        default=None, description="GitHub repo URL for automated pushes"
    )

    # LLM (Groq)
    groq_api_key: Optional[str] = Field(
        default=None, description="Groq API key for LLM calls"
    )
    groq_model: str = Field(default="openai/gpt-oss-120b", description="Groq model name")
    groq_max_tokens: int = Field(default=1200, ge=256, le=32000, description="Max tokens per call")
    groq_max_concurrency: int = Field(default=1, ge=1, le=8, description="Max concurrent LLM calls")
    groq_rpm: int = Field(default=30, description="Groq requests per minute")
    groq_rpd: int = Field(default=1000, description="Groq requests per day")
    groq_tpm: int = Field(default=8000, description="Groq tokens per minute")
    groq_tpd: int = Field(default=200000, description="Groq tokens per day")
    groq_min_interval_sec: float = Field(default=1.2, description="Min interval between calls")
    groq_max_retries: int = Field(default=4, description="Max retries for rate limits")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")

    # Docker
    agent_docker_image: str = Field(default="python:3.11-slim", description="Docker image for agents")
    agent_docker_py_pkgs: str = Field(
        default="numpy pandas scikit-learn", description="Python packages to install in Docker"
    )

    # ML
    ml_optuna_trials: int = Field(default=50, description="Optuna trials per model")
    ml_n_jobs: int = Field(default=1, description="Parallel jobs for ML training")

    # Security
    groq_rate_limit_enabled: bool = Field(default=True, description="Enable Groq rate limiting")

    # Development
    ai_platform_cli_init: bool = Field(default=False, description="Enable CLI project init")
    ai_platform_force_start: bool = Field(default=False, description="Force start if lock exists")

    @field_validator("platform_storage_root", mode="before")
    @classmethod
    def expand_path(cls, v: str) -> str:
        if v:
            return str(Path(v).expanduser().resolve())
        return v

    @field_validator("frontend_origins", mode="before")
    @classmethod
    def parse_origins(cls, v: str | list[str]) -> str:
        if isinstance(v, list):
            return ",".join(v)
        return v

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.frontend_origins.split(",") if o.strip()]

    @property
    def storage_path(self) -> Path:
        path = Path(self.platform_storage_root).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()