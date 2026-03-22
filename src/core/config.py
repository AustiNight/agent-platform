"""
Configuration management using pydantic-settings.

All configuration is loaded from environment variables.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # LLM Provider Configuration
    # -------------------------------------------------------------------------
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL",
    )

    default_llm_provider: Literal["anthropic", "openai", "ollama"] = Field(
        default="anthropic",
        description="Default LLM provider",
    )
    default_llm_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Default model name",
    )

    # -------------------------------------------------------------------------
    # Database Configuration
    # -------------------------------------------------------------------------
    database_url: str = Field(
        default="sqlite+aiosqlite:///./agent_platform.db",
        description="Database connection string",
    )

    # -------------------------------------------------------------------------
    # Redis Configuration
    # -------------------------------------------------------------------------
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection string",
    )

    # -------------------------------------------------------------------------
    # Browser Agent Settings
    # -------------------------------------------------------------------------
    browser_headless: bool = Field(
        default=False,
        description="Run browser in headless mode",
    )
    browser_timeout: int = Field(
        default=60,
        description="Default browser timeout in seconds",
    )
    screenshot_dir: Path = Field(
        default=Path("./data/screenshots"),
        description="Directory for storing screenshots",
    )

    # -------------------------------------------------------------------------
    # API Server Settings
    # -------------------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        description="Comma-separated CORS origins",
    )
    api_secret_key: str = Field(
        default="change-me-in-production",
        description="Secret key for API security",
    )
    api_auth_key: str | None = Field(
        default=None,
        description="Optional API authentication key",
    )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: Literal["console", "json"] = Field(
        default="console",
        description="Log output format",
    )

    # -------------------------------------------------------------------------
    # Worker Settings
    # -------------------------------------------------------------------------
    worker_concurrency: int = Field(
        default=2,
        description="Number of concurrent worker tasks",
    )
    max_retries: int = Field(default=3, description="Maximum task retries")
    retry_delay_seconds: int = Field(
        default=5,
        description="Delay between retries",
    )

    # -------------------------------------------------------------------------
    # Computed Properties
    # -------------------------------------------------------------------------
    @computed_field
    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings


# Convenience alias
settings = get_settings()
