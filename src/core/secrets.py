"""
Secrets management.

Provides a unified interface for retrieving secrets from:
- Environment variables (default, zero-cost)
- HashiCorp Vault (production)
- AWS Secrets Manager (if needed later)

Secrets are cached in memory for the lifetime of the process.
"""

import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

import structlog

logger = structlog.get_logger()


class SecretsBackend(ABC):
    """Abstract base class for secrets backends."""

    @abstractmethod
    def get(self, key: str) -> str | None:
        """Retrieve a secret by key."""
        ...

    @abstractmethod
    def get_required(self, key: str) -> str:
        """Retrieve a required secret, raising if not found."""
        ...


class EnvironmentSecretsBackend(SecretsBackend):
    """
    Secrets from environment variables.
    
    Zero-cost, works everywhere, good for development.
    Secret keys are uppercased automatically.
    
    Example:
        backend.get("database_password") -> reads DATABASE_PASSWORD env var
    """

    def __init__(self, prefix: str = "") -> None:
        """
        Initialize environment backend.
        
        Args:
            prefix: Optional prefix for all env var names (e.g., "AGENT_")
        """
        self.prefix = prefix.upper()

    def _env_key(self, key: str) -> str:
        """Convert secret key to environment variable name."""
        return f"{self.prefix}{key.upper()}"

    def get(self, key: str) -> str | None:
        """Get secret from environment."""
        env_key = self._env_key(key)
        value = os.environ.get(env_key)
        if value:
            logger.debug("secret_retrieved", key=key, source="environment")
        return value

    def get_required(self, key: str) -> str:
        """Get required secret from environment."""
        value = self.get(key)
        if value is None:
            env_key = self._env_key(key)
            raise ValueError(
                f"Required secret '{key}' not found. "
                f"Set the {env_key} environment variable."
            )
        return value


class VaultSecretsBackend(SecretsBackend):
    """
    Secrets from HashiCorp Vault.
    
    Requires:
        - VAULT_ADDR: Vault server address
        - VAULT_TOKEN: Authentication token (or use other auth methods)
        - VAULT_SECRET_PATH: Base path for secrets (default: "secret/data/agent-platform")
    
    Install: pip install hvac
    """

    def __init__(
        self,
        addr: str | None = None,
        token: str | None = None,
        secret_path: str = "secret/data/agent-platform",
    ) -> None:
        """
        Initialize Vault backend.
        
        Args:
            addr: Vault server address (or VAULT_ADDR env var)
            token: Vault token (or VAULT_TOKEN env var)
            secret_path: Base path for secrets in Vault
        """
        try:
            import hvac
        except ImportError:
            raise ImportError(
                "HashiCorp Vault support requires hvac. "
                "Install with: pip install hvac"
            )

        self.addr = addr or os.environ.get("VAULT_ADDR", "http://localhost:8200")
        self.token = token or os.environ.get("VAULT_TOKEN")
        self.secret_path = secret_path
        
        if not self.token:
            raise ValueError(
                "Vault token required. Set VAULT_TOKEN environment variable "
                "or pass token to VaultSecretsBackend."
            )

        self.client = hvac.Client(url=self.addr, token=self.token)
        
        if not self.client.is_authenticated():
            raise ValueError("Vault authentication failed. Check your token.")
        
        logger.info("vault_connected", addr=self.addr, path=self.secret_path)

        # Cache for secrets
        self._cache: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        """Get secret from Vault."""
        # Check cache first
        if key in self._cache:
            return self._cache[key]

        try:
            # Read from Vault
            response = self.client.secrets.kv.v2.read_secret_version(
                path=f"{self.secret_path}/{key}",
                raise_on_deleted_version=True,
            )
            
            # Vault KV v2 stores data under 'data' -> 'data'
            value = response["data"]["data"].get("value")
            
            if value:
                self._cache[key] = value
                logger.debug("secret_retrieved", key=key, source="vault")
                return value
            
            return None
            
        except Exception as e:
            logger.warning("vault_secret_not_found", key=key, error=str(e))
            return None

    def get_required(self, key: str) -> str:
        """Get required secret from Vault."""
        value = self.get(key)
        if value is None:
            raise ValueError(
                f"Required secret '{key}' not found in Vault at "
                f"{self.secret_path}/{key}"
            )
        return value


class CompositeSecretsBackend(SecretsBackend):
    """
    Tries multiple backends in order.
    
    Useful for falling back from Vault to environment variables.
    """

    def __init__(self, backends: list[SecretsBackend]) -> None:
        """
        Initialize with ordered list of backends.
        
        Args:
            backends: Backends to try in order
        """
        if not backends:
            raise ValueError("At least one backend required")
        self.backends = backends

    def get(self, key: str) -> str | None:
        """Try each backend until secret is found."""
        for backend in self.backends:
            value = backend.get(key)
            if value is not None:
                return value
        return None

    def get_required(self, key: str) -> str:
        """Get required secret from any backend."""
        value = self.get(key)
        if value is None:
            raise ValueError(f"Required secret '{key}' not found in any backend")
        return value


# =============================================================================
# Singleton accessor
# =============================================================================

_secrets_backend: SecretsBackend | None = None


def configure_secrets(
    backend: str = "environment",
    **kwargs: Any,
) -> SecretsBackend:
    """
    Configure the global secrets backend.
    
    Args:
        backend: Backend type ("environment", "vault", or "composite")
        **kwargs: Backend-specific configuration
        
    Returns:
        Configured secrets backend
    """
    global _secrets_backend

    if backend == "environment":
        _secrets_backend = EnvironmentSecretsBackend(
            prefix=kwargs.get("prefix", ""),
        )
    elif backend == "vault":
        _secrets_backend = VaultSecretsBackend(
            addr=kwargs.get("addr"),
            token=kwargs.get("token"),
            secret_path=kwargs.get("secret_path", "secret/data/agent-platform"),
        )
    elif backend == "composite":
        # Default: try Vault first, fall back to environment
        backends = []
        
        # Try Vault if configured
        if os.environ.get("VAULT_ADDR") and os.environ.get("VAULT_TOKEN"):
            try:
                backends.append(VaultSecretsBackend())
            except Exception as e:
                logger.warning("vault_unavailable", error=str(e))
        
        # Always have environment as fallback
        backends.append(EnvironmentSecretsBackend())
        
        _secrets_backend = CompositeSecretsBackend(backends)
    else:
        raise ValueError(f"Unknown secrets backend: {backend}")

    logger.info("secrets_configured", backend=backend)
    return _secrets_backend


@lru_cache
def get_secrets() -> SecretsBackend:
    """
    Get the configured secrets backend.
    
    Auto-configures environment backend if not explicitly configured.
    """
    global _secrets_backend
    
    if _secrets_backend is None:
        # Auto-configure based on environment
        if os.environ.get("VAULT_ADDR") and os.environ.get("VAULT_TOKEN"):
            configure_secrets("composite")
        else:
            configure_secrets("environment")
    
    return _secrets_backend


# =============================================================================
# Convenience functions
# =============================================================================


def get_secret(key: str) -> str | None:
    """Get a secret value."""
    return get_secrets().get(key)


def get_required_secret(key: str) -> str:
    """Get a required secret value."""
    return get_secrets().get_required(key)


# Pre-defined secret keys for the platform
class SecretKeys:
    """Standard secret key names used by the platform."""
    
    ANTHROPIC_API_KEY = "anthropic_api_key"
    OPENAI_API_KEY = "openai_api_key"
    DATABASE_URL = "database_url"
    REDIS_URL = "redis_url"
    API_SECRET_KEY = "api_secret_key"
    
    # Site-specific credentials (pattern: site_domain_field)
    # Example: get_secret("example_com_username")
