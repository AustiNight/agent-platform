"""
LLM client abstraction layer.

Provides a unified interface for multiple LLM providers using LiteLLM.
"""

from typing import Any

import litellm
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import settings

logger = structlog.get_logger()

# Configure LiteLLM
litellm.set_verbose = False


class LLMClient:
    """
    Unified LLM client supporting multiple providers.
    
    Supports:
    - Anthropic (Claude models)
    - OpenAI (GPT models)
    - Ollama (local models)
    """

    # Model name mappings for LiteLLM
    PROVIDER_PREFIXES = {
        "anthropic": "",  # LiteLLM handles Anthropic natively
        "openai": "",     # LiteLLM handles OpenAI natively
        "ollama": "ollama/",  # Prefix for Ollama models
    }

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize the LLM client.
        
        Args:
            provider: LLM provider (anthropic, openai, ollama)
            model: Model name
            api_key: Optional API key override
        """
        self.provider = provider or settings.default_llm_provider
        self.model = model or settings.default_llm_model
        self._api_key = api_key

        # Set up API keys based on provider
        self._configure_provider()

        self.logger = logger.bind(provider=self.provider, model=self.model)

    def _configure_provider(self) -> None:
        """Configure provider-specific settings."""
        if self.provider == "anthropic":
            key = self._api_key or settings.anthropic_api_key
            if key:
                litellm.anthropic_key = key
        elif self.provider == "openai":
            key = self._api_key or settings.openai_api_key
            if key:
                litellm.openai_key = key
        elif self.provider == "ollama":
            # Ollama doesn't need an API key
            litellm.api_base = settings.ollama_base_url

    def _get_model_string(self, model: str | None = None) -> str:
        """Get the full model string for LiteLLM."""
        model_name = model or self.model
        prefix = self.PROVIDER_PREFIXES.get(self.provider, "")
        return f"{prefix}{model_name}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Send a chat completion request.
        
        Args:
            messages: List of messages in the conversation
            tools: Optional list of tool definitions
            model: Model override
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Parsed response with content, tool_calls, usage, etc.
        """
        model_string = self._get_model_string(model)
        
        self.logger.debug(
            "llm_request",
            model=model_string,
            message_count=len(messages),
            has_tools=bool(tools),
        )

        try:
            # Prepare the request
            request_params = {
                "model": model_string,
                "messages": self._format_messages(messages),
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs,
            }

            # Add tools if provided
            if tools:
                request_params["tools"] = self._format_tools(tools)

            # Make the request
            response = await litellm.acompletion(**request_params)

            # Parse and return
            return self._parse_response(response)

        except Exception as e:
            self.logger.error("llm_error", error=str(e))
            raise

    def _format_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Format messages for LiteLLM."""
        formatted = []
        
        for msg in messages:
            role = msg.get("role", "user")
            
            if role == "system":
                # Handle system messages (Anthropic uses system parameter)
                formatted.append({
                    "role": "system",
                    "content": msg.get("content", ""),
                })
            elif role == "tool":
                # Tool results need special formatting
                formatted.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                })
            else:
                formatted.append({
                    "role": role,
                    "content": msg.get("content", ""),
                })

        return formatted

    def _format_tools(
        self,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Format tools for LiteLLM (OpenAI function calling format)."""
        formatted = []
        
        for tool in tools:
            formatted.append({
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })

        return formatted

    def _parse_response(self, response: Any) -> dict[str, Any]:
        """Parse LiteLLM response into a standard format."""
        choice = response.choices[0]
        message = choice.message

        result = {
            "content": message.content or "",
            "stop_reason": self._map_stop_reason(choice.finish_reason),
            "usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }

        # Handle tool calls
        if message.tool_calls:
            result["stop_reason"] = "tool_use"
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": self._parse_tool_arguments(tc.function.arguments),
                }
                for tc in message.tool_calls
            ]

        return result

    def _map_stop_reason(self, finish_reason: str | None) -> str:
        """Map provider-specific stop reasons to a standard format."""
        mapping = {
            "stop": "end_turn",
            "end_turn": "end_turn",
            "tool_calls": "tool_use",
            "tool_use": "tool_use",
            "length": "max_tokens",
            "max_tokens": "max_tokens",
        }
        return mapping.get(finish_reason or "stop", "end_turn")

    def _parse_tool_arguments(self, arguments: str | dict) -> dict[str, Any]:
        """Parse tool arguments which may be JSON string or dict."""
        if isinstance(arguments, dict):
            return arguments
        
        import json
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return {"raw": arguments}

    async def embed(
        self,
        text: str | list[str],
        model: str | None = None,
    ) -> list[list[float]]:
        """
        Generate embeddings for text.
        
        Args:
            text: Text or list of texts to embed
            model: Embedding model override
            
        Returns:
            List of embedding vectors
        """
        # Normalize to list
        texts = [text] if isinstance(text, str) else text
        
        # Use default embedding model based on provider
        if model is None:
            if self.provider == "openai":
                model = "text-embedding-3-small"
            elif self.provider == "ollama":
                model = "nomic-embed-text"
            else:
                # Anthropic doesn't have embeddings, fall back to OpenAI
                model = "text-embedding-3-small"

        response = await litellm.aembedding(
            model=model,
            input=texts,
        )

        return [item["embedding"] for item in response.data]


# Factory function for convenience
def create_llm_client(
    provider: str | None = None,
    model: str | None = None,
) -> LLMClient:
    """Create an LLM client with the specified configuration."""
    return LLMClient(provider=provider, model=model)
