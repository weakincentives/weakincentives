"""Integration adapters for optional third-party providers."""

from .openai import OpenAIProtocol, create_openai_client

__all__ = ["create_openai_client", "OpenAIProtocol"]
