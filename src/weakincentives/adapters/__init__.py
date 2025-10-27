"""Integration adapters for optional third-party providers."""

from .openai import create_openai_client

__all__ = ["create_openai_client"]
