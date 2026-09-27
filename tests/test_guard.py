"""The network guard itself: a real client with a real code path must be stopped at DNS."""

import pytest
from pydantic import SecretStr

from app.core.settings import get_settings
from app.llm.client import Gemini


async def test_a_real_model_call_from_a_test_is_blocked_before_leaving_the_machine():
    settings = get_settings().model_copy(
        update={
            "google_api_key": SecretStr("not-a-real-key"),
            "gemini_backend": "api_key",
            "llm_fallback_models": [],
        }
    )
    with pytest.raises(Exception, match="model calls must be faked"):
        await Gemini(settings).generate("hello")


async def test_the_connection_backstop_blocks_public_addresses_directly():
    """Layer 2 on its own: no hostname, no resolver, just a public IP."""
    import asyncio

    with pytest.raises(RuntimeError, match="model calls must be faked"):
        await asyncio.open_connection("142.250.1.1", 443)  # a Google address, never contacted
