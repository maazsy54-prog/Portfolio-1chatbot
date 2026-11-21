"""Unit tests for settings normalisation."""

import pytest

from app.config import Settings

BASE = "u:p@ep-x-pooler.neon.tech/db"


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            f"postgresql://{BASE}?sslmode=require&channel_binding=require",
            f"postgresql+asyncpg://{BASE}?ssl=require",
        ),
        (
            f"postgres://{BASE}?sslmode=require",
            f"postgresql+asyncpg://{BASE}?ssl=require",
        ),
        (
            f"postgresql+asyncpg://{BASE}?ssl=require",
            f"postgresql+asyncpg://{BASE}?ssl=require",
        ),
    ],
)
def test_database_url_is_normalized(raw: str, expected: str) -> None:
    settings = Settings(database_url=raw, _env_file=None)
    assert settings.database_url.get_secret_value() == expected
