"""FastAPI dependencies that hand routes the shared components."""
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.components import Components
from app.config import Settings


def get_components(request: Request) -> Components:
    return request.app.state.components


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """One database session per request."""
    async with request.app.state.components.session_factory() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]
ComponentsDep = Annotated[Components, Depends(get_components)]
SettingsDep = Annotated[Settings, Depends(get_app_settings)]
