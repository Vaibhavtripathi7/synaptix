from typing import Annotated

from arq import ArqRedis
from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.settings import Settings, get_settings
from app.db.session import get_session
from app.storage import LocalStorage


def _storage(request: Request) -> LocalStorage:
    return request.app.state.storage


def _queue(request: Request) -> ArqRedis:
    return request.app.state.queue


SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
StorageDep = Annotated[LocalStorage, Depends(_storage)]
QueueDep = Annotated[ArqRedis, Depends(_queue)]
