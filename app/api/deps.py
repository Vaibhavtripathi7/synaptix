import hashlib
from typing import Annotated

from arq import ArqRedis
from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.settings import Settings, get_settings
from app.db.session import get_session
from app.rag.answer import Answerer
from app.storage import LocalStorage


def _storage(request: Request) -> LocalStorage:
    return request.app.state.storage


def _queue(request: Request) -> ArqRedis:
    return request.app.state.queue


def _workspace(x_workspace: Annotated[str | None, Header()] = None) -> str:
    """Resolves the caller's workspace from its token.

    The token is a bearer capability — whoever holds it owns the library — so it must be long
    and random (the UI generates 256 bits). Only its SHA-256 is stored and compared: a copy of
    the database does not grant access to anyone's documents."""
    if not x_workspace or len(x_workspace) < 32:
        raise HTTPException(401, "send an X-Workspace token of at least 32 characters")
    return hashlib.sha256(x_workspace.encode()).hexdigest()


def _answerer(request: Request) -> Answerer:
    return request.app.state.answerer


SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
StorageDep = Annotated[LocalStorage, Depends(_storage)]
QueueDep = Annotated[ArqRedis, Depends(_queue)]
WorkspaceDep = Annotated[str, Depends(_workspace)]
AnswererDep = Annotated[Answerer, Depends(_answerer)]
