import hashlib
from typing import Annotated

from arq import ArqRedis
from fastapi import Depends, Header, HTTPException, Request
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.limits import Admission, RateLimiter
from app.core.settings import Settings, get_settings
from app.db.session import get_session
from app.rag.answer import Answerer, MissingKeyError, answerer_for
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


def _gemini_key(x_gemini_key: Annotated[str | None, Header()] = None) -> SecretStr | None:
    """The caller's own Gemini key, if they sent one. Wrapped at once so it is masked in every
    repr — including arq's job log lines, which print job arguments."""
    key = (x_gemini_key or "").strip()
    return SecretStr(key) if key else None


def _answerer(key: Annotated[SecretStr | None, Depends(_gemini_key)]) -> Answerer:
    try:
        return answerer_for(key.get_secret_value() if key else None)
    except MissingKeyError as exc:
        raise HTTPException(400, str(exc)) from None


def _limit(bucket: str):
    """Dependency factory: charge one request to `bucket` for the calling workspace."""

    async def check(request: Request, workspace: Annotated[str, Depends(_workspace)]) -> None:
        limiter: RateLimiter = request.app.state.limiter
        await limiter.check(workspace, bucket)

    return Depends(check)


def _admission(request: Request) -> Admission:
    return request.app.state.admission


SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
StorageDep = Annotated[LocalStorage, Depends(_storage)]
QueueDep = Annotated[ArqRedis, Depends(_queue)]
WorkspaceDep = Annotated[str, Depends(_workspace)]
GeminiKeyDep = Annotated[SecretStr | None, Depends(_gemini_key)]
# Resolving the answerer also proves a key is available, so jobs are never queued without one.
AnswererDep = Annotated[Answerer, Depends(_answerer)]
AdmissionDep = Annotated[Admission, Depends(_admission)]
ModelLimit = _limit("model")
UploadLimit = _limit("upload")
ReadLimit = _limit("read")
