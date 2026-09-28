import hashlib
from typing import Annotated

from arq import ArqRedis
from fastapi import Depends, Header, HTTPException, Request
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.limits import Admission, RateLimiter
from app.core.settings import Settings, get_settings
from app.core.validation import GEMINI_KEY, WORKSPACE_TOKEN
from app.db.session import get_session
from app.rag.answer import Answerer, MissingKeyError, answerer_for
from app.rag.cache import AnswerCache
from app.storage import Storage


def _storage(request: Request) -> Storage:
    return request.app.state.storage


def _queue(request: Request) -> ArqRedis:
    return request.app.state.queue


def _workspace(x_workspace: Annotated[str | None, Header()] = None) -> str:
    """Resolves the caller's workspace from its token.

    The token is a bearer capability — whoever holds it owns the library — so it must be long
    and random (the UI generates 256 bits). Only its SHA-256 is stored and compared: a copy of
    the database does not grant access to anyone's documents."""
    if not x_workspace or not WORKSPACE_TOKEN.fullmatch(x_workspace):
        raise HTTPException(
            401, "send an X-Workspace token: 32 to 256 characters of A-Z, a-z, 0-9, - and _"
        )
    return hashlib.sha256(x_workspace.encode()).hexdigest()


def _gemini_key(x_gemini_key: Annotated[str | None, Header()] = None) -> SecretStr | None:
    """The caller's own Gemini key, if they sent one. Wrapped at once so it is masked in every
    repr — including arq's job log lines, which print job arguments."""
    key = (x_gemini_key or "").strip()
    if not key:
        return None
    if not GEMINI_KEY.fullmatch(key):
        raise HTTPException(400, "the X-Gemini-Key header is not a well-formed Gemini API key")
    return SecretStr(key)


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


def _answer_cache(request: Request) -> AnswerCache:
    return request.app.state.answer_cache


SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
StorageDep = Annotated[Storage, Depends(_storage)]
QueueDep = Annotated[ArqRedis, Depends(_queue)]
WorkspaceDep = Annotated[str, Depends(_workspace)]
GeminiKeyDep = Annotated[SecretStr | None, Depends(_gemini_key)]
# Resolving the answerer also proves a key is available, so jobs are never queued without one.
AnswererDep = Annotated[Answerer, Depends(_answerer)]
AdmissionDep = Annotated[Admission, Depends(_admission)]
AnswerCacheDep = Annotated[AnswerCache, Depends(_answer_cache)]
ModelLimit = _limit("model")
UploadLimit = _limit("upload")
ReadLimit = _limit("read")
