"""Blob storage for source PDFs and extracted figures. Keys are `<document_id>/<name>`.

A local directory, which is right while the API and the workers share a machine. Callers go
through this class only, so an object store can replace it without touching them."""

import asyncio
import shutil
from pathlib import Path


class LocalStorage:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        p = (self._root / key).resolve()
        if not p.is_relative_to(self._root):
            raise ValueError(f"key escapes storage root: {key}")
        return p

    async def put(self, key: str, data: bytes) -> None:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(p.write_bytes, data)

    async def get(self, key: str) -> bytes:
        return await asyncio.to_thread(self.path(key).read_bytes)

    async def delete_prefix(self, prefix: str) -> None:
        await asyncio.to_thread(shutil.rmtree, self.path(prefix), True)
