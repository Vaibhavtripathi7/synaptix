"""Parse untrusted PDFs in a child process with hard resource limits.

MuPDF is a large C library parsing files strangers upload. A crafted PDF can exhaust memory,
spin forever, or exploit a parser bug. Inside the worker, none of that is containable: a thread
cannot be killed, and a crash takes the worker down. So each parse runs in its own process:

* the kernel enforces an address-space limit and a CPU-time limit (setrlimit, set by the child
  on itself before it reads a byte of the PDF), and the parent kills it at a wall-clock deadline;
* glibc arenas and OpenMP threads are capped in the child: measured on this workload, that takes
  reserved address space from ~2.6 GB to ~0.9 GB at identical memory in use and speed, which is
  what lets the address-space limit be tight instead of decorative;
* the result comes back as JSON, never pickle. If a PDF ever compromised the child, unpickling
  its output would hand the attacker code execution in the worker; JSON is only ever data.

`python -m app.ingest.sandbox <memory_mb> <cpu_seconds>` is the child: PDF bytes on stdin,
the parsed document as JSON on stdout."""

import asyncio
import base64
import json
import os
import resource
import sys

from app.db.models import ElementKind
from app.ingest.parse import Block, ParsedDoc, parse_pdf

_CHILD_ENV = {"MALLOC_ARENA_MAX": "2", "OMP_NUM_THREADS": "2"}


class ParseRejected(RuntimeError):
    """The PDF could not be parsed within the limits: too big, too slow, or malformed."""


async def parse_isolated(data: bytes, *, memory_mb: int, timeout_s: int) -> ParsedDoc:
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "app.ingest.sandbox",
        str(memory_mb),
        str(timeout_s),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ | _CHILD_ENV,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(data), timeout=timeout_s)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise ParseRejected(f"the PDF took longer than {timeout_s} s to read") from None
    except asyncio.CancelledError:
        proc.kill()  # the job was cancelled: never leave an orphan parsing in the background
        await proc.wait()
        raise
    if proc.returncode != 0:
        raise ParseRejected(_reason(proc.returncode, err))
    return _decode(json.loads(out))


def _reason(code: int, stderr: bytes) -> str:
    """A user-facing reason. The child's traceback stays out of it: it is attacker-influenced
    text and internal detail, and the error string is shown to the uploader."""
    if code < 0:  # killed by a signal: SIGXCPU for the CPU limit, SIGKILL/SIGSEGV otherwise
        return "the PDF exceeded the processing limits and was rejected"
    tail = stderr.decode(errors="replace").strip().rsplit("\n", 1)[-1]
    if "MemoryError" in tail or "std::bad_alloc" in tail:
        return "the PDF needed more memory than allowed and was rejected"
    return "the file could not be read as a PDF"


def _encode(parsed: ParsedDoc) -> dict:
    return {
        "title": parsed.title,
        "page_count": parsed.page_count,
        "blocks": [
            {
                "page": b.page,
                "role": b.role,
                "kind": b.kind.value,
                "content": b.content,
                "bbox": list(b.bbox) if b.bbox else None,
                "image_png": base64.b64encode(b.image_png).decode() if b.image_png else None,
                "meta": b.meta,
            }
            for b in parsed.blocks
        ],
    }


def _decode(payload: dict) -> ParsedDoc:
    return ParsedDoc(
        title=payload["title"],
        page_count=int(payload["page_count"]),
        blocks=[
            Block(
                page=int(b["page"]),
                role=str(b["role"]),
                kind=ElementKind(b["kind"]),
                content=str(b["content"]),
                bbox=tuple(float(x) for x in b["bbox"]) if b["bbox"] else None,
                image_png=base64.b64decode(b["image_png"]) if b["image_png"] else None,
                meta=dict(b["meta"]),
            )
            for b in payload["blocks"]
        ],
    )


def _child(memory_mb: int, cpu_seconds: int) -> None:
    limit = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    parsed = parse_pdf(sys.stdin.buffer.read())
    sys.stdout.write(json.dumps(_encode(parsed)))


if __name__ == "__main__":
    _child(int(sys.argv[1]), int(sys.argv[2]))
