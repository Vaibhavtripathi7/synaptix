"""The PDF parser runs in a child process under kernel-enforced limits. These tests prove the
limits bite and that a hostile file ends in a clean rejection, never a crashed worker."""

import os
from pathlib import Path

import pytest

from app.ingest.parse import parse_pdf
from app.ingest.sandbox import ParseRejected, parse_isolated


async def test_isolated_parse_matches_in_process_parse(small_pdf):
    isolated = await parse_isolated(small_pdf, memory_mb=1536, timeout_s=120)
    direct = parse_pdf(small_pdf)
    assert isolated.page_count == direct.page_count and isolated.title == direct.title
    assert [(b.page, b.role, b.kind, b.content, b.bbox, b.image_png) for b in isolated.blocks] == [
        (b.page, b.role, b.kind, b.content, b.bbox, b.image_png) for b in direct.blocks
    ]


async def test_a_file_that_is_not_really_a_pdf_is_rejected_cleanly():
    with pytest.raises(ParseRejected, match="could not be read"):
        await parse_isolated(b"%PDF-1.7\n" + b"\x00garbage" * 1000, memory_mb=1536, timeout_s=60)


async def test_the_memory_limit_is_enforced_by_the_kernel(small_pdf):
    with pytest.raises(ParseRejected, match="memory|limits|could not"):
        await parse_isolated(small_pdf, memory_mb=64, timeout_s=60)


def _live_children() -> list[str]:
    """Commands of this process's living (non-zombie) children, read from /proc."""
    me, found = os.getpid(), []
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
            if int(fields[1]) == me and fields[0] != "Z":
                found.append((stat.parent / "cmdline").read_bytes().replace(b"\0", b" ").decode())
        except (OSError, IndexError, ValueError):
            continue
    return found


async def test_a_parse_over_its_deadline_is_killed_not_abandoned(small_pdf):
    with pytest.raises(ParseRejected, match="longer than"):
        await parse_isolated(small_pdf, memory_mb=1536, timeout_s=0.2)
    assert not [c for c in _live_children() if "sandbox" in c], "timed-out child left running"
