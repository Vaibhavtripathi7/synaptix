"""Input validation for everything that crosses the trust boundary and is not a typed request
body: headers, filenames, identifiers echoed into logs or responses.

Kept in one module so the rules can be reviewed together. Each rule is a whitelist (what is
allowed), never a blacklist (what is known to be bad): the second approach loses to the input
nobody thought of."""

import re
import unicodedata
from urllib.parse import quote

# Workspace tokens are bearer capabilities: the UI mints 256 random bits as base64url (43 chars),
# the docs mint them with secrets.token_urlsafe. Long enough to be unguessable, bounded so a
# huge header cannot be used to make every request hash megabytes.
WORKSPACE_TOKEN = re.compile(r"[A-Za-z0-9_-]{32,256}")

# Google AI Studio keys ("AIza...", 39 chars) and the newer dotted ones ("AQ." prefix). This checks
# shape only - whether a key works is known on first use - but it keeps junk out of job payloads.
GEMINI_KEY = re.compile(r"[A-Za-z0-9_.-]{20,200}")

# A client-supplied request id is echoed into every log line and a response header.
REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")

_MAX_FILENAME = 200


def clean_filename(name: str | None, default: str = "document.pdf") -> str:
    """A display name safe to store and to put in a header: the last path component only, NFC
    normalised, control and quoting characters removed, whitespace collapsed, length capped with
    the extension kept."""
    raw = (name or "").replace("\\", "/").rsplit("/", 1)[-1]
    text = unicodedata.normalize("NFC", raw)
    text = "".join(c for c in text if unicodedata.category(c)[0] != "C" and c not in "\"'`<>;")
    text = " ".join(text.split()).strip(" .")
    if not text:
        return default
    if len(text) > _MAX_FILENAME:
        stem, dot, ext = text.rpartition(".")
        keep = ext if dot and len(ext) <= 10 else ""
        text = text[: _MAX_FILENAME - len(keep) - 1].rstrip() + ("." + keep if keep else "")
    return text


def content_disposition(filename: str, disposition: str = "inline") -> str:
    """RFC 6266: an ASCII fallback for old clients plus `filename*` carrying the exact UTF-8 name.
    Built from a cleaned name, so no quote or line break can escape the header value."""
    name = clean_filename(filename)
    fallback = name.encode("ascii", "replace").decode().replace("?", "_")
    return f"{disposition}; filename=\"{fallback}\"; filename*=UTF-8''{quote(name, safe='')}"
