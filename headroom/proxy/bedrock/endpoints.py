"""Bedrock endpoint vocabulary: which actions exist, and what a URL tells us.

The one thing the handler cannot know on its own is whether the configured
upstream is AWS itself or a gateway standing in for it, and that single fact
decides whether we re-sign. It is derivable from the endpoint hostname, so it
lives here next to the action table rather than in the request path.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

# Path suffixes we serve, mapped to (streaming, converse). Keyed by the literal
# that appears in the URL so the route decorators and the handler cannot drift.
BEDROCK_ACTIONS: dict[str, tuple[bool, bool]] = {
    "invoke": (False, False),
    "invoke-with-response-stream": (True, False),
    "converse": (False, True),
    "converse-stream": (True, True),
}

# Leading hostname label of an AWS-operated Bedrock endpoint. Model invocation
# and the control plane live on separate hosts (``bedrock-runtime.`` vs
# ``bedrock.``) but share one SigV4 signing service name.
AWS_HOST_LABELS = frozenset({"bedrock", "bedrock-fips", "bedrock-runtime", "bedrock-runtime-fips"})


def aws_bedrock_region(url: str) -> str | None:
    """Region of an AWS-operated Bedrock endpoint, or ``None`` if not one.

    ``https://bedrock-runtime.us-east-1.amazonaws.com`` yields ``us-east-1``.
    Any other host — a LiteLLM gateway, LocalStack, a corporate proxy — yields
    ``None``, and that is exactly what switches re-signing off.

    >>> aws_bedrock_region("https://bedrock-runtime.eu-west-1.amazonaws.com")
    'eu-west-1'
    >>> aws_bedrock_region("http://127.0.0.1:4000") is None
    True
    """
    host = (urlsplit(url).hostname or "").lower()
    for suffix in (".amazonaws.com", ".amazonaws.com.cn"):
        if host.endswith(suffix):
            labels = host[: -len(suffix)].split(".")
            if len(labels) == 2 and labels[0] in AWS_HOST_LABELS:
                return labels[1] or None
            return None
    return None


def control_plane_base(url: str) -> str:
    """The same endpoint on the Bedrock control plane, scheme and port kept.

    Model invocation lives on ``bedrock-runtime.<region>...``; the
    ``/inference-profiles`` listing lives on ``bedrock.<region>...``. A non-AWS
    upstream comes back unchanged, since a gateway fronts whatever it fronts on
    a single hostname.

    >>> control_plane_base("https://bedrock-runtime.us-east-1.amazonaws.com/")
    'https://bedrock.us-east-1.amazonaws.com'
    """
    if aws_bedrock_region(url) is None:
        return url.rstrip("/")
    parts = urlsplit(url)
    label, _, rest = (parts.hostname or "").lower().partition(".")
    netloc = f"{label.replace('bedrock-runtime', 'bedrock')}.{rest}"
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, "", "", ""))
