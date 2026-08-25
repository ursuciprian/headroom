"""Bedrock-specific support for the proxy: endpoints, SigV4, Converse shims.

The request handling itself lives with its siblings in
``headroom.proxy.handlers.bedrock``; what is here is the Bedrock knowledge that
handler needs and nothing else does — which endpoint hostnames belong to AWS,
how to sign for them, and how a Converse body differs from an Anthropic one.
"""

from headroom.proxy.bedrock.converse import tag_converse_text, untag_converse_text
from headroom.proxy.bedrock.endpoints import (
    AWS_HOST_LABELS,
    BEDROCK_ACTIONS,
    aws_bedrock_region,
    aws_bedrock_signing_target,
    control_plane_base,
)
from headroom.proxy.bedrock.signing import (
    BEARER_TOKEN_ENV,
    INBOUND_SIGNATURE_HEADERS,
    SIGNING_SERVICE,
    aws_session,
    bearer_token,
    has_bearer_auth,
    sigv4_headers,
)

__all__ = [
    "AWS_HOST_LABELS",
    "BEARER_TOKEN_ENV",
    "BEDROCK_ACTIONS",
    "INBOUND_SIGNATURE_HEADERS",
    "SIGNING_SERVICE",
    "aws_bedrock_region",
    "aws_bedrock_signing_target",
    "aws_session",
    "bearer_token",
    "control_plane_base",
    "has_bearer_auth",
    "sigv4_headers",
    "tag_converse_text",
    "untag_converse_text",
]
