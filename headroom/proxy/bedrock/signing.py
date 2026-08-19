"""Outbound credentials for requests Headroom sends on to AWS Bedrock.

A SigV4 signature covers a hash of the body and the host it was signed for. The
caller signed a different body for Headroom's own host, so an inbound signature
is void twice over by the time we forward — re-signing is not an optimization, it
is the only way a compressed request can reach AWS at all.

A Bedrock API key is the opposite: a bearer token commits to neither the body nor
the host, so it survives compression untouched and needs no work from us beyond
being left alone. :func:`has_bearer_auth` is how that case is recognized, and
:func:`bearer_token` supplies one from Headroom's own environment for callers
that have no AWS credentials to sign with at all.

Nothing here decides *whether* the upstream is AWS; that is
:func:`headroom.proxy.bedrock.endpoints.aws_bedrock_region`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

SIGNING_SERVICE = "bedrock"

# The variable AWS's own SDKs read a Bedrock API key from. Honouring the same
# name means a key that already works for boto3 or the AWS CLI works here with
# no extra configuration.
BEARER_TOKEN_ENV = "AWS_BEARER_TOKEN_BEDROCK"

# Headers that describe the inbound signature. Every one of them is void once we
# sign for ourselves, and leaving any in place makes AWS reject the request.
INBOUND_SIGNATURE_HEADERS = frozenset(
    {"authorization", "x-amz-date", "x-amz-security-token", "x-amz-content-sha256"}
)


def has_bearer_auth(headers: Mapping[str, str]) -> bool:
    """Whether these headers already carry a bearer credential.

    A Bedrock API key arrives as ``Authorization: Bearer <key>``. Unlike a SigV4
    signature it covers nothing we changed, so the right handling is to forward
    it verbatim and sign nothing.
    """
    for key, value in headers.items():
        if key.lower() == "authorization":
            return value.lstrip().lower().startswith("bearer ")
    return False


def bearer_token() -> str | None:
    """Bedrock API key from Headroom's own environment, if one is set.

    Read per request rather than at startup so a rotated key takes effect without
    a restart. Returns ``None`` for an unset or blank value.
    """
    return os.environ.get(BEARER_TOKEN_ENV, "").strip() or None


@lru_cache(maxsize=8)
def aws_session(profile: str | None, region: str) -> Any:
    """Cached boto3 session for a (profile, region) pair.

    Credentials handed out by a session refresh themselves, so one session per
    pair lasts the process lifetime. Building one costs a config-file read and
    an SSO cache lookup, which is not something to repeat per request.
    """
    import boto3

    return boto3.Session(profile_name=profile, region_name=region)


def sigv4_headers(
    *, method: str, url: str, body: bytes, region: str, profile: str | None
) -> dict[str, str]:
    """SigV4 headers for a request we are about to send to AWS.

    Signs only what SigV4 itself needs (host, timestamp, session token), so the
    pass-through headers we forward alongside stay unsigned and therefore legal.
    Raises when no credentials resolve; the caller decides what to do about it.
    """
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    credentials = aws_session(profile, region).get_credentials()
    if credentials is None:
        raise RuntimeError(f"no AWS credentials for profile={profile or 'default'}")
    signable = AWSRequest(method=method, url=url, data=body)
    SigV4Auth(credentials.get_frozen_credentials(), SIGNING_SERVICE, region).add_auth(signable)
    return dict(signable.headers)
