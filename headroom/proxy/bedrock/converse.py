"""Converse-to-Anthropic block shims around the compression pipeline.

An InvokeModel body for an Anthropic model already IS the Anthropic Messages
shape (the model travels in the URL), so the pipeline applies to it untranslated.
A Converse body is not: its content blocks are typeless single-key unions
(``{"text": ...}``, ``{"toolUse": ...}``, ``{"cachePoint": ...}``) while every
transform dispatches on ``block["type"]``.

These two functions add and remove that discriminator around the pipeline call.
They are deliberately the narrowest possible translation — a full Converse
dialect in the transforms would be the alternative, and a far larger surface to
get wrong.
"""

from __future__ import annotations

from typing import Any


def tag_converse_text(messages: list[Any]) -> None:
    """Add the Anthropic ``type`` discriminator to Converse text blocks, in place.

    Without this an untagged text block is invisible to every transform, which is
    why Converse bodies compressed to nothing even once they reached the
    pipeline. Tagging is purely additive and undone by :func:`untag_converse_text`
    before the body goes out.

    Text blocks only. ``toolUse``/``toolResult`` nest their payload a level
    deeper than the Anthropic equivalents, so a discriminator alone would not
    make them legible to the transforms that crush tool output; they pass
    through untouched instead of being mistranslated.
    """
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and "type" not in block
                and isinstance(block.get("text"), str)
            ):
                block["type"] = "text"


def untag_converse_text(messages: list[Any]) -> list[Any]:
    """Restore Converse block shape after the pipeline has run.

    Drops the discriminator :func:`tag_converse_text` added, and drops any
    ``cache_control`` a transform attached: Converse spells a cache breakpoint as
    a standalone ``cachePoint`` block, and minting new ones risks blowing the
    four-breakpoint limit, so the client's own ``cachePoint`` blocks (untouched,
    since no transform recognizes them) are the ones that survive. A transform
    that collapsed a message to a bare string is re-wrapped, because Converse
    accepts only a block list.
    """
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = [{"text": content}]
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                del block["type"]
                block.pop("cache_control", None)
    return messages
