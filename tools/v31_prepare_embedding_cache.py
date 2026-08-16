#!/usr/bin/env python3
"""Prepare the VoiceProbe v3.1 local embedding prototype cache.

No phone, Asterisk, Deepgram, or Telnyx calls are made. The only request is to
the loopback Ollama /api/embed endpoint.
"""

from __future__ import annotations

import argparse
import time

import httpx

from voiceprobe.v3.embedding_semantics import (
    DEFAULT_EMBED_URL,
    DEFAULT_MODEL,
    build_cache_payload,
    corpus_sha256,
    default_cache_path,
    training_items,
    validate_loopback_url,
    write_cache_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--url", default=DEFAULT_EMBED_URL)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    validate_loopback_url(args.url)

    items = training_items()
    texts = [text for _, text in items]
    path = default_cache_path(args.model)

    print(f"model={args.model}")
    print(f"prototype_count={len(texts)}")
    print(f"cache_path={path}")
    print(f"corpus_sha256={corpus_sha256(args.model)}")
    print("phone_calls=0")
    print("external_api_calls=0")
    print("local_ollama_embed=yes")

    started = time.perf_counter()
    with httpx.Client(timeout=args.timeout) as client:
        response = client.post(
            args.url,
            json={
                "model": args.model,
                "input": texts,
                "truncate": True,
                "keep_alive": "30m",
            },
        )
    response.raise_for_status()
    payload = response.json()
    embeddings = payload.get("embeddings")

    if not isinstance(embeddings, list):
        raise SystemExit("Ollama did not return an embeddings array.")

    cache = build_cache_payload(
        model=args.model,
        embeddings=embeddings,
    )
    write_cache_payload(cache, path=path)

    elapsed = time.perf_counter() - started

    print(f"dimension={cache['dimension']}")
    print(f"prepare_seconds={elapsed:.3f}")
    print("VOICEPROBE V3.1 EMBEDDING CACHE: PASS")


if __name__ == "__main__":
    main()
