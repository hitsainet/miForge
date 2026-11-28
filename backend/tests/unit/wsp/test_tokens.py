"""The session token (FTASKS 5.1, unit half): random, stored as a digest, compared safely."""

from __future__ import annotations

import hashlib

from wsp.tokens import TOKEN_BYTES, digest, mint, verify


def test_mint_returns_the_plaintext_and_its_sha256() -> None:
    plaintext, sha = mint()
    assert sha == hashlib.sha256(plaintext.encode()).hexdigest() == digest(plaintext)
    assert len(plaintext) >= TOKEN_BYTES and plaintext not in sha


def test_two_mints_differ() -> None:
    assert mint()[0] != mint()[0]


def test_verify_accepts_only_the_right_token() -> None:
    plaintext, sha = mint()
    assert verify(sha, plaintext)
    assert not verify(sha, plaintext + "x")
    assert not verify(sha, mint()[0])


def test_verify_refuses_absent_values() -> None:
    plaintext, sha = mint()
    assert not verify(None, plaintext)
    assert not verify(sha, None)
    assert not verify("", "")
