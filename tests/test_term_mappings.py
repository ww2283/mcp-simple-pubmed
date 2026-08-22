"""Offline tests for surfacing PubMed's Automatic Term Mapping as envelope term_mappings."""

import asyncio
import http.client
from pathlib import Path
from typing import Any, Dict, List, cast

import pytest

from mcp_simple_pubmed import pubmed_client as pubmed_client_module
from mcp_simple_pubmed.pubmed_client import PubMedClient


FIXTURES = Path(__file__).parent / "fixtures"

TAU_JOURNAL_EXPANSION = '"Transl Androl Urol"[Journal:__jid101581119] OR "tau"[All Fields]'


class FakeHTTPResponse(http.client.HTTPResponse):
    """Passes the production isinstance gate without touching a socket."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self, amt: int | None = None) -> bytes:
        return self._payload

    def close(self) -> None:
        return None


def article(pmid: str) -> Dict[str, Any]:
    return {
        "pmid": pmid,
        "title": f"Title {pmid}",
        "journal": "J Test",
        "nlm_unique_id": "101581119",
        "authors": ["Doe Jane"],
        "keywords": [],
        "mesh_terms": [],
    }


@pytest.fixture
def client() -> PubMedClient:
    return PubMedClient(email="test@example.com", tool="pytest-suite")


@pytest.fixture
def serve_esearch(monkeypatch):
    def _serve(fixture_name: str) -> None:
        payload = (FIXTURES / fixture_name).read_bytes()

        def fake_esearch(**kwargs):
            return FakeHTTPResponse(payload)

        monkeypatch.setattr(pubmed_client_module.Entrez, "esearch", fake_esearch)

    return _serve


@pytest.fixture
def serve_articles(monkeypatch, client: PubMedClient):
    def _serve(articles: List[Dict[str, Any]]) -> None:
        async def fake_fetch(*args, **kwargs):
            return articles, 0

        monkeypatch.setattr(client, "_fetch_articles_in_batches", fake_fetch)

    return _serve


def run_search(client: PubMedClient, **kwargs: Any) -> Dict[str, Any]:
    return cast(Dict[str, Any], asyncio.run(client.search_articles(**kwargs)))


def test_journal_expansion_is_reported_as_a_single_from_to_mapping(
    client, serve_esearch, serve_articles
):
    """An untagged `tau` was silently OR'd with a journal name; the caller must see it."""
    # Given
    serve_esearch("esearch_normal.xml")
    serve_articles([article("29016855"), article("29298892"), article("30936201")])

    # When
    envelope = run_search(client, query="hspa8 AND tau", max_results=3)

    # Then
    assert envelope["term_mappings"] == [{"from": "tau", "to": TAU_JOURNAL_EXPANSION}]
    assert "[Journal" in envelope["term_mappings"][0]["to"]
    assert '"tau"[All Fields]' in envelope["term_mappings"][0]["to"]


def test_term_mappings_key_is_absent_when_quoting_suppressed_the_expansion(
    client, serve_esearch, serve_articles
):
    """Quoting `tau` is the caller's escape hatch; only meaningful against the unquoted run."""
    # Given
    serve_articles([article("29016855"), article("29298892"), article("30936201")])

    # When
    serve_esearch("esearch_normal.xml")
    unquoted = run_search(client, query="hspa8 AND tau", max_results=3)
    serve_esearch("esearch_suppressed_expansion.xml")
    quoted = run_search(client, query='hspa8 AND "tau"', max_results=3)

    # Then
    assert "term_mappings" in unquoted
    assert "term_mappings" not in quoted
    assert quoted["query_translation"] == '"hspa8"[All Fields] AND "tau"[All Fields]'


def test_term_mappings_key_is_absent_but_errors_still_surface_for_empty_translation_set(
    client, serve_esearch, serve_articles
):
    # Given
    serve_esearch("esearch_bad_field_tag.xml")
    serve_articles([article("24121476"), article("37973552"), article("38395908")])

    # When
    envelope = run_search(client, query="hspa8[NoSuchField]", max_results=3)

    # Then
    assert "term_mappings" not in envelope
    assert {"type": "FieldNotFound", "value": "NoSuchField"} in envelope["errors"]


def test_mapping_values_carry_no_whitespace_from_the_captured_xml_layout(
    client, serve_esearch, serve_articles
):
    """The wire format pads children with spaces: `<Translation>     <From>...`."""
    # Given
    serve_esearch("esearch_zero_hits.xml")
    serve_articles([])

    # When
    envelope = run_search(client, query="nonexistent term xyz", max_results=3)

    # Then
    mapping = envelope["term_mappings"][0]
    assert mapping["from"] == "term"
    assert mapping["to"].startswith('"term birth"[MeSH Terms]')
    assert mapping["to"].endswith('"term"[All Fields]')
    assert mapping["from"] == mapping["from"].strip()
    assert mapping["to"] == mapping["to"].strip()


def test_existing_envelope_fields_are_unchanged_when_mappings_are_present(
    client, serve_esearch, serve_articles
):
    # Given
    articles = [article("29016855"), article("29298892"), article("30936201")]
    serve_esearch("esearch_normal.xml")
    serve_articles(articles)

    # When
    envelope = run_search(client, query="hspa8 AND tau", max_results=3)

    # Then
    assert envelope["total_count"] == 20
    assert envelope["returned"] == 3
    assert envelope["truncated"] is True
    assert envelope["articles"] == articles
    assert envelope["query_translation"] == (
        '"hspa8"[All Fields] AND ("transl androl urol"[Journal] OR "tau"[All Fields])'
    )
