"""Offline tests for offset paging."""

import asyncio
import http.client
import logging
from pathlib import Path
from typing import Any, Dict, List, cast

import pytest

from mcp_simple_pubmed import pubmed_client as pubmed_client_module
from mcp_simple_pubmed import server
from mcp_simple_pubmed.pubmed_client import PubMedClient


FIXTURES = Path(__file__).parent / "fixtures"

# fastmcp 2.x wraps the tool in a FunctionTool exposing .fn; 3.x leaves the function bare.
search_pubmed = getattr(server.search_pubmed, "fn", server.search_pubmed)


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
def serve_esearch(monkeypatch) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    payload = (FIXTURES / "esearch_normal.xml").read_bytes()

    def fake_esearch(**kwargs):
        calls.append(kwargs)
        return FakeHTTPResponse(payload)

    monkeypatch.setattr(pubmed_client_module.Entrez, "esearch", fake_esearch)
    return calls


@pytest.fixture
def serve_articles(monkeypatch, client: PubMedClient):
    def _serve(articles: List[Dict[str, Any]]) -> None:
        async def fake_fetch(*args, **kwargs):
            return articles, 0

        monkeypatch.setattr(client, "_fetch_articles_in_batches", fake_fetch)

    return _serve


@pytest.fixture
def search_articles_calls(monkeypatch) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []

    async def fake_search_articles(**kwargs):
        calls.append(kwargs)
        return {
            "total_count": 0,
            "returned": 0,
            "truncated": False,
            "query_translation": "",
            "articles": [],
        }

    monkeypatch.setattr(server.pubmed_client, "search_articles", fake_search_articles)
    return calls


def run_search(client: PubMedClient, **kwargs: Any) -> Dict[str, Any]:
    return cast(Dict[str, Any], asyncio.run(client.search_articles(**kwargs)))


def paging_warnings(envelope: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Warnings the envelope added itself; the fixture carries no NCBI WarningList."""
    return envelope.get("warnings", [])


def test_offset_reaches_esearch_as_retstart(client, serve_esearch, serve_articles):
    # Given
    serve_articles([article("29016855")])

    # When
    run_search(client, query="hspa8", max_results=10, offset=20, sort="pub_date")

    # Then
    assert str(serve_esearch[0]["retstart"]) == "20"


def test_default_offset_sends_retstart_zero_and_warns_nothing(
    client, serve_esearch, serve_articles
):
    # Given
    serve_articles([article("29016855"), article("29298892"), article("30936201")])

    # When
    envelope = run_search(client, query="hspa8", max_results=3)

    # Then
    assert str(serve_esearch[0]["retstart"]) == "0"
    assert envelope["offset"] == 0
    assert paging_warnings(envelope) == []


def test_envelope_reports_the_offset_used_and_has_more_when_hits_remain(
    client, serve_esearch, serve_articles
):
    # Given
    serve_articles([article("29016855"), article("29298892"), article("30936201")])

    # When
    envelope = run_search(client, query="hspa8", max_results=3, offset=5, sort="pub_date")

    # Then
    assert envelope["total_count"] == 20
    assert envelope["returned"] == 3
    assert envelope["offset"] == 5
    assert envelope["has_more"] is True


def test_has_more_is_false_when_the_page_reaches_the_last_hit(
    client, serve_esearch, serve_articles
):
    # Given
    serve_articles([article("29016855"), article("29298892"), article("30936201")])

    # When
    envelope = run_search(client, query="hspa8", max_results=3, offset=17, sort="pub_date")

    # Then
    assert envelope["total_count"] == 20
    assert envelope["returned"] == 3
    assert envelope["has_more"] is False


def test_paging_emits_no_warning_for_any_sort_order(
    client, serve_esearch, serve_articles
):
    # Measured live: plain retstart paging is order-stable for every sort tested,
    # so a page must not manufacture a warning about its own ordering.
    for sort in ("relevance", "pub_date", "Author", "JournalName"):
        # Given
        serve_articles([article("29016855"), article("29298892"), article("30936201")])

        # When
        envelope = run_search(client, query="hspa8", max_results=3, offset=10, sort=sort)

        # Then
        assert paging_warnings(envelope) == []


def test_negative_offset_is_clamped_to_zero_and_warned(
    client, serve_esearch, serve_articles, caplog
):
    # Given
    serve_articles([article("29016855")])

    # When
    with caplog.at_level(logging.WARNING, logger="pubmed-client"):
        envelope = run_search(client, query="hspa8", max_results=3, offset=-5, sort="pub_date")

    # Then
    assert str(serve_esearch[0]["retstart"]) == "0"
    assert envelope["offset"] == 0
    assert any(
        record.levelno == logging.WARNING and record.name == "pubmed-client"
        for record in caplog.records
    )


def test_server_forwards_offset_to_the_client(search_articles_calls):
    # When
    asyncio.run(search_pubmed(query="crispr", offset=30))

    # Then
    assert search_articles_calls[0]["offset"] == 30
