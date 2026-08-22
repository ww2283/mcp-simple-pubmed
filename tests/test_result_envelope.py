"""Offline tests for the search_articles result envelope (count, truncation, diagnostics)."""

import asyncio
import http.client
from pathlib import Path
from typing import Any, Dict, List, cast

import pytest

from mcp_simple_pubmed import pubmed_client as pubmed_client_module
from mcp_simple_pubmed.pubmed_client import PubMedClient


FIXTURES = Path(__file__).parent / "fixtures"


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


def test_total_count_comes_from_esearch_count_not_article_count(
    client, serve_esearch, serve_articles
):
    # Given
    serve_esearch("esearch_normal.xml")
    serve_articles([article("29016855"), article("29298892"), article("30936201")])

    # When
    envelope = run_search(client, query="hspa8 AND tau", max_results=3)

    # Then
    assert envelope["total_count"] == 20
    assert envelope["returned"] == 3
    assert envelope["total_count"] != envelope["returned"]


def test_truncated_is_true_when_total_count_exceeds_returned(
    client, serve_esearch, serve_articles
):
    # Given
    serve_esearch("esearch_normal.xml")
    serve_articles([article("29016855"), article("29298892"), article("30936201")])

    # When
    envelope = run_search(client, query="hspa8 AND tau", max_results=3)

    # Then
    assert envelope["truncated"] is True


def test_truncated_is_false_when_every_hit_is_returned(
    client, serve_esearch, serve_articles
):
    # Given
    serve_esearch("esearch_normal.xml")
    serve_articles([article(str(30000000 + n)) for n in range(20)])

    # When
    envelope = run_search(client, query="hspa8 AND tau", max_results=20)

    # Then
    assert envelope["returned"] == 20
    assert envelope["total_count"] == 20
    assert envelope["truncated"] is False


def test_query_translation_is_passed_through_verbatim(
    client, serve_esearch, serve_articles
):
    # Given
    serve_esearch("esearch_normal.xml")
    serve_articles([article("29016855")])

    # When
    envelope = run_search(client, query="hspa8 AND tau", max_results=3)

    # Then
    assert envelope["query_translation"] == (
        '"hspa8"[All Fields] AND ("transl androl urol"[Journal] OR "tau"[All Fields])'
    )


def test_bad_field_tag_surfaces_broadened_count_and_field_not_found_error(
    client, serve_esearch, serve_articles
):
    # Given
    serve_esearch("esearch_bad_field_tag.xml")
    serve_articles([article("24121476"), article("37973552"), article("38395908")])

    # When
    envelope = run_search(client, query="hspa8[NoSuchField]", max_results=3)

    # Then
    assert envelope["total_count"] == 1419
    assert envelope["truncated"] is True
    assert envelope["query_translation"] == '"hspa8"[All Fields]'
    assert {"type": "FieldNotFound", "value": "NoSuchField"} in envelope["errors"]


def test_warnings_are_populated_from_warning_list(client, serve_esearch, serve_articles):
    # Given
    serve_esearch("esearch_zero_hits.xml")
    serve_articles([])

    # When
    envelope = run_search(client, query="nonexistent term xyz", max_results=3)

    # Then
    assert envelope["total_count"] == 0
    assert envelope["returned"] == 0
    assert envelope["truncated"] is False
    assert envelope["warnings"] == [{"type": "OutputMessage", "value": "No items found."}]
    assert {"type": "PhraseNotFound", "value": "asdkjhasdkjhasd"} in envelope["errors"]


def test_errors_and_warnings_keys_are_absent_when_ncbi_omits_them(
    client, serve_esearch, serve_articles
):
    # Given
    serve_esearch("esearch_normal.xml")
    serve_articles([article("29016855")])

    # When
    envelope = run_search(client, query="hspa8 AND tau", max_results=3)

    # Then
    assert envelope["total_count"] == 20
    assert "errors" not in envelope
    assert "warnings" not in envelope


def test_articles_list_keeps_the_existing_article_dict_shape(
    client, serve_esearch, serve_articles
):
    # Given
    articles = [article("29016855"), article("29298892"), article("30936201")]
    serve_esearch("esearch_normal.xml")
    serve_articles(articles)

    # When
    envelope = run_search(client, query="hspa8 AND tau", max_results=3)

    # Then
    assert envelope["articles"] == articles
    assert envelope["articles"][0]["pmid"] == "29016855"
    assert envelope["articles"][0]["authors"] == ["Doe Jane"]
