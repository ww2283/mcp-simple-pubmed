"""Offline tests for abstract truncation defaults in search results."""

import asyncio
import inspect
import xml.etree.ElementTree as ET

import pytest

from mcp_simple_pubmed import pubmed_client as pubmed_client_module
from mcp_simple_pubmed import server
from mcp_simple_pubmed.pubmed_client import PubMedClient


search_pubmed = getattr(server.search_pubmed, "fn", server.search_pubmed)

LONG_ABSTRACT = ("Tau aggregation was measured across cohorts. " * 40).strip()
SHORT_ABSTRACT = "A short abstract that comfortably fits the budget."


@pytest.fixture
def client() -> PubMedClient:
    return PubMedClient(email="test@example.com", tool="pytest-suite")


@pytest.fixture
def search_articles_calls(monkeypatch) -> list[dict]:
    calls: list[dict] = []

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


def _article(abstract_text: str) -> ET.Element:
    return ET.fromstring(
        "<PubmedArticle><MedlineCitation><PMID>37144894</PMID>"
        "<Article><ArticleTitle>Plain title</ArticleTitle>"
        f"<Abstract><AbstractText>{abstract_text}</AbstractText></Abstract>"
        "</Article></MedlineCitation></PubmedArticle>"
    )


def test_default_abstract_chars_is_300():
    # Then
    assert pubmed_client_module.DEFAULT_ABSTRACT_CHARS == 300


def test_search_articles_includes_abstracts_by_default():
    # Given
    parameters = inspect.signature(PubMedClient.search_articles).parameters

    # Then
    assert parameters["include_abstracts"].default is True
    assert (
        parameters["abstract_chars"].default
        == pubmed_client_module.DEFAULT_ABSTRACT_CHARS
    )


def test_long_abstract_is_cut_to_default_budget_and_flagged(client):
    # Given
    elem = _article(LONG_ABSTRACT)
    budget = pubmed_client_module.DEFAULT_ABSTRACT_CHARS

    # When
    article = client._parse_article_element(elem, include_abstract=True)

    # Then
    assert article is not None
    assert len(article["abstract"]) == budget
    assert article["abstract"] == LONG_ABSTRACT[:budget]
    assert article["abstract_truncated"] is True


def test_short_abstract_is_returned_whole_without_truncation_flag(client):
    # Given
    elem = _article(SHORT_ABSTRACT)

    # When
    article = client._parse_article_element(elem, include_abstract=True)

    # Then
    assert article is not None
    assert article["abstract"] == SHORT_ABSTRACT
    assert "abstract_truncated" not in article


def test_zero_abstract_chars_disables_truncation(client):
    # Given
    elem = _article(LONG_ABSTRACT)

    # When
    article = client._parse_article_element(
        elem, include_abstract=True, abstract_chars=0
    )

    # Then
    assert article is not None
    assert article["abstract"] == LONG_ABSTRACT
    assert "abstract_truncated" not in article


def test_explicit_abstract_chars_overrides_the_default_budget(client):
    # Given
    elem = _article(LONG_ABSTRACT)

    # When
    article = client._parse_article_element(
        elem, include_abstract=True, abstract_chars=50
    )

    # Then
    assert article is not None
    assert article["abstract"] == LONG_ABSTRACT[:50]
    assert article["abstract_truncated"] is True


def test_include_abstract_false_omits_the_abstract_key_entirely(client):
    # Given
    elem = _article(LONG_ABSTRACT)

    # When
    article = client._parse_article_element(
        elem, include_abstract=False, abstract_chars=50
    )

    # Then
    assert article is not None
    assert "abstract" not in article
    assert "abstract_truncated" not in article


def test_server_forwards_abstract_chars_to_client(search_articles_calls):
    # When
    asyncio.run(search_pubmed(query="crispr", abstract_chars=120))

    # Then
    assert search_articles_calls[0]["abstract_chars"] == 120


def test_server_defaults_abstract_chars_to_the_module_budget(search_articles_calls):
    # When
    asyncio.run(search_pubmed(query="crispr"))

    # Then
    assert (
        search_articles_calls[0]["abstract_chars"]
        == pubmed_client_module.DEFAULT_ABSTRACT_CHARS
    )
    assert search_articles_calls[0]["include_abstracts"] is True
