"""Offline tests for search sort ordering and max_results clamping."""

import asyncio
import inspect
import logging

import pytest

from mcp_simple_pubmed import pubmed_client as pubmed_client_module
from mcp_simple_pubmed import server
from mcp_simple_pubmed.pubmed_client import PubMedClient


@pytest.fixture
def client() -> PubMedClient:
    return PubMedClient(email="test@example.com", tool="pytest-suite")


@pytest.fixture
def esearch_calls(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    def fake_esearch(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(pubmed_client_module.Entrez, "esearch", fake_esearch)
    return calls


@pytest.fixture
def search_articles_calls(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    async def fake_search_articles(**kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(server.pubmed_client, "search_articles", fake_search_articles)
    return calls


def test_default_sort_is_relevance_and_reaches_esearch(client, esearch_calls):
    # Given
    sort_param = inspect.signature(PubMedClient.search_articles).parameters["sort"]

    # When
    asyncio.run(client.search_articles(query="crispr", max_results=5))

    # Then
    assert sort_param.default == "relevance"
    assert esearch_calls[0]["sort"] == "relevance"


def test_explicit_pub_date_sort_reaches_esearch(client, esearch_calls):
    # When
    asyncio.run(client.search_articles(query="crispr", max_results=5, sort="pub_date"))

    # Then
    assert esearch_calls[0]["sort"] == "pub_date"


def test_invalid_sort_raises_value_error_before_calling_esearch(client, esearch_calls):
    # Given
    valid_orders = pubmed_client_module.VALID_SORT_ORDERS

    # When
    with pytest.raises(ValueError) as excinfo:
        asyncio.run(client.search_articles(query="crispr", sort="popularity"))

    # Then
    assert esearch_calls == []
    assert "popularity" in str(excinfo.value)
    assert set(valid_orders) == {"relevance", "pub_date", "Author", "JournalName"}
    assert all(order in str(excinfo.value) for order in valid_orders)


def test_max_results_above_limit_is_clamped_and_warned(search_articles_calls, caplog):
    # When
    with caplog.at_level(logging.WARNING, logger="pubmed-server"):
        asyncio.run(server.search_pubmed.fn(query="crispr", max_results=10000))

    # Then
    assert search_articles_calls[0]["max_results"] == 1000
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_max_results_below_one_is_clamped_to_one_and_warned(search_articles_calls, caplog):
    # When
    with caplog.at_level(logging.WARNING, logger="pubmed-server"):
        asyncio.run(server.search_pubmed.fn(query="crispr", max_results=0))

    # Then
    assert search_articles_calls[0]["max_results"] == 1
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_in_range_max_results_passes_through_untouched(search_articles_calls):
    # Given
    limit = server.MAX_RESULTS_LIMIT

    # When
    asyncio.run(server.search_pubmed.fn(query="crispr", max_results=25))

    # Then
    assert limit == 1000
    assert search_articles_calls[0]["max_results"] == 25


def test_server_forwards_explicit_sort_to_client(search_articles_calls):
    # When
    asyncio.run(server.search_pubmed.fn(query="crispr", sort="pub_date"))

    # Then
    assert search_articles_calls[0]["sort"] == "pub_date"


def test_server_defaults_sort_to_relevance_when_caller_omits_it(search_articles_calls):
    # When
    asyncio.run(server.search_pubmed.fn(query="crispr"))

    # Then
    assert search_articles_calls[0]["sort"] == "relevance"


def test_invalid_sort_message_survives_tool_boundary_wrapping():
    # Characterization guard: the tool re-wraps exceptions, and the wrapped text
    # must still name the bad value and the valid orders.
    # When
    with pytest.raises(ValueError) as excinfo:
        asyncio.run(server.search_pubmed.fn(query="crispr", sort="bogus"))

    # Then
    message = str(excinfo.value)
    assert "bogus" in message
    assert "relevance" in message
    assert "pub_date" in message
