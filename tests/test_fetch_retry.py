"""Offline tests for efetch retry/backoff and surfacing of batches that never succeeded."""

import asyncio
import http.client
from pathlib import Path
from typing import Any, Callable, Dict, List, cast

import pytest

from mcp_simple_pubmed import pubmed_client as pubmed_client_module
from mcp_simple_pubmed.pubmed_client import PubMedClient


FIXTURES = Path(__file__).parent / "fixtures"

ESEARCH_PMIDS = ["29016855", "29298892", "30936201"]


class FakeHTTPResponse(http.client.HTTPResponse):
    """Passes the production isinstance gate without touching a socket."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self, amt: int | None = None) -> bytes:
        return self._payload

    def close(self) -> None:
        return None


def efetch_payload(pmids: List[str]) -> bytes:
    articles = "".join(
        "<PubmedArticle><MedlineCitation>"
        f"<PMID>{pmid}</PMID>"
        f"<Article><ArticleTitle>Title {pmid}</ArticleTitle>"
        "<Journal><Title>J Test</Title></Journal></Article>"
        "</MedlineCitation></PubmedArticle>"
        for pmid in pmids
    )
    return f"<PubmedArticleSet>{articles}</PubmedArticleSet>".encode()


class EfetchStub:
    """Records calls and serves matching PubmedArticle XML, failing on chosen attempts."""

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls: List[str] = []

    def __call__(self, **kwargs: Any) -> FakeHTTPResponse:
        requested = str(kwargs["id"])
        self.calls.append(requested)
        if len(self.calls) <= self.fail_times:
            raise RuntimeError("HTTP Error 429: Too Many Requests")
        return FakeHTTPResponse(efetch_payload(requested.split(",")))


@pytest.fixture
def sleeps(monkeypatch) -> List[float]:
    recorded: List[float] = []

    def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)

    monkeypatch.setattr(pubmed_client_module.time, "sleep", fake_sleep)
    return recorded


@pytest.fixture
def serve_esearch(monkeypatch) -> Callable[[str], None]:
    def _serve(fixture_name: str) -> None:
        payload = (FIXTURES / fixture_name).read_bytes()
        monkeypatch.setattr(
            pubmed_client_module.Entrez,
            "esearch",
            lambda **kwargs: FakeHTTPResponse(payload),
        )

    return _serve


@pytest.fixture
def serve_efetch(monkeypatch) -> Callable[[int], EfetchStub]:
    def _serve(fail_times: int = 0) -> EfetchStub:
        stub = EfetchStub(fail_times)
        monkeypatch.setattr(pubmed_client_module.Entrez, "efetch", stub)
        return stub

    return _serve


@pytest.fixture
def client() -> PubMedClient:
    return PubMedClient(email="test@example.com", tool="pytest-suite")


@pytest.fixture
def keyed_client(monkeypatch) -> PubMedClient:
    monkeypatch.setattr(
        pubmed_client_module.Entrez,
        "api_key",
        getattr(pubmed_client_module.Entrez, "api_key", None),
        raising=False,
    )
    return PubMedClient(email="test@example.com", tool="pytest-suite", api_key="k-123")


def run_search(client: PubMedClient, **kwargs: Any) -> Dict[str, Any]:
    return cast(Dict[str, Any], asyncio.run(client.search_articles(**kwargs)))


def fetch_incomplete_warnings(envelope: Dict[str, Any]) -> List[Dict[str, str]]:
    return [w for w in envelope.get("warnings", []) if w["type"] == "FetchIncomplete"]


def test_max_fetch_attempts_constant_is_three():
    assert pubmed_client_module.MAX_FETCH_ATTEMPTS == 3


def test_batch_that_succeeds_on_retry_is_indistinguishable_from_a_clean_batch(
    client, serve_esearch, serve_efetch, sleeps
):
    """A transient failure costs an extra attempt, never articles and never a warning."""
    # Given
    serve_esearch("esearch_normal.xml")
    stub = serve_efetch(fail_times=1)

    # When
    envelope = run_search(client, query="hspa8 AND tau", max_results=3)

    # Then
    assert len(stub.calls) == 2
    assert [a["pmid"] for a in envelope["articles"]] == ESEARCH_PMIDS
    assert fetch_incomplete_warnings(envelope) == []


def test_always_failing_batch_is_attempted_max_fetch_attempts_times(
    client, serve_efetch, sleeps
):
    # Given
    stub = serve_efetch(fail_times=99)

    # When
    asyncio.run(client._fetch_articles_in_batches(ESEARCH_PMIDS, batch_size=200))

    # Then
    assert len(stub.calls) == pubmed_client_module.MAX_FETCH_ATTEMPTS


def test_backoff_delay_grows_between_retry_attempts(client, serve_efetch, sleeps):
    # Given
    serve_efetch(fail_times=99)

    # When
    asyncio.run(client._fetch_articles_in_batches(ESEARCH_PMIDS, batch_size=200))

    # Then
    assert len(sleeps) >= 2
    assert all(later > earlier for earlier, later in zip(sleeps, sleeps[1:]))


def test_exhausted_batch_surfaces_fetch_incomplete_warning_naming_lost_pmid_count(
    client, serve_esearch, serve_efetch, sleeps
):
    # Given
    serve_esearch("esearch_normal.xml")
    serve_efetch(fail_times=99)

    # When
    envelope = run_search(client, query="hspa8 AND tau", max_results=3)

    # Then
    warnings = fetch_incomplete_warnings(envelope)
    assert len(warnings) == 1
    assert str(len(ESEARCH_PMIDS)) in warnings[0]["value"]


def test_clean_run_returns_every_article_and_no_fetch_incomplete_warning(
    client, serve_esearch, serve_efetch, sleeps
):
    # Given
    serve_esearch("esearch_normal.xml")
    stub = serve_efetch(fail_times=0)

    # When
    envelope = run_search(client, query="hspa8 AND tau", max_results=3)

    # Then
    assert len(stub.calls) == 1
    assert [a["pmid"] for a in envelope["articles"]] == ESEARCH_PMIDS
    assert fetch_incomplete_warnings(envelope) == []


def test_rate_limit_sleep_happens_between_batches_without_api_key(
    client, serve_efetch, sleeps
):
    # Given
    serve_efetch(fail_times=0)

    # When
    asyncio.run(client._fetch_articles_in_batches(ESEARCH_PMIDS, batch_size=1))

    # Then
    assert len(sleeps) >= 1


def test_keyed_client_still_rate_limits_between_batches_but_sleeps_less(
    client, keyed_client, serve_efetch, sleeps
):
    # Given
    serve_efetch(fail_times=0)

    # When
    asyncio.run(keyed_client._fetch_articles_in_batches(ESEARCH_PMIDS, batch_size=1))
    keyed = list(sleeps)
    sleeps.clear()
    asyncio.run(client._fetch_articles_in_batches(ESEARCH_PMIDS, batch_size=1))
    unkeyed = list(sleeps)

    # Then
    assert keyed
    assert unkeyed
    assert max(keyed) < max(unkeyed)


def esearch_payload(pmids: List[str]) -> bytes:
    ids = "".join(f"<Id>{pmid}</Id>" for pmid in pmids)
    return (
        f"<eSearchResult><Count>{len(pmids)}</Count><IdList>{ids}</IdList>"
        "<QueryTranslation>q</QueryTranslation></eSearchResult>"
    ).encode()


class SuspendingFetchClient(PubMedClient):
    """A client whose batch fetch suspends after fetching, as real async I/O would."""

    async def _fetch_articles_in_batches(self, *args: Any, **kwargs: Any) -> Any:
        result = await super()._fetch_articles_in_batches(*args, **kwargs)
        await asyncio.sleep(0)
        return result


def test_concurrent_searches_on_one_client_each_report_their_own_fetch_completeness(
    monkeypatch, sleeps
):
    # Given
    doomed_pmids = ["11111111", "22222222"]
    clean_pmids = ["33333333", "44444444"]
    by_query = {"doomed": doomed_pmids, "clean": clean_pmids}

    monkeypatch.setattr(
        pubmed_client_module.Entrez,
        "esearch",
        lambda **kwargs: FakeHTTPResponse(esearch_payload(by_query[kwargs["term"]])),
    )

    def fake_efetch(**kwargs: Any) -> FakeHTTPResponse:
        requested = str(kwargs["id"]).split(",")
        if requested[0] in doomed_pmids:
            raise RuntimeError("HTTP Error 500: Internal Server Error")
        return FakeHTTPResponse(efetch_payload(requested))

    monkeypatch.setattr(pubmed_client_module.Entrez, "efetch", fake_efetch)

    client = SuspendingFetchClient(email="test@example.com", tool="pytest-suite")

    async def both() -> Any:
        return await asyncio.gather(
            client.search_articles(query="doomed", max_results=2),
            client.search_articles(query="clean", max_results=2),
        )

    # When
    doomed_envelope, clean_envelope = asyncio.run(both())

    # Then
    assert len(fetch_incomplete_warnings(doomed_envelope)) == 1
    assert str(len(doomed_pmids)) in fetch_incomplete_warnings(doomed_envelope)[0]["value"]
    assert fetch_incomplete_warnings(clean_envelope) == []
    assert [a["pmid"] for a in clean_envelope["articles"]] == clean_pmids


def test_unfetched_pmid_count_is_never_stored_on_the_client_instance(
    client, serve_esearch, serve_efetch, sleeps
):
    """Per-call fetch state on a shared singleton client is what lets searches clobber each other."""
    # Given
    serve_esearch("esearch_normal.xml")
    serve_efetch(fail_times=99)

    # When
    run_search(client, query="hspa8 AND tau", max_results=3)

    # Then
    assert not hasattr(client, "_unfetched_pmid_count")
