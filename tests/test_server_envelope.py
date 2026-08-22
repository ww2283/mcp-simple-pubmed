"""Offline tests for the search_pubmed tool boundary: envelope pass-through and file output."""

import asyncio
import json
import re

import pytest

from mcp_simple_pubmed import server


# fastmcp 2.x wraps the tool in a FunctionTool exposing .fn; 3.x leaves the function bare.
search_pubmed = getattr(server.search_pubmed, "fn", server.search_pubmed)


def _articles() -> list[dict]:
    return [
        {
            "pmid": "39661433",
            "title": "Alpha study",
            "journal": "Journal of Alpha",
            "doi": "10.1000/alpha",
            "pmcid": "PMC12345",
        },
        {
            "pmid": "39661434",
            "title": "Beta study",
            "journal": "Journal of Beta",
        },
    ]


def _envelope(**overrides) -> dict:
    envelope = {
        "total_count": 4312,
        "returned": 2,
        "truncated": True,
        "query_translation": "foo[All Fields]",
        "articles": _articles(),
    }
    envelope.update(overrides)
    return envelope


@pytest.fixture
def stub_envelope(monkeypatch):
    def _install(**overrides) -> dict:
        envelope = _envelope(**overrides)

        async def fake_search_articles(**kwargs):
            return json.loads(json.dumps(envelope))

        monkeypatch.setattr(server.pubmed_client, "search_articles", fake_search_articles)
        return envelope

    return _install


def test_envelope_scalar_fields_survive_the_tool_boundary(stub_envelope):
    # Given
    stub_envelope()

    # When
    payload = json.loads(asyncio.run(search_pubmed(query="foo", max_results=2)))

    # Then
    assert payload["total_count"] == 4312
    assert payload["returned"] == 2
    assert payload["truncated"] is True
    assert payload["query_translation"] == "foo[All Fields]"


def test_articles_are_enriched_with_resource_uris_inside_the_envelope(stub_envelope):
    # Given
    stub_envelope()

    # When
    payload = json.loads(asyncio.run(search_pubmed(query="foo", max_results=2)))

    # Then
    first = payload["articles"][0]
    assert first["abstract_uri"] == "pubmed://39661433/abstract"
    assert first["full_text_uri"] == "pubmed://39661433/full_text"
    assert first["pubmed_url"] == "https://pubmed.ncbi.nlm.nih.gov/39661433/"
    assert first["doi_url"] == "https://doi.org/10.1000/alpha"
    assert first["pmc_url"] == "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12345/"
    assert "doi_url" not in payload["articles"][1]
    assert "pmc_url" not in payload["articles"][1]


def test_errors_and_warnings_pass_through_the_tool_boundary(stub_envelope):
    # Given
    stub_envelope(
        errors={"FieldNotFound": ["Titel"]},
        warnings={"QuotedPhraseNotFound": ['"gene editig"']},
    )

    # When
    payload = json.loads(asyncio.run(search_pubmed(query="foo", max_results=2)))

    # Then
    assert payload["errors"] == {"FieldNotFound": ["Titel"]}
    assert payload["warnings"] == {"QuotedPhraseNotFound": ['"gene editig"']}


def test_output_file_receives_the_full_envelope_not_a_bare_article_list(stub_envelope, tmp_path):
    # Given
    stub_envelope()
    out = tmp_path / "out.json"

    # When
    asyncio.run(search_pubmed(query="foo", max_results=2, output_file=str(out)))

    # Then
    written = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(written, dict)
    assert written["total_count"] == 4312
    assert written["returned"] == 2
    assert written["truncated"] is True
    assert written["query_translation"] == "foo[All Fields]"
    assert [a["pmid"] for a in written["articles"]] == ["39661433", "39661434"]


def test_file_output_summary_warns_that_written_results_are_truncated(stub_envelope, tmp_path):
    # Given
    stub_envelope()

    # When
    summary = asyncio.run(
        search_pubmed(query="foo", max_results=2, output_file=str(tmp_path / "out.json"))
    )

    # Then
    assert "4312" in summary
    assert re.search(r"truncat|incomplete|not a complete", summary, re.IGNORECASE)


def test_file_output_summary_does_not_claim_truncation_when_results_are_complete(
    stub_envelope, tmp_path
):
    # Given
    stub_envelope(total_count=2, returned=2, truncated=False)

    # When
    summary = asyncio.run(
        search_pubmed(query="foo", max_results=2, output_file=str(tmp_path / "out.json"))
    )

    # Then
    assert not re.search(r"truncat|incomplete|not a complete", summary, re.IGNORECASE)
    assert re.search(r"complete", summary, re.IGNORECASE)
