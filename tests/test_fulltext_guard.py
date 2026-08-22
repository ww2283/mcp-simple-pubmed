"""Offline tests for the length guard and offset paging on get_paper_fulltext."""

import asyncio
import logging
import re

import pytest

from mcp_simple_pubmed import server


# fastmcp 2.x wraps the tool in a FunctionTool exposing .fn; 3.x leaves the function bare.
get_paper_fulltext = getattr(server.get_paper_fulltext, "fn", server.get_paper_fulltext)

LONG_TEXT = "x" * 139453
SHORT_TEXT = "y" * 1200

TRUNCATION_SIGNAL = re.compile(r"TRUNCAT", re.IGNORECASE)


def trailer_of(result: str, body_length: int) -> str:
    return result[body_length:]


@pytest.fixture
def pmc_text(monkeypatch):
    def _install(text: str) -> None:
        async def fake_check(pmid):
            return True, f"PMC{pmid}"

        async def fake_get_full_text(pmid):
            return text

        monkeypatch.setattr(server.fulltext_client, "check_full_text_availability", fake_check)
        monkeypatch.setattr(server.fulltext_client, "get_full_text", fake_get_full_text)

    return _install


@pytest.fixture
def not_in_pmc(monkeypatch):
    async def fake_check(pmid):
        return False, None

    async def fake_get_article_details(pmid, include_abstract=True):
        return {"pmid": pmid, "doi": "10.1000/example.doi"}

    monkeypatch.setattr(server.fulltext_client, "check_full_text_availability", fake_check)
    monkeypatch.setattr(server.pubmed_client, "get_article_details", fake_get_article_details)


def test_default_fulltext_max_chars_constant_is_50000():
    # Then
    assert server.DEFAULT_FULLTEXT_MAX_CHARS == 50000


def test_long_fulltext_is_bounded_to_max_chars_and_carries_trailer(pmc_text):
    # Given
    pmc_text(LONG_TEXT)

    # When
    result = asyncio.run(get_paper_fulltext(pmid="39661433"))

    # Then
    assert result.startswith(LONG_TEXT[:50000])
    trailer = trailer_of(result, 50000)
    assert TRUNCATION_SIGNAL.search(trailer)
    assert "139453" in trailer
    assert "50000" in trailer


def test_short_fulltext_is_returned_whole_without_trailer(pmc_text):
    # Given
    pmc_text(SHORT_TEXT)

    # When
    result = asyncio.run(get_paper_fulltext(pmid="39661433"))

    # Then
    assert result == SHORT_TEXT
    assert not TRUNCATION_SIGNAL.search(result)


def test_offset_returns_slice_starting_at_offset_with_trailer_naming_next_offset(pmc_text):
    """Uses a non-default max_chars so a hardcoded 50000 cannot satisfy it."""
    # Given
    pmc_text(LONG_TEXT)

    # When
    result = asyncio.run(get_paper_fulltext(pmid="39661433", max_chars=30000, offset=50000))

    # Then
    assert result.startswith(LONG_TEXT[50000:80000])
    trailer = trailer_of(result, 30000)
    assert TRUNCATION_SIGNAL.search(trailer)
    assert "139453" in trailer
    assert "80000" in trailer


def test_final_page_reaching_end_of_text_has_no_trailer(pmc_text):
    # Given
    pmc_text(LONG_TEXT)

    # When
    result = asyncio.run(get_paper_fulltext(pmid="39661433", max_chars=50000, offset=100000))

    # Then
    assert result == LONG_TEXT[100000:]
    assert not TRUNCATION_SIGNAL.search(result)


def test_offset_beyond_end_returns_explanatory_message_naming_total_length(pmc_text):
    # Given
    pmc_text(LONG_TEXT)

    # When
    result = asyncio.run(get_paper_fulltext(pmid="39661433", max_chars=50000, offset=200000))

    # Then
    assert isinstance(result, str)
    assert "x" not in result
    assert "139453" in result
    assert not TRUNCATION_SIGNAL.search(result)


def test_not_in_pmc_message_is_returned_intact_and_untruncated(not_in_pmc):
    # When
    result = asyncio.run(get_paper_fulltext(pmid="39661433"))

    # Then
    assert "Full text is not available in PubMed Central." in result
    assert "https://pubmed.ncbi.nlm.nih.gov/39661433/" in result
    assert "https://doi.org/10.1000/example.doi" in result
    assert not TRUNCATION_SIGNAL.search(result)


TRAILER_START = "\n\n[TRUNCATED:"
NEXT_OFFSET = re.compile(r"offset=(-?\d+)")


def split_bounded(result: str) -> tuple[str, str]:
    body, marker, rest = result.partition(TRAILER_START)
    return body, marker + rest


def next_offset_of(trailer: str) -> int:
    match = NEXT_OFFSET.search(trailer)
    assert match, f"no next offset in trailer: {trailer!r}"
    return int(match.group(1))


def test_max_chars_zero_still_returns_content_and_advances_the_next_offset():
    # When
    result = server._bound_fulltext(SHORT_TEXT, 0, 0)

    # Then
    body, trailer = split_bounded(result)
    assert len(body) >= 1
    assert body == SHORT_TEXT[: len(body)]
    assert next_offset_of(trailer) > 0


def test_negative_max_chars_is_clamped_to_one_like_zero():
    # When
    from_zero = server._bound_fulltext(SHORT_TEXT, 0, 0)
    from_negative = server._bound_fulltext(SHORT_TEXT, -25, 0)

    # Then
    assert from_negative == from_zero
    assert next_offset_of(split_bounded(from_negative)[1]) > 0


def test_negative_offset_reads_from_the_start_and_reports_no_negative_range():
    # When
    result = server._bound_fulltext("abcdefghij", 5, -3)

    # Then
    body, trailer = split_bounded(result)
    assert body == "abcde"
    assert next_offset_of(trailer) == 5
    assert "-3" not in result
    assert re.search(r"characters\s+-", trailer) is None


def test_clamping_out_of_range_arguments_logs_a_warning(caplog):
    # When
    with caplog.at_level(logging.WARNING, logger="pubmed-server"):
        server._bound_fulltext(SHORT_TEXT, 0, -10)

    # Then
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_in_range_arguments_are_unchanged_and_log_no_warning(pmc_text, caplog):
    # Given
    pmc_text(LONG_TEXT)

    # When
    with caplog.at_level(logging.WARNING, logger="pubmed-server"):
        result = asyncio.run(get_paper_fulltext(pmid="39661433", max_chars=30000, offset=0))

    # Then
    body, trailer = split_bounded(result)
    assert body == LONG_TEXT[:30000]
    assert next_offset_of(trailer) == 30000
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]
