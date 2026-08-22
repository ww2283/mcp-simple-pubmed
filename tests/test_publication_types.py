"""Offline tests for surfacing PublicationTypeList as a publication_types field."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from mcp_simple_pubmed.pubmed_client import PubMedClient


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def client() -> PubMedClient:
    return PubMedClient(email="test@example.com", tool="pytest-suite")


def _captured_article(pmid: str) -> ET.Element:
    root = ET.parse(FIXTURES / "efetch_publication_types.xml").getroot()
    for elem in root.findall("PubmedArticle"):
        pmid_elem = elem.find(".//PMID")
        if pmid_elem is not None and pmid_elem.text == pmid:
            return elem
    raise AssertionError(f"PMID {pmid} not in efetch_publication_types.xml")


def _article_without_publication_type_list() -> ET.Element:
    return ET.fromstring(
        "<PubmedArticle><MedlineCitation><PMID>37144894</PMID>"
        "<Article><ArticleTitle>A title with no publication types</ArticleTitle>"
        "<Abstract><AbstractText>An abstract.</AbstractText></Abstract>"
        "</Article></MedlineCitation></PubmedArticle>"
    )


def _article_with_text_empty_publication_type() -> ET.Element:
    return ET.fromstring(
        "<PubmedArticle><MedlineCitation><PMID>37144895</PMID>"
        "<Article><ArticleTitle>A title with a malformed publication type</ArticleTitle>"
        "<PublicationTypeList>"
        '<PublicationType UI="D016428">Journal Article</PublicationType>'
        '<PublicationType UI="D016454"/>'
        '<PublicationType UI="D016420">Comment</PublicationType>'
        "</PublicationTypeList>"
        "</Article></MedlineCitation></PubmedArticle>"
    )


def test_review_article_surfaces_both_journal_article_and_review(client):
    # Given
    elem = _captured_article("42625132")

    # When
    article = client._parse_article_element(elem, include_abstract=False)

    # Then
    assert article is not None
    assert article["publication_types"] == ["Journal Article", "Review"]


def test_primary_research_article_surfaces_journal_article_without_review(client):
    # Given
    elem = _captured_article("25554788")

    # When
    article = client._parse_article_element(elem, include_abstract=False)

    # Then
    assert article is not None
    assert article["publication_types"] == [
        "Journal Article",
        "Research Support, N.I.H., Extramural",
        "Research Support, Non-U.S. Gov't",
    ]
    assert "Review" not in article["publication_types"]


def test_article_without_publication_type_list_yields_present_empty_list(client):
    # Given
    elem = _article_without_publication_type_list()

    # When
    article = client._parse_article_element(elem, include_abstract=False)

    # Then
    assert article is not None
    assert "publication_types" in article
    assert article["publication_types"] == []


def test_publication_types_identical_regardless_of_include_abstract(client):
    # Given
    elem = _captured_article("42625132")

    # When
    with_abstract = client._parse_article_element(elem, include_abstract=True)
    without_abstract = client._parse_article_element(elem, include_abstract=False)

    # Then
    assert with_abstract is not None
    assert without_abstract is not None
    assert with_abstract["publication_types"] == ["Journal Article", "Review"]
    assert without_abstract["publication_types"] == with_abstract["publication_types"]


def test_retracted_article_surfaces_retracted_publication(client):
    # Given
    elem = _captured_article("42420159")

    # When
    article = client._parse_article_element(elem, include_abstract=False)

    # Then
    assert article is not None
    assert article["publication_types"] == [
        "Journal Article",
        "Retracted Publication",
        "Retraction Notice",
    ]


def test_text_empty_publication_type_is_skipped_keeping_well_formed_siblings(client):
    # Given
    elem = _article_with_text_empty_publication_type()

    # When
    article = client._parse_article_element(elem, include_abstract=False)

    # Then
    assert article is not None
    assert article["publication_types"] == ["Journal Article", "Comment"]
    assert all(isinstance(t, str) for t in article["publication_types"])
