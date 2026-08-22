"""Offline parsing tests for titles/abstracts carrying inline XML markup."""

import xml.etree.ElementTree as ET

import pytest

from mcp_simple_pubmed.pubmed_client import PubMedClient


@pytest.fixture
def client() -> PubMedClient:
    return PubMedClient(email="test@example.com", tool="pytest-suite")


def _article(title_xml: str, abstract_xml: str = "") -> ET.Element:
    return ET.fromstring(
        "<PubmedArticle><MedlineCitation><PMID>37144894</PMID>"
        f"<Article><ArticleTitle>{title_xml}</ArticleTitle>"
        f"{abstract_xml}</Article></MedlineCitation></PubmedArticle>"
    )


def test_get_xml_text_keeps_text_inside_and_after_inline_child(client):
    # Given
    elem = _article(
        "<b>Benzothiazole Substitution Analogs of Rhodacyanine "
        "Hsp70 Inhibitors Modulate Tau Accumulation</b>."
    )

    # When
    title = client._get_xml_text(elem, ".//ArticleTitle")

    # Then
    assert title == (
        "Benzothiazole Substitution Analogs of Rhodacyanine "
        "Hsp70 Inhibitors Modulate Tau Accumulation."
    )


def test_parse_article_element_does_not_fall_back_to_no_title_for_marked_up_title(client):
    # Given
    elem = _article("Effects of <i>BRCA1</i> on <sup>18</sup>F uptake")

    # When
    article = client._parse_article_element(elem, include_abstract=False)

    # Then
    assert article is not None
    assert article["title"] == "Effects of BRCA1 on 18F uptake"


def test_single_section_abstract_keeps_inline_child_text(client):
    # Given
    elem = _article(
        "Plain title",
        "<Abstract><AbstractText>Inhibition of <i>Hsp70</i> "
        "reduced tau by <b>60</b>%.</AbstractText></Abstract>",
    )

    # When
    abstract = client._get_full_abstract(elem)

    # Then
    assert abstract == "Inhibition of Hsp70 reduced tau by 60%."


def test_structured_abstract_keeps_inline_child_text_in_each_section(client):
    # Given
    elem = _article(
        "Plain title",
        "<Abstract>"
        '<AbstractText Label="BACKGROUND">Tau aggregates in <i>AD</i> brains.</AbstractText>'
        '<AbstractText Label="RESULTS">IC<sub>50</sub> was 2 nM.</AbstractText>'
        "</Abstract>",
    )

    # When
    abstract = client._get_full_abstract(elem)

    # Then
    assert abstract == (
        "BACKGROUND: Tau aggregates in AD brains.\n\nRESULTS: IC50 was 2 nM."
    )


def test_plain_markup_free_title_and_abstract_are_unchanged(client):
    # Given
    elem = _article(
        "A plain title without markup",
        "<Abstract><AbstractText>A plain abstract without markup.</AbstractText></Abstract>",
    )

    # When
    article = client._parse_article_element(elem, include_abstract=True)

    # Then
    assert article is not None
    assert article["title"] == "A plain title without markup"
    assert article["abstract"] == "A plain abstract without markup."
