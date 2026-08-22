"""
Client for interacting with PubMed/Entrez API.
"""
import time
import logging
import http.client
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Any
from Bio import Entrez

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pubmed-client")

VALID_SORT_ORDERS = ("relevance", "pub_date", "Author", "JournalName")

class PubMedClient:
    """Client for interacting with PubMed/Entrez API."""

    def __init__(self, email: str, tool: str, api_key: Optional[str] = None):
        """Initialize PubMed client with required credentials.

        Args:
            email: Valid email address for API access
            tool: Unique identifier for the tool
            api_key: Optional API key for higher rate limits
        """
        self.email = email
        self.tool = tool
        self.api_key = api_key
        
        # Configure Entrez
        Entrez.email = email
        Entrez.tool = tool
        if api_key:
            Entrez.api_key = api_key

    @staticmethod
    def _empty_envelope() -> Dict[str, Any]:
        return {
            "total_count": 0,
            "returned": 0,
            "truncated": False,
            "query_translation": "",
            "articles": [],
        }

    @staticmethod
    def _parse_esearch_envelope(root: ET.Element, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build the search result envelope from an esearch root and fetched articles."""
        count_elem = root.find('.//Count')
        try:
            total_count = int(count_elem.text) if count_elem is not None and count_elem.text else 0
        except ValueError:
            total_count = 0

        translation_elem = root.find('.//QueryTranslation')
        query_translation = (translation_elem.text or "") if translation_elem is not None else ""

        envelope: Dict[str, Any] = {
            "total_count": total_count,
            "returned": len(articles),
            "truncated": total_count > len(articles),
            "query_translation": query_translation,
        }

        for key, tag in (("errors", "ErrorList"), ("warnings", "WarningList")):
            container = root.find(f'.//{tag}')
            if container is not None:
                envelope[key] = [
                    {"type": child.tag, "value": child.text or ""} for child in container
                ]

        translation_set = root.find('.//TranslationSet')
        if translation_set is not None:
            mappings = []
            for translation in translation_set.findall('Translation'):
                from_elem = translation.find('From')
                to_elem = translation.find('To')
                mappings.append({
                    "from": ((from_elem.text or "") if from_elem is not None else "").strip(),
                    "to": ((to_elem.text or "") if to_elem is not None else "").strip(),
                })
            if mappings:
                envelope["term_mappings"] = mappings

        envelope["articles"] = articles
        return envelope

    async def search_articles(self, query: str, max_results: int = 10, include_abstracts: bool = False, sort: str = "relevance") -> Dict[str, Any]:
        """Search for articles matching the query, ordered by `sort`.

        Returns an envelope with total_count, returned, truncated, query_translation,
        optional errors/warnings, and articles.
        """
        if sort not in VALID_SORT_ORDERS:
            raise ValueError(
                f"Invalid sort order {sort!r}. Valid sort orders are: {', '.join(VALID_SORT_ORDERS)}"
            )

        try:
            logger.info(f"Searching PubMed with query: {query}")

            handle = Entrez.esearch(db="pubmed", term=query, retmax=str(max_results), sort=sort)
            if not handle:
                logger.error("Got None handle from esearch")
                return self._empty_envelope()

            if not isinstance(handle, http.client.HTTPResponse):
                return self._empty_envelope()

            logger.info("Got valid HTTP response from esearch")
            xml_content = handle.read()
            handle.close()

            root = ET.fromstring(xml_content)
            id_list = root.findall('.//Id')

            if not id_list:
                logger.info("No results found")
                return self._parse_esearch_envelope(root, [])

            pmids = [id_elem.text for id_elem in id_list if id_elem.text]
            logger.info(f"Found {len(pmids)} articles")

            articles = await self._fetch_articles_in_batches(pmids, include_abstracts)

            return self._parse_esearch_envelope(root, articles)

        except Exception as e:
            logger.exception(f"Error in search_articles: {str(e)}")
            raise

    async def _fetch_articles_in_batches(self, pmids: List[str], include_abstracts: bool = False, batch_size: int = 200) -> List[Dict[str, Any]]:
        """Fetch article details in batches, skipping any batch that errors."""
        all_articles = []

        # Split PMIDs into batches
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            logger.info(f"Fetching batch {i//batch_size + 1}: {len(batch_pmids)} articles")

            try:
                # Fetch batch of articles
                id_list = ",".join(batch_pmids)
                detail_handle = Entrez.efetch(db="pubmed", id=id_list, rettype="xml")

                if detail_handle and isinstance(detail_handle, http.client.HTTPResponse):
                    article_xml = detail_handle.read()
                    detail_handle.close()

                    # Parse batch XML - contains multiple PubmedArticle elements
                    root = ET.fromstring(article_xml)
                    article_elements = root.findall('.//PubmedArticle')

                    logger.info(f"Parsing {len(article_elements)} articles from batch")

                    # Process each article in the batch
                    for article_elem in article_elements:
                        article = self._parse_article_element(article_elem, include_abstracts)
                        if article:
                            all_articles.append(article)

                # Respect rate limits (3 requests/sec without API key, 10/sec with API key)
                if not self.api_key and i + batch_size < len(pmids):
                    time.sleep(0.34)  # ~3 requests per second

            except Exception as e:
                logger.exception(f"Error fetching batch starting at index {i}: {str(e)}")
                # Continue with next batch even if one fails
                continue

        logger.info(f"Successfully fetched {len(all_articles)} articles from {len(pmids)} PMIDs")
        return all_articles

    def _parse_article_element(self, article_elem: ET.Element, include_abstract: bool) -> Optional[Dict[str, Any]]:
        """Parse a single PubmedArticle element, returning None if it is unusable."""
        try:
            # Get PMID
            pmid_elem = article_elem.find('.//PMID')
            if pmid_elem is None or not pmid_elem.text:
                return None

            pmid = pmid_elem.text

            # Get basic article data
            article = {
                "pmid": pmid,
                "title": self._get_xml_text(article_elem, './/ArticleTitle') or "No title",
                "journal": self._get_xml_text(article_elem, './/Journal/Title') or "",
                "nlm_unique_id": self._get_xml_text(article_elem, './/NlmUniqueID') or "",
                "authors": [],
                "keywords": [],
                "mesh_terms": []
            }

            # Include abstract only if requested
            if include_abstract:
                article["abstract"] = self._get_full_abstract(article_elem) or "No abstract available"

            # Get authors
            author_list = article_elem.findall('.//Author')
            for author in author_list:
                last_name = self._get_xml_text(author, 'LastName') or ""
                fore_name = self._get_xml_text(author, 'ForeName') or ""
                if last_name or fore_name:
                    article["authors"].append(f"{last_name} {fore_name}".strip())

            # Get publication date
            pub_date = article_elem.find('.//PubDate')
            if pub_date is not None:
                year = self._get_xml_text(pub_date, 'Year')
                month = self._get_xml_text(pub_date, 'Month')
                day = self._get_xml_text(pub_date, 'Day')
                article["publication_date"] = {
                    "year": year,
                    "month": month,
                    "day": day
                }

            # Get DOI and PMCID if available
            pubmed_data = article_elem.find('.//PubmedData')
            if pubmed_data is not None:
                article_id_list_elem = pubmed_data.find('ArticleIdList')
                if article_id_list_elem is not None:
                    for article_id in article_id_list_elem:
                        id_type = article_id.get('IdType')
                        if id_type == 'doi':
                            article["doi"] = article_id.text
                        elif id_type == 'pmc':
                            article["pmcid"] = article_id.text

            # Get Keywords
            keyword_list = article_elem.findall('.//Keyword')
            for keyword in keyword_list:
                if keyword.text:
                    clean_keyword = keyword.text.strip().rstrip('.')
                    if clean_keyword:
                        article["keywords"].append(clean_keyword)

            # Get MeSH terms
            mesh_heading_list = article_elem.findall('.//MeshHeading')
            for mesh_heading in mesh_heading_list:
                descriptor = mesh_heading.find('DescriptorName')
                if descriptor is not None and descriptor.text:
                    mesh_term = {
                        "descriptor": descriptor.text,
                        "ui": descriptor.get('UI', ''),
                        "qualifiers": []
                    }

                    # Get qualifiers if present
                    qualifiers = mesh_heading.findall('QualifierName')
                    for qualifier in qualifiers:
                        if qualifier.text:
                            mesh_term["qualifiers"].append({
                                "name": qualifier.text,
                                "ui": qualifier.get('UI', '')
                            })

                    article["mesh_terms"].append(mesh_term)

            return article

        except Exception as e:
            logger.exception(f"Error parsing article element: {str(e)}")
            return None

    async def get_article_details(self, pmid: str, include_abstract: bool = True) -> Optional[Dict[str, Any]]:
        """Get details for a specific article by PMID.

        Args:
            pmid: PubMed ID of the article
            include_abstract: Whether to include abstract in result (default: True for backward compatibility)

        Returns:
            Dictionary with article metadata or None if not found
        """
        try:
            logger.info(f"Fetching details for PMID {pmid}")
            detail_handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml")

            if detail_handle and isinstance(detail_handle, http.client.HTTPResponse):
                article_xml = detail_handle.read()
                detail_handle.close()

                # Parse article XML
                root = ET.fromstring(article_xml)
                article_elem = root.find('.//PubmedArticle')

                if article_elem is not None:
                    return self._parse_article_element(article_elem, include_abstract)

            return None

        except Exception as e:
            logger.exception(f"Error getting article details for PMID {pmid}: {str(e)}")
            return None
            
    def _get_xml_text(self, elem: Optional[ET.Element], xpath: str) -> Optional[str]:
        """Helper method to safely get text from XML element."""
        if elem is None:
            return None
        found = elem.find(xpath)
        if found is None:
            return None
        return "".join(found.itertext()).strip() or None

    def _get_full_abstract(self, article_root: Optional[ET.Element]) -> Optional[str]:
        """Get complete abstract text, handling structured abstracts with multiple sections."""
        if article_root is None:
            return None

        abstract_texts = article_root.findall('.//Abstract/AbstractText')

        if not abstract_texts:
            return None

        # If there's only one AbstractText element, return it directly
        if len(abstract_texts) == 1:
            return "".join(abstract_texts[0].itertext()).strip()

        # For structured abstracts with multiple sections
        abstract_parts = []
        for text_elem in abstract_texts:
            label = text_elem.get('Label')
            text = "".join(text_elem.itertext()).strip()

            if label:
                # Format as "LABEL: text"
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)

        # Join all parts with double newline for readability
        return "\n\n".join(abstract_parts)