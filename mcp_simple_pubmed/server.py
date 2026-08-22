"""
MCP server implementation for PubMed integration using FastMCP SDK.
"""
import os
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

from fastmcp import FastMCP
from mcp_simple_pubmed.pubmed_client import PubMedClient, DEFAULT_ABSTRACT_CHARS
from mcp_simple_pubmed.fulltext_client import FullTextClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pubmed-server")

MAX_RESULTS_LIMIT = 1000
DEFAULT_FULLTEXT_MAX_CHARS = 50000

# Initialize FastMCP app
app = FastMCP("pubmed-server")

def configure_clients() -> Tuple[PubMedClient, FullTextClient]:
    """Configure PubMed and full text clients with environment settings."""
    email = os.environ.get("PUBMED_EMAIL")
    if not email:
        raise ValueError("PUBMED_EMAIL environment variable is required")
        
    tool = os.environ.get("PUBMED_TOOL", "mcp-simple-pubmed")
    api_key = os.environ.get("PUBMED_API_KEY")

    pubmed_client = PubMedClient(email=email, tool=tool, api_key=api_key)
    fulltext_client = FullTextClient(email=email, tool=tool, api_key=api_key)
    
    return pubmed_client, fulltext_client

# Initialize the clients
pubmed_client, fulltext_client = configure_clients()


def _bound_fulltext(text: str, max_chars: int, offset: int) -> str:
    """Return at most max_chars of text from offset, with a paging trailer if more remains."""
    clamped_max_chars = max(1, max_chars)
    clamped_offset = max(0, offset)
    if clamped_max_chars != max_chars or clamped_offset != offset:
        logger.warning(
            f"_bound_fulltext arguments out of range (max_chars={max_chars}, offset={offset}), "
            f"clamped to max_chars={clamped_max_chars}, offset={clamped_offset}"
        )
    max_chars = clamped_max_chars
    offset = clamped_offset

    total = len(text)
    if offset >= total:
        return (
            f"[No content at offset {offset}: the article is only {total} "
            f"characters long. Use a smaller offset.]"
        )

    end = offset + max_chars
    chunk = text[offset:end]
    if end >= total:
        return chunk

    return (
        f"{chunk}\n\n[TRUNCATED: returned characters {offset}-{end} of {total}. "
        f"Call again with offset={end} to continue.]"
    )

@app.tool(
    annotations={
        "title": "Search articles about medical and life sciences research available on PubMed.",
        "readOnlyHint": True,
        "openWorldHint": True  # Calls external PubMed API
    }
)
async def search_pubmed(query: str, max_results: int = 10, include_abstracts: bool = True, output_file: Optional[str] = None, sort: str = "relevance", abstract_chars: int = DEFAULT_ABSTRACT_CHARS, offset: int = 0) -> str:
    """Search PubMed for medical and life sciences research articles.

    Results default to relevance ("Best Match") order; use the sort parameter to change it.
    Best Match degrades sharply on OR-heavy boolean queries, so anchor the core concept on
    [MeSH Terms] and prefer AND-ing specific terms over OR-ing generic ones.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (clamped to 1..1000, default: 10)
        include_abstracts: Include abstracts in results (default: True). Set to False to omit them entirely.
        abstract_chars: Abstract character budget (default: 300); 0 returns the full abstract.
        output_file: Optional file path to save results; results are written wherever this path points
                     (parent directories are created) and only a summary is returned.
        sort: Result ordering - one of relevance, pub_date, Author, JournalName (default: relevance)
        offset: Index of the first hit to return (default: 0); page with has_more to go
                past max_results, and keep sort constant across pages.

    You can use these search features:
    - Simple keyword search: "covid vaccine"
    - Field-specific search:
      - Title search: [Title]
      - Author search: [Author]
      - MeSH terms: [MeSH Terms]
      - Journal: [Journal]
    - Date ranges: Add year or date range like "2020:2024[Date - Publication]"
    - Combine terms with AND, OR, NOT
    - Use quotation marks for exact phrases
    - Quoting a term or giving it a field tag (e.g. tau[tiab]) suppresses automatic term mapping;
      term_mappings in the result shows which terms PubMed expanded on your behalf

    Examples:
    - "covid vaccine" - basic search
    - "breast cancer"[Title] AND "2023"[Date - Publication]
    - "Smith J"[Author] AND "diabetes"
    - "RNA"[MeSH Terms] AND "therapy"

    The search returns an envelope:
    - total_count / returned / truncated: how many matched vs. how many are here
    - offset / has_more: where this page started and whether further hits remain
    - query_translation: how PubMed actually interpreted the query
    - term_mappings: from/to pairs, present only when PubMed remapped a term automatically
    - errors / warnings: present only when PubMed reports them (e.g. an unknown field tag)
    - articles: title, authors, journal, publication details, links, DOI, keywords, MeSH terms,
      publication_types

    Note: Use quotes around multi-word terms for best results.
          PubMed indexes a review as both "Journal Article" and "Review", so "Journal Article"[PT]
          does not exclude reviews - filter on publication_types instead. It also carries
          "Retracted Publication".
          Abstracts are included but capped at abstract_chars; abstract_truncated marks the cut ones.
    """
    try:
        clamped_max_results = min(MAX_RESULTS_LIMIT, max(1, max_results))
        if clamped_max_results != max_results:
            logger.warning(f"max_results {max_results} out of range, clamped to {clamped_max_results}")
        max_results = clamped_max_results

        logger.info(f"Processing search with query: {query}, max_results: {max_results}, include_abstracts: {include_abstracts}")

        # Perform the search
        results = await pubmed_client.search_articles(
            query=query,
            max_results=max_results,
            include_abstracts=include_abstracts,
            sort=sort,
            abstract_chars=abstract_chars,
            offset=offset
        )
        
        # Create resource URIs for articles
        articles_with_resources = []
        for article in results["articles"]:
            pmid = article["pmid"]
            # Add original URIs
            article["abstract_uri"] = f"pubmed://{pmid}/abstract"
            article["full_text_uri"] = f"pubmed://{pmid}/full_text"
            
            # Add DOI URL if DOI exists
            if "doi" in article:
                article["doi_url"] = f"https://doi.org/{article['doi']}"
                
            # Add PubMed URLs
            article["pubmed_url"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            # Add PMC URL only if PMCID is available
            if "pmcid" in article:
                article["pmc_url"] = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{article['pmcid']}/"
            
            articles_with_resources.append(article)

        envelope = {**results, "articles": articles_with_resources}
        total_count = envelope.get("total_count", len(articles_with_resources))
        truncated = envelope.get("truncated", False)

        # Handle file output if requested
        if output_file:
            try:
                # Resolve to absolute path
                file_path = Path(output_file).expanduser().resolve()

                try:
                    # Create parent directories if they don't exist
                    file_path.parent.mkdir(parents=True, exist_ok=True)

                    # Write results to file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(envelope, f, indent=2, ensure_ascii=False)

                    logger.info(f"Results written to file: {file_path}")

                    truncation_note = (
                        f"TRUNCATED: only {len(articles_with_resources)} of {total_count} matching records were written; "
                        "this file is not a complete view of the literature."
                        if truncated
                        else f"Complete: all {total_count} matching records were written."
                    )

                    # Create summary response
                    summary_lines = [
                        f"Search completed successfully. {total_count} total matches, {len(articles_with_resources)} returned.",
                        truncation_note,
                        f"Results written to: {file_path}",
                        "",
                        "Top results:"
                    ]

                    # Add top 3 results to summary
                    for i, article in enumerate(articles_with_resources[:3], 1):
                        title = article.get('title', 'No title')
                        pmid = article.get('pmid', 'Unknown')
                        journal = article.get('journal', 'Unknown journal')
                        summary_lines.append(f"{i}. {title}")
                        summary_lines.append(f"   Journal: {journal} | PMID: {pmid}")
                        summary_lines.append("")

                    if len(articles_with_resources) > 3:
                        summary_lines.append(f"... and {len(articles_with_resources) - 3} more results in the file.")

                    return "\n".join(summary_lines)

                except PermissionError:
                    raise ValueError(f"Permission denied: Cannot write to {file_path}")
                except OSError as e:
                    raise ValueError(f"Cannot write to file {file_path}: {str(e)}")

            except Exception as e:
                logger.error(f"Error writing to file: {str(e)}")
                raise ValueError(f"Error writing results to file: {str(e)}")

        # Format the response for direct return
        formatted_results = json.dumps(envelope, indent=2)
        logger.info(f"Search completed successfully, found {total_count} results")

        return formatted_results
        
    except Exception as e:
        logger.exception("Error in search_pubmed")
        raise ValueError(f"Error processing search request: {str(e)}")

@app.tool(
    annotations={
        "title": "Get a paper's full text",
        "readOnlyHint": True,
        "openWorldHint": True  # Calls external PubMed API
    }
)
async def get_paper_fulltext(pmid: str, max_chars: int = DEFAULT_FULLTEXT_MAX_CHARS, offset: int = 0) -> str:
    """Get full text from PubMed Central, or a message naming where else to find it.

    Returns at most max_chars characters starting at offset; a longer article comes
    back with a trailer stating the offset to call again with.
    """
    try:
        logger.info(f"Attempting to get full text for PMID: {pmid}")

        # First check PMC availability
        available, pmc_id = await fulltext_client.check_full_text_availability(pmid)
        
        if available:
            full_text = await fulltext_client.get_full_text(pmid)
            if full_text:
                logger.info(f"Successfully retrieved full text from PMC for PMID {pmid}")
                return _bound_fulltext(full_text, max_chars, offset)

        # Get article details to provide alternative locations
        article = await pubmed_client.get_article_details(pmid)
        
        message = "Full text is not available in PubMed Central.\n\n"
        message += "The article may be available at these locations:\n"
        message += f"- PubMed page: https://pubmed.ncbi.nlm.nih.gov/{pmid}/\n"
        
        if article and "doi" in article:
            message += f"- Publisher's site (via DOI): https://doi.org/{article['doi']}\n"
            
        logger.info(f"Full text not available in PMC for PMID {pmid}, provided alternative locations")
        return message
        
    except Exception as e:
        logger.exception("Error in get_paper_fulltext")
        raise ValueError(f"Error retrieving full text: {str(e)}")


@app.resource("pubmed://{pmid}/{resource_type}")
async def read_pubmed_resource(pmid: str, resource_type: str) -> str:
    """
    Reads different types of content for a given PubMed ID (PMID).
    This can be the article's abstract or its full text.

    You can find PMIDs by searching for articles using the search_pubmed tool.

    Example usage:
    read_pubmed_resource(pmid="39661433", resource_type="abstract")
    read_pubmed_resource(pmid="39661433", resource_type="full_text")
    """
    logger.info(f"Reading resource for pmid={pmid}, type={resource_type}")
    try:
        if resource_type == "abstract":
            article = await pubmed_client.get_article_details(pmid)
            return json.dumps(article, indent=2)

        elif resource_type == "full_text":
            available, pmc_id = await fulltext_client.check_full_text_availability(pmid)
            if available:
                full_text = await fulltext_client.get_full_text(pmid)
                if full_text:
                    return _bound_fulltext(full_text, DEFAULT_FULLTEXT_MAX_CHARS, 0)
            
            # If not available, provide the same helpful message as the tool
            article = await pubmed_client.get_article_details(pmid)
            message = "Full text is not available in PubMed Central.\n\n"
            message += "The article may be available at these locations:\n"
            message += f"- PubMed page: https://pubmed.ncbi.nlm.nih.gov/{pmid}/\n"
            if article and "doi" in article:
                message += f"- Publisher's site (via DOI): https://doi.org/{article['doi']}\n"
            return message

        else:
            raise ValueError(f"Invalid resource type requested: {resource_type}")

    except Exception as e:
        logger.exception(f"Error reading resource pmid={pmid}, type={resource_type}")
        raise ValueError(f"Error reading resource: {str(e)}")


def main():
    """Run the MCP server."""
    app.run()

if __name__ == "__main__":
    main() 