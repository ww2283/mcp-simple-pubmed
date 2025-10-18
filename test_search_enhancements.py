#!/usr/bin/env python3
"""
Test script for search enhancements:
- include_abstracts parameter
- output_file parameter
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# Set env var before importing to avoid server initialization error
if "PUBMED_EMAIL" not in os.environ:
    os.environ["PUBMED_EMAIL"] = "test@example.com"

from mcp_simple_pubmed.pubmed_client import PubMedClient

async def test_search_without_abstracts():
    """Test search without abstracts (default behavior)."""
    print("=== Test 1: Search without abstracts ===")

    email = os.environ.get("PUBMED_EMAIL", "test@example.com")
    client = PubMedClient(email=email, tool="test-script")

    results = await client.search_articles(
        query="covid vaccine",
        max_results=2,
        include_abstracts=False
    )

    print(f"Found {len(results)} results")
    if results:
        first_result = results[0]
        print(f"First result keys: {list(first_result.keys())}")
        print(f"Has abstract: {'abstract' in first_result}")
        print(f"Has journal: {'journal' in first_result}")
        if 'journal' in first_result:
            print(f"Journal: {first_result['journal']}")
    print()

async def test_search_with_abstracts():
    """Test search with abstracts."""
    print("=== Test 2: Search with abstracts ===")

    email = os.environ.get("PUBMED_EMAIL", "test@example.com")
    client = PubMedClient(email=email, tool="test-script")

    results = await client.search_articles(
        query="covid vaccine",
        max_results=2,
        include_abstracts=True
    )

    print(f"Found {len(results)} results")
    if results:
        first_result = results[0]
        print(f"First result keys: {list(first_result.keys())}")
        print(f"Has abstract: {'abstract' in first_result}")
        print(f"Has journal: {'journal' in first_result}")
        if 'abstract' in first_result:
            abstract_preview = first_result['abstract'][:100] + "..." if len(first_result['abstract']) > 100 else first_result['abstract']
            print(f"Abstract preview: {abstract_preview}")
        if 'journal' in first_result:
            print(f"Journal: {first_result['journal']}")
    print()

async def test_file_output():
    """Test file output functionality."""
    print("=== Test 3: File output (simulated) ===")

    email = os.environ.get("PUBMED_EMAIL", "test@example.com")
    client = PubMedClient(email=email, tool="test-script")

    results = await client.search_articles(
        query="diabetes treatment",
        max_results=5,
        include_abstracts=False
    )

    # Simulate file output
    output_file = Path("/tmp/test_pubmed_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results written to: {output_file}")
    print(f"File size: {output_file.stat().st_size} bytes")

    # Read back and verify
    with open(output_file, 'r', encoding='utf-8') as f:
        loaded_results = json.load(f)

    print(f"Verified: {len(loaded_results)} results in file")

    # Print summary (like server would)
    print("\nTop results:")
    for i, article in enumerate(loaded_results[:3], 1):
        title = article.get('title', 'No title')
        pmid = article.get('pmid', 'Unknown')
        journal = article.get('journal', 'Unknown journal')
        print(f"{i}. {title}")
        print(f"   Journal: {journal} | PMID: {pmid}")

    # Cleanup
    output_file.unlink()
    print(f"\nTest file removed")
    print()

async def main():
    """Run all tests."""
    try:
        await test_search_without_abstracts()
        await test_search_with_abstracts()
        await test_file_output()
        print("✅ All tests completed successfully!")
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
