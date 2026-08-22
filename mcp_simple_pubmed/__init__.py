"""
MCP server providing access to PubMed articles through Entrez API.
"""
from importlib.metadata import PackageNotFoundError, version

from . import server

def main():
    """Main entry point for the package."""
    server.main()

try:
    __version__ = version("mcp-simple-pubmed")
except PackageNotFoundError:
    __version__ = "unknown"