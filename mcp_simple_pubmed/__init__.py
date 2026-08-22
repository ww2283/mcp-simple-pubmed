"""
MCP server providing access to PubMed articles through Entrez API.
"""
from . import server

def main():
    """Main entry point for the package."""
    server.main()

__version__ = "0.1.0"