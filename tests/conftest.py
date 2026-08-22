import os

# Importing the package runs configure_clients() at module scope, which requires this.
os.environ.setdefault("PUBMED_EMAIL", "test@example.com")

import pytest

from mcp_simple_pubmed import pubmed_client as pubmed_client_module


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Fail loudly instead of reaching NCBI when a test forgets to stub Entrez."""

    def _blocked(*args, **kwargs):
        raise RuntimeError("test attempted a live Entrez call")

    for name in ("esearch", "efetch"):
        monkeypatch.setattr(pubmed_client_module.Entrez, name, _blocked)
