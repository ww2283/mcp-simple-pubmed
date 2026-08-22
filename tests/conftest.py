import os

# Importing the package runs configure_clients() at module scope, which requires this.
os.environ.setdefault("PUBMED_EMAIL", "test@example.com")
