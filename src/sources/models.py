"""Normalized source-record schema shared by external connectors."""
from dataclasses import dataclass


@dataclass
class SourceDocument:
    source_id: str
    title: str
    text: str
    source_type: str
    url: str
    published_at: str = ""
    updated_at: str = ""
    license: str = ""

