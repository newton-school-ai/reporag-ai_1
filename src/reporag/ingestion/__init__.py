"""RepoRAG ingestion package.

Exposes the repository cloner, file-discovery service, and AST parser.
"""

from .cloner import (
    LANGUAGE_EXTENSIONS,
    CloneResult,
    FileInfo,
    RepoCloner,
)
from .parser import (
    ASTParser,
    NodeInfo,
    ParseResult,
    register_language,
    supported_languages,
)
from .symbol_extractor import Symbol, SymbolExtractor, SymbolType

__all__ = [
    # cloner
    "LANGUAGE_EXTENSIONS",
    "CloneResult",
    "FileInfo",
    "RepoCloner",
    # parser
    "ASTParser",
    "NodeInfo",
    "ParseResult",
    "register_language",
    "supported_languages",
    # symbol extractor
    "Symbol",
    "SymbolExtractor",
    "SymbolType",
]
