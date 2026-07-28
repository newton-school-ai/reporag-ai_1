import pytest

from src.reporag.graph.symbol_table import SymbolTable
from src.reporag.ingestion.symbol_extractor import Symbol, SymbolType


@pytest.fixture
def table() -> SymbolTable:
    return SymbolTable()


@pytest.fixture
def function_symbol() -> Symbol:
    return Symbol(
        name="parse",
        symbol_type=SymbolType.FUNCTION,
        file_path="src/reporag/parser.py",
        start_line=10,
        end_line=20,
        signature="(source: str)",
    )


def test_register_and_lookup(
    table: SymbolTable,
    function_symbol: Symbol,
) -> None:
    symbol_id = table.register(function_symbol)

    assert symbol_id in table._symbols

    symbols = table.lookup("parse")

    assert len(symbols) == 1
    assert symbols[0] == function_symbol


def test_lookup_qualified(
    table: SymbolTable,
    function_symbol: Symbol,
) -> None:
    table.register(function_symbol)

    symbol = table.lookup_qualified("src.reporag.parser.parse")

    assert symbol == function_symbol


def test_lookup_regex(
    table: SymbolTable,
    function_symbol: Symbol,
) -> None:
    table.register(function_symbol)

    symbols = table.lookup_regex(r"par.*")

    assert len(symbols) == 1
    assert symbols[0] == function_symbol


def test_lookup_file(
    table: SymbolTable,
    function_symbol: Symbol,
) -> None:
    table.register(function_symbol)

    symbols = table.lookup_file("src/reporag/parser.py")

    assert len(symbols) == 1
    assert symbols[0] == function_symbol


def test_duplicate_registration(
    table: SymbolTable,
    function_symbol: Symbol,
) -> None:
    table.register(function_symbol)
    table.register(function_symbol)

    symbols = table.lookup("parse")

    assert len(symbols) == 1


def test_name_collision() -> None:
    table = SymbolTable()

    first = Symbol(
        name="parse",
        symbol_type=SymbolType.FUNCTION,
        file_path="src/reporag/parser.py",
        start_line=10,
        end_line=20,
    )

    second = Symbol(
        name="parse",
        symbol_type=SymbolType.FUNCTION,
        file_path="src/reporag/utils.py",
        start_line=5,
        end_line=15,
    )

    table.register(first)
    table.register(second)

    symbols = table.lookup("parse")

    assert len(symbols) == 2


def test_json_serialization(
    table: SymbolTable,
    function_symbol: Symbol,
) -> None:
    table.register(function_symbol)

    data = table.to_json()

    restored = SymbolTable.from_json(data)

    symbols = restored.lookup("parse")

    assert len(symbols) == 1
    assert symbols[0] == function_symbol
