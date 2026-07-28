from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from src.reporag.ingestion.symbol_extractor import Symbol


class SymbolTable:
    """Central registry for extracted code symbols."""

    def __init__(self) -> None:
        self._symbols: dict[str, Symbol] = {}
        self._name_index: defaultdict[str, list[str]] = defaultdict(list)
        self._qualified_index: dict[str, str] = {}
        self._file_index: defaultdict[str, list[str]] = defaultdict(list)

    def _build_qualified_name(self, symbol: Symbol) -> str:
        module = Path(symbol.file_path).with_suffix("").as_posix().replace("/", ".")

        if symbol.parent_class:
            return f"{module}.{symbol.parent_class}.{symbol.name}"

        return f"{module}.{symbol.name}"

    def _build_symbol_id(self, symbol: Symbol) -> str:
        qualified_name = self._build_qualified_name(symbol)
        return f"{qualified_name}:{symbol.start_line}"

    def register(self, symbol: Symbol) -> str:
        symbol_id = self._build_symbol_id(symbol)

        if symbol_id in self._symbols:
            return symbol_id

        qualified_name = self._build_qualified_name(symbol)

        self._symbols[symbol_id] = symbol
        self._name_index[symbol.name].append(symbol_id)
        self._qualified_index[qualified_name] = symbol_id
        self._file_index[symbol.file_path].append(symbol_id)

        return symbol_id

    def lookup(self, name: str) -> list[Symbol]:
        symbol_ids = self._name_index.get(name, [])
        return [self._symbols[symbol_id] for symbol_id in symbol_ids]

    def lookup_qualified(self, qualified_name: str) -> Symbol | None:
        symbol_id = self._qualified_index.get(qualified_name)

        if symbol_id is None:
            return None

        return self._symbols[symbol_id]

    def lookup_regex(self, pattern: str) -> list[Symbol]:
        regex = re.compile(pattern)

        return [
            symbol for symbol in self._symbols.values() if regex.search(symbol.name)
        ]

    def lookup_file(self, file_path: str) -> list[Symbol]:
        symbol_ids = self._file_index.get(file_path, [])
        return [self._symbols[symbol_id] for symbol_id in symbol_ids]

    def to_json(self) -> str:
        data = {}

        for symbol_id, symbol in self._symbols.items():
            data[symbol_id] = {
                "qualified_name": self._build_qualified_name(symbol),
                "symbol": asdict(symbol),
            }

        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, data: str) -> SymbolTable:
        table = cls()

        for entry in json.loads(data).values():
            symbol = Symbol(**entry["symbol"])
            table.register(symbol)

        return table
