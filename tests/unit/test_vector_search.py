from __future__ import annotations

import importlib.util
import sys
import types
from typing import Any
from unittest.mock import MagicMock

# 1. Safely import real NumPy if installed, otherwise create a complete MagicMock
if importlib.util.find_spec("numpy") is not None:
    import numpy  # noqa: F401
else:
    mock_np = MagicMock()
    mock_np.__version__ = "1.26.0"
    mock_np.int_ = int
    mock_np.uint = int
    mock_np.float_ = float
    mock_np.bool_ = bool
    mock_np.number = int | float
    mock_np.integer = int

    class FakeNdArray:
        def __init__(self, data: Any = None) -> None:
            self.data = data if data is not None else []
            self.shape = (
                (len(self.data),) if isinstance(self.data, list | tuple) else ()
            )

        def __repr__(self) -> str:
            return f"FakeNdArray({self.data})"

        def tolist(self) -> list:
            return (
                list(self.data) if isinstance(self.data, list | tuple) else [self.data]
            )

    mock_np.ndarray = FakeNdArray
    mock_np.array = lambda val, **kw: (
        FakeNdArray(val) if not isinstance(val, FakeNdArray) else val
    )
    mock_np.zeros = lambda shape, **kw: FakeNdArray(
        [0] * (shape[0] if isinstance(shape, tuple) else shape)
    )
    mock_np.isscalar = lambda obj: isinstance(
        obj, int | float | complex | bool | str | bytes
    )

    sys.modules["numpy"] = mock_np

# 2. Setup Mock PyTorch Module
if importlib.util.find_spec("torch") is None:
    mock_torch = types.ModuleType("torch")

    class FakeTensor:
        def __init__(self, data: Any = None) -> None:
            self.data = data if data is not None else []

        def cpu(self) -> FakeTensor:
            return self

        def detach(self) -> FakeTensor:
            return self

        def numpy(self) -> Any:
            return sys.modules["numpy"].array(self.data)

        def tolist(self) -> list:
            return (
                list(self.data) if isinstance(self.data, list | tuple) else [self.data]
            )

    mock_torch.Tensor = FakeTensor
    mock_torch.from_numpy = lambda data: FakeTensor(data)
    sys.modules["torch"] = mock_torch


# 3. Setup Mock Qdrant Client / Models
class PointStruct:
    def __init__(self, id: Any, vector: Any, payload: dict | None = None) -> None:
        self.id = id
        self.vector = vector
        self.payload = payload or {}


class ScoredPoint:
    def __init__(
        self,
        id: Any,
        score: float = 1.0,
        payload: dict | None = None,
        vector: Any = None,
    ) -> None:
        self.id = id
        self.score = score
        self.payload = payload or {}
        self.vector = vector


class MatchValue:
    def __init__(self, value: Any) -> None:
        self.value = value


class FieldCondition:
    def __init__(self, key: str, match: Any) -> None:
        self.key = key
        self.match = match


class Filter:
    def __init__(self, must: list | None = None) -> None:
        self.must = must or []
