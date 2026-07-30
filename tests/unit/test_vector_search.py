from __future__ import annotations

import sys
import types

if "numpy" not in sys.modules:
    numpy_module = types.ModuleType("numpy")

    class FakeNdArray:
        pass

    numpy_module.ndarray = FakeNdArray
    numpy_module.array = lambda val, **kw: val
    numpy_module.zeros = lambda shape, **kw: []
    sys.modules["numpy"] = numpy_module

if "torch" not in sys.modules:
    torch_module = types.ModuleType("torch")

    class FakeTensor:
        pass

    torch_module.Tensor = FakeTensor
    torch_module.no_grad = lambda: (lambda fn: fn)
    sys.modules["torch"] = torch_module

# 2. Setup Mock Qdrant Modules
qdrant_module = types.ModuleType("qdrant_client")
http_module = types.ModuleType("qdrant_client.http")
models_module = types.ModuleType("qdrant_client.http.models")


class QdrantClient:
    def __init__(self, *args, **kwargs) -> None:
        pass


class MatchValue:
    def __init__(self, value) -> None:
        self.value = value


class FieldCondition:
    def __init__(self, key, match) -> None:
        self.key = key
        self.match = match


class Filter:
    def __init__(self, must) -> None:
        self.must = must


models_module.MatchValue = MatchValue
models_module.FieldCondition = FieldCondition
models_module.Filter = Filter

qdrant_module.QdrantClient = QdrantClient
http_module.models = models_module

sys.modules["qdrant_client"] = qdrant_module
sys.modules["qdrant_client.http"] = http_module
sys.modules["qdrant_client.http.models"] = models_module
