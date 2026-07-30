from __future__ import annotations

import sys
import types

import pytest

# 1. Setup Mock Heavy Dependencies for CI/Tests
if "numpy" not in sys.modules:
    numpy_module = types.ModuleType("numpy")

    class FakeNdArray:
        def __init__(self, data=None) -> None:
            self.data = data or []
            self.shape = (len(self.data),)
            self.dtype = "float32"

        def tolist(self) -> list:
            return self.data if isinstance(self.data, list) else [self.data]

        def __getitem__(self, idx):
            return self.data[idx] if isinstance(self.data, list) else self.data

    numpy_module.ndarray = FakeNdArray
    numpy_module.array = lambda val, **kw: FakeNdArray(val)
    numpy_module.zeros = lambda shape, **kw: FakeNdArray(
        [0.0] * (shape[0] if isinstance(shape, tuple) else shape)
    )
    sys.modules["numpy"] = numpy_module

if "torch" not in sys.modules:
    torch_module = types.ModuleType("torch")

    class FakeTensor:
        def __init__(self, data=None) -> None:
            self.data = data or []
            self.shape = (len(self.data),) if isinstance(self.data, list) else ()

        def tolist(self) -> list:
            return self.data if isinstance(self.data, list) else [self.data]

        def numpy(self):
            return sys.modules["numpy"].array(self.data)

        def cpu(self):
            return self

        def detach(self):
            return self

    class FakeNoGrad:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def __call__(self, fn):
            return fn

    torch_module.Tensor = FakeTensor
    torch_module.no_grad = FakeNoGrad
    sys.modules["torch"] = torch_module

if "transformers" not in sys.modules:
    transformers_module = types.ModuleType("transformers")

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    transformers_module.AutoModel = FakeAutoModel
    transformers_module.AutoTokenizer = FakeAutoTokenizer
    sys.modules["transformers"] = transformers_module

if "sentence_transformers" not in sys.modules:
    st_module = types.ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_name_or_path: str | None = None, **kwargs) -> None:
            self.model_name_or_path = model_name_or_path

        def encode(self, sentences, **kwargs):
            return [[1.0] * 384]

    st_module.SentenceTransformer = FakeSentenceTransformer
    sys.modules["sentence_transformers"] = st_module

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

# 3. Import application code AFTER mocking dependencies in sys.modules
from src.reporag.retrieval.vector_search import (  # noqa: E402
    RetrievalResult,
    VectorSearcher,
)


# 4. Test Doubles & Mocks
class FakeVector:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tolist(self) -> list[float]:
        return self.values


class FakeMatrix:
    def __init__(self, row: list[float]) -> None:
        self.row = row

    def __getitem__(self, index: int) -> FakeVector:
        assert index == 0
        return FakeVector(self.row)


class FakeCodeEmbedder:
    def embed_batch(self, texts: list[str]) -> FakeMatrix:
        return FakeMatrix([1.0] * 768)


class FakeDocEmbedder:
    def embed_batch(self, docs: list[dict[str, str]]) -> list[dict]:
        return [
            {
                "embedding": FakeVector([1.0] * 384),
                "symbol_id": docs[0]["symbol_id"],
            }
        ]


# 5. Pytest Fixtures & Tests
@pytest.fixture
def searcher() -> VectorSearcher:
    return VectorSearcher(
        code_embedder=FakeCodeEmbedder(),
        doc_embedder=FakeDocEmbedder(),
    )


def test_build_filter(searcher: VectorSearcher) -> None:
    search_filter = searcher._build_filter(
        language="python",
        file_path="src/reporag/parser.py",
        symbol_type="function",
    )

    assert search_filter is not None
    assert len(search_filter.must) == 3

    keys = {condition.key for condition in search_filter.must}

    assert keys == {"language", "file", "symbol"}


def test_search_merges_and_ranks_results(searcher: VectorSearcher) -> None:
    code_result = RetrievalResult(
        score=0.80,
        file_path="src/reporag/parser.py",
        start_line=10,
        end_line=20,
        symbol="parse",
        chunk_text="def parse(source): ...",
        language="python",
        source="code",
    )

    doc_result = RetrievalResult(
        score=0.95,
        file_path="src/reporag/parser.py",
        start_line=10,
        end_line=20,
        symbol="parse",
        chunk_text="Parse source code into an AST.",
        language="python",
        source="doc",
    )

    other_result = RetrievalResult(
        score=0.70,
        file_path="src/reporag/utils.py",
        start_line=5,
        end_line=15,
        symbol="helper",
        chunk_text="def helper(): ...",
        language="python",
        source="code",
    )

    def fake_search(vector_name, query_vector, top_k, search_filter):
        if vector_name == "code":
            return [code_result, other_result]
        return [doc_result]

    searcher._search_named_vector = fake_search

    results = searcher.search("parse function", top_k=5)

    assert len(results) == 2
    assert results[0].score == pytest.approx(0.95)
    assert results[0].symbol == "parse"
    assert results[1].symbol == "helper"
