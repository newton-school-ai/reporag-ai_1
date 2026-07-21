"""Unit tests for src/reporag/ingestion/cloner.py.

Tests cover:
- FileInfo dataclass behaviour
- Local-path discovery
- Language / extension filtering
- Skip-directories logic
- Large-file filtering
- Error handling (bad URL, bad path, size limit)
- Shallow-clone flag plumbing (mocked)
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.reporag.ingestion.cloner import (
    LANGUAGE_EXTENSIONS,
    FileInfo,
    RepoCloner,
    _EXT_TO_LANGUAGE,
    _SKIP_DIRS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_repo(tmp_path: Path) -> Path:
    """Create a small fake repository tree for discovery tests."""
    (tmp_path / "main.py").write_text("print('hello')")
    (tmp_path / "utils.py").write_text("def helper(): pass")
    (tmp_path / "app.ts").write_text("const x = 1;")
    (tmp_path / "style.css").write_text("body { margin: 0; }")
    (tmp_path / "README.md").write_text("# Readme")

    # Nested source file
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "models.py").write_text("class Model: pass")
    (pkg / "__init__.py").write_text("")

    # Should be ignored – node_modules
    nm = tmp_path / "node_modules"
    nm.mkdir()
    (nm / "lodash.js").write_text("// lodash")

    # Should be ignored – __pycache__
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    (pycache / "main.cpython-311.pyc").write_bytes(b"\x00" * 10)

    return tmp_path


@pytest.fixture()
def cloner() -> RepoCloner:
    return RepoCloner()


# ---------------------------------------------------------------------------
# FileInfo tests
# ---------------------------------------------------------------------------


class TestFileInfo:
    def test_fields(self):
        fi = FileInfo(file_path="/a/b.py", language="python", size_bytes=42, relative_path="b.py")
        assert fi.file_path == "/a/b.py"
        assert fi.language == "python"
        assert fi.size_bytes == 42
        assert fi.relative_path == "b.py"

    def test_tuple_unpacking(self):
        fi = FileInfo(file_path="/a/b.py", language="python", size_bytes=100, relative_path="b.py")
        path, lang, size = fi
        assert path == "/a/b.py"
        assert lang == "python"
        assert size == 100

    def test_frozen(self):
        fi = FileInfo(file_path="/a/b.py", language="python", size_bytes=1, relative_path="b.py")
        with pytest.raises(Exception):
            fi.language = "rust"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Language / extension mapping tests
# ---------------------------------------------------------------------------


class TestLanguageExtensions:
    def test_python_extensions(self):
        assert ".py" in LANGUAGE_EXTENSIONS["python"]
        assert ".pyi" in LANGUAGE_EXTENSIONS["python"]

    def test_ext_to_language_lookup(self):
        assert _EXT_TO_LANGUAGE[".py"] == "python"
        assert _EXT_TO_LANGUAGE[".ts"] == "typescript"
        assert _EXT_TO_LANGUAGE[".go"] == "go"
        assert _EXT_TO_LANGUAGE[".rs"] == "rust"

    def test_unknown_extension_not_in_lookup(self):
        assert ".xyz" not in _EXT_TO_LANGUAGE
        assert ".pdf" not in _EXT_TO_LANGUAGE


# ---------------------------------------------------------------------------
# RepoCloner._is_local_path tests
# ---------------------------------------------------------------------------


class TestIsLocalPath:
    def test_https_url(self):
        assert RepoCloner._is_local_path("https://github.com/user/repo") is False

    def test_http_url(self):
        assert RepoCloner._is_local_path("http://example.com/repo.git") is False

    def test_ssh_url(self):
        assert RepoCloner._is_local_path("git@github.com:user/repo.git") is False

    def test_absolute_path(self, tmp_path):
        assert RepoCloner._is_local_path(str(tmp_path)) is True

    def test_relative_path_that_exists(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sub = tmp_path / "sub"
        sub.mkdir()
        assert RepoCloner._is_local_path("sub") is True


# ---------------------------------------------------------------------------
# Local discovery tests
# ---------------------------------------------------------------------------


class TestDiscoverLocal:
    def test_finds_python_files(self, cloner: RepoCloner, tmp_repo: Path):
        manifest = cloner.clone_and_discover(str(tmp_repo))
        langs = {fi.language for fi in manifest}
        assert "python" in langs

    def test_finds_typescript_files(self, cloner: RepoCloner, tmp_repo: Path):
        manifest = cloner.clone_and_discover(str(tmp_repo))
        langs = {fi.language for fi in manifest}
        assert "typescript" in langs

    def test_finds_css_files(self, cloner: RepoCloner, tmp_repo: Path):
        manifest = cloner.clone_and_discover(str(tmp_repo))
        langs = {fi.language for fi in manifest}
        assert "css" in langs

    def test_finds_markdown_files(self, cloner: RepoCloner, tmp_repo: Path):
        manifest = cloner.clone_and_discover(str(tmp_repo))
        langs = {fi.language for fi in manifest}
        assert "markdown" in langs

    def test_skips_node_modules(self, cloner: RepoCloner, tmp_repo: Path):
        manifest = cloner.clone_and_discover(str(tmp_repo))
        paths = [fi.relative_path for fi in manifest]
        assert not any("node_modules" in p for p in paths)

    def test_skips_pycache(self, cloner: RepoCloner, tmp_repo: Path):
        manifest = cloner.clone_and_discover(str(tmp_repo))
        paths = [fi.relative_path for fi in manifest]
        assert not any("__pycache__" in p for p in paths)

    def test_discovers_nested_files(self, cloner: RepoCloner, tmp_repo: Path):
        manifest = cloner.clone_and_discover(str(tmp_repo))
        rel_paths = [fi.relative_path for fi in manifest]
        assert any("models.py" in p for p in rel_paths)

    def test_manifest_is_sorted(self, cloner: RepoCloner, tmp_repo: Path):
        manifest = cloner.clone_and_discover(str(tmp_repo))
        rel_paths = [fi.relative_path for fi in manifest]
        assert rel_paths == sorted(rel_paths)

    def test_size_bytes_populated(self, cloner: RepoCloner, tmp_repo: Path):
        manifest = cloner.clone_and_discover(str(tmp_repo))
        for fi in manifest:
            assert fi.size_bytes >= 0

    def test_file_path_is_absolute(self, cloner: RepoCloner, tmp_repo: Path):
        manifest = cloner.clone_and_discover(str(tmp_repo))
        for fi in manifest:
            assert os.path.isabs(fi.file_path)

    def test_raises_for_missing_path(self, cloner: RepoCloner):
        with pytest.raises(ValueError, match="not a directory"):
            cloner.clone_and_discover("/nonexistent/path/that/does/not/exist")

    def test_raises_for_empty_source(self, cloner: RepoCloner):
        with pytest.raises(ValueError, match="non-empty"):
            cloner.clone_and_discover("")


# ---------------------------------------------------------------------------
# Large-file filter test
# ---------------------------------------------------------------------------


class TestLargeFileFilter:
    def test_large_files_are_skipped(self, tmp_path: Path):
        cloner = RepoCloner(max_file_size_bytes=10)
        big = tmp_path / "big.py"
        big.write_bytes(b"x" * 100)
        small = tmp_path / "small.py"
        small.write_bytes(b"y" * 5)

        manifest = cloner.clone_and_discover(str(tmp_path))
        rel_paths = [fi.relative_path for fi in manifest]
        assert "small.py" in rel_paths
        assert "big.py" not in rel_paths


# ---------------------------------------------------------------------------
# Custom extension filter test
# ---------------------------------------------------------------------------


class TestCustomExtensions:
    def test_only_configured_extensions_returned(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("x=1")
        (tmp_path / "app.ts").write_text("const y=1;")

        cloner = RepoCloner(extensions={"python": [".py"]})
        manifest = cloner.clone_and_discover(str(tmp_path))
        langs = {fi.language for fi in manifest}
        assert langs == {"python"}


# ---------------------------------------------------------------------------
# Remote clone tests (mocked)
# ---------------------------------------------------------------------------


class TestCloneRemote:
    """Test remote cloning behaviour with git.Repo.clone_from mocked."""

    def _make_mock_repo(self, tmp_path: Path) -> MagicMock:
        """Write a minimal file tree into tmp_path and return a mock Repo."""
        (tmp_path / "main.py").write_text("print('hi')")
        (tmp_path / "README.md").write_text("# hi")
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "utils.py").write_text("def f(): pass")
        return MagicMock()

    @patch("src.reporag.ingestion.cloner.tempfile.mkdtemp")
    @patch("src.reporag.ingestion.cloner.git.Repo.clone_from")
    def test_clone_calls_git(self, mock_clone, mock_mkdtemp, tmp_path):
        mock_mkdtemp.return_value = str(tmp_path)
        self._make_mock_repo(tmp_path)

        cloner = RepoCloner()
        manifest = cloner.clone_and_discover(
            "https://github.com/example/repo",
            branch="main",
            shallow=True,
        )

        mock_clone.assert_called_once()
        call_kwargs = mock_clone.call_args.kwargs
        assert call_kwargs.get("branch") == "main"
        assert call_kwargs.get("depth") == 1
        assert len(manifest) > 0

    @patch("src.reporag.ingestion.cloner.tempfile.mkdtemp")
    @patch("src.reporag.ingestion.cloner.git.Repo.clone_from")
    def test_no_shallow_omits_depth(self, mock_clone, mock_mkdtemp, tmp_path):
        mock_mkdtemp.return_value = str(tmp_path)
        self._make_mock_repo(tmp_path)

        cloner = RepoCloner()
        cloner.clone_and_discover(
            "https://github.com/example/repo",
            shallow=False,
        )

        call_kwargs = mock_clone.call_args.kwargs
        assert "depth" not in call_kwargs

    @patch("src.reporag.ingestion.cloner.tempfile.mkdtemp")
    @patch("src.reporag.ingestion.cloner.shutil.rmtree")
    def test_temp_dir_cleaned_on_error(self, mock_rmtree, mock_mkdtemp, tmp_path):
        """Temp dir must be cleaned up when cloning fails."""
        mock_mkdtemp.return_value = str(tmp_path)

        from git import GitCommandError as _GCE

        with patch(
            "src.reporag.ingestion.cloner.git.Repo.clone_from",
            side_effect=_GCE("clone", 128, "fatal: repository not found"),
        ):
            cloner = RepoCloner()
            with pytest.raises(RuntimeError, match="Failed to clone"):
                cloner.clone_and_discover("https://github.com/example/repo")

        mock_rmtree.assert_called_once_with(str(tmp_path), ignore_errors=True)

    @patch("src.reporag.ingestion.cloner.tempfile.mkdtemp")
    @patch("src.reporag.ingestion.cloner.git.Repo.clone_from")
    def test_size_limit_raises(self, mock_clone, mock_mkdtemp, tmp_path):
        mock_mkdtemp.return_value = str(tmp_path)
        # Fill tmp_path with enough data
        big = tmp_path / "big.py"
        big.write_bytes(b"x" * 1024)  # 1 KB

        cloner = RepoCloner(max_repo_size_mb=0.0009)  # < 1 KB limit
        with pytest.raises(RuntimeError, match="exceeds"):
            cloner.clone_and_discover("https://github.com/example/repo")

    @patch("src.reporag.ingestion.cloner.tempfile.mkdtemp")
    @patch("src.reporag.ingestion.cloner.git.Repo.clone_from")
    def test_no_branch_skips_branch_kwarg(self, mock_clone, mock_mkdtemp, tmp_path):
        mock_mkdtemp.return_value = str(tmp_path)
        self._make_mock_repo(tmp_path)

        cloner = RepoCloner()
        cloner.clone_and_discover("https://github.com/example/repo", branch=None)

        call_kwargs = mock_clone.call_args.kwargs
        assert "branch" not in call_kwargs
