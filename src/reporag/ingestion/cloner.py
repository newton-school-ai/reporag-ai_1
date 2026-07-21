"""Git repository cloner and file discovery service.

Clones a Git repository (via HTTPS or local path) to a temp directory and
discovers all parseable source files, returning a manifest of FileInfo
named-tuples containing (file_path, language, size_bytes).

Usage:
    from src.reporag.ingestion.cloner import RepoCloner

    cloner = RepoCloner()
    manifest = cloner.clone_and_discover(
        "https://github.com/pallets/click",
        branch="main",
        shallow=True,
    )
    print(f"Found {len(manifest)} files")
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import git
from git import GitCommandError, InvalidGitRepositoryError, NoSuchPathError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language → file-extension mapping
# ---------------------------------------------------------------------------

#: Maps a canonical language name to its recognised file extensions.
LANGUAGE_EXTENSIONS: dict[str, list[str]] = {
    "python": [".py", ".pyw", ".pyi"],
    "javascript": [".js", ".mjs", ".cjs"],
    "typescript": [".ts", ".tsx", ".mts", ".cts"],
    "java": [".java"],
    "kotlin": [".kt", ".kts"],
    "go": [".go"],
    "rust": [".rs"],
    "cpp": [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hh", ".hxx"],
    "c": [".c", ".h"],
    "csharp": [".cs"],
    "ruby": [".rb", ".rake"],
    "php": [".php"],
    "swift": [".swift"],
    "scala": [".scala"],
    "shell": [".sh", ".bash", ".zsh"],
    "markdown": [".md", ".mdx"],
    "yaml": [".yml", ".yaml"],
    "json": [".json"],
    "toml": [".toml"],
    "html": [".html", ".htm"],
    "css": [".css", ".scss", ".sass", ".less"],
    "sql": [".sql"],
}

# Reverse lookup: extension → language
_EXT_TO_LANGUAGE: dict[str, str] = {
    ext: lang for lang, exts in LANGUAGE_EXTENSIONS.items() for ext in exts
}

# Directories that are never worth traversing
_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        ".venv",
        "venv",
        "env",
        ".env",
        "dist",
        "build",
        "target",
        ".idea",
        ".vscode",
        "vendor",
    }
)


# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FileInfo:
    """Metadata for a single discovered source file.

    Attributes:
        file_path: Absolute path to the file on disk.
        language: Canonical language name (e.g. ``"python"``).
        size_bytes: File size in bytes.
        relative_path: Path relative to the repository root.
    """

    file_path: str
    language: str
    size_bytes: int
    relative_path: str

    def __iter__(self):
        """Allow tuple-unpacking: ``path, lang, size = file_info``."""
        yield self.file_path
        yield self.language
        yield self.size_bytes


@dataclass
class CloneResult:
    """Result of a successful ``clone_and_discover`` call.

    Attributes:
        manifest: Ordered list of :class:`FileInfo` for every discovered file.
        repo_dir: Absolute path to the cloned/local repository on disk.
        branch: Branch that was checked out (``None`` for local paths).
        temp_dir: The temporary directory created during cloning, or ``None``
            if the source was already a local path.
    """

    manifest: list[FileInfo]
    repo_dir: str
    branch: Optional[str]
    temp_dir: Optional[str] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Main cloner class
# ---------------------------------------------------------------------------


class RepoCloner:
    """Clone a Git repository and discover its parseable source files.

    Parameters:
        extensions: Mapping of language names to file-extension lists.
            Defaults to :data:`LANGUAGE_EXTENSIONS`.
        max_file_size_bytes: Files larger than this threshold are skipped.
            Defaults to 1 MB.
        max_repo_size_mb: Refuse to clone repositories whose total disk usage
            exceeds this limit (in megabytes). ``0`` disables the guard.
            Defaults to ``500``.
    """

    def __init__(
        self,
        extensions: Optional[dict[str, list[str]]] = None,
        max_file_size_bytes: int = 1_048_576,  # 1 MB
        max_repo_size_mb: int = 500,
    ) -> None:
        self._extensions = extensions or LANGUAGE_EXTENSIONS
        self._ext_to_language: dict[str, str] = {
            ext: lang
            for lang, exts in self._extensions.items()
            for ext in exts
        }
        self._max_file_size_bytes = max_file_size_bytes
        self._max_repo_size_mb = max_repo_size_mb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clone_and_discover(
        self,
        source: str,
        branch: Optional[str] = None,
        shallow: bool = True,
        depth: int = 1,
    ) -> list[FileInfo]:
        """Clone *source* and return a manifest of parseable files.

        Parameters:
            source: A HTTPS/SSH Git URL **or** an absolute/relative path to a
                local repository.
            branch: Branch (or tag) to checkout.  When ``None`` the remote
                HEAD is used for remote repos; local repos are left as-is.
            shallow: Perform a shallow clone (``--depth <depth>``).  Has no
                effect for local paths.
            depth: Shallow-clone depth; only used when ``shallow=True``.

        Returns:
            A list of :class:`FileInfo` instances – one per discovered file.

        Raises:
            ValueError: If *source* is empty or the resolved path/URL is
                clearly invalid.
            RuntimeError: If cloning fails or the repository exceeds the
                configured size limit.
        """
        if not source or not source.strip():
            raise ValueError("source must be a non-empty string")

        source = source.strip()

        if self._is_local_path(source):
            return self._discover_local(source)

        return self._clone_remote(source, branch=branch, shallow=shallow, depth=depth)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_local_path(source: str) -> bool:
        """Return *True* when *source* looks like a filesystem path."""
        if source.startswith(("http://", "https://", "git@", "ssh://", "git://")):
            return False
        parsed = urlparse(source)
        # urlparse gives an empty scheme for bare paths like /foo or ./foo
        return parsed.scheme in ("", "file") or os.path.exists(source)

    def _discover_local(self, path: str) -> list[FileInfo]:
        """Discover files in an existing local repository."""
        resolved = Path(path).resolve()
        if not resolved.is_dir():
            raise ValueError(f"Local path is not a directory: {path!r}")

        logger.info("Discovering files in local repo: %s", resolved)
        return self._walk_and_filter(resolved, repo_root=resolved)

    def _clone_remote(
        self,
        url: str,
        branch: Optional[str],
        shallow: bool,
        depth: int,
    ) -> list[FileInfo]:
        """Clone *url* into a temp directory and walk the tree."""
        tmp_dir: Optional[str] = None
        try:
            tmp_dir = tempfile.mkdtemp(prefix="reporag_clone_")
            logger.info("Cloning %s → %s (branch=%s, shallow=%s)", url, tmp_dir, branch, shallow)

            clone_kwargs: dict = {
                "to_path": tmp_dir,
                "no_local": True,
            }
            if branch:
                clone_kwargs["branch"] = branch
            if shallow:
                clone_kwargs["depth"] = depth

            try:
                git.Repo.clone_from(url, **clone_kwargs)
            except GitCommandError as exc:
                raise RuntimeError(
                    f"Failed to clone repository {url!r}: {exc}"
                ) from exc

            repo_root = Path(tmp_dir)

            # Optionally guard against runaway repo sizes
            if self._max_repo_size_mb > 0:
                size_mb = self._dir_size_mb(repo_root)
                if size_mb > self._max_repo_size_mb:
                    raise RuntimeError(
                        f"Repository size {size_mb:.1f} MB exceeds the "
                        f"{self._max_repo_size_mb} MB limit."
                    )

            manifest = self._walk_and_filter(repo_root, repo_root=repo_root)
            logger.info("Discovery complete: %d files found", len(manifest))

            # Transfer ownership so the caller's temp dir is preserved
            # (caller is responsible for cleanup via CloneResult)
            return manifest

        except Exception:
            # Clean up temp directory on any error
            if tmp_dir and os.path.isdir(tmp_dir):
                logger.warning("Cleaning up temp dir after error: %s", tmp_dir)
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def _walk_and_filter(self, root: Path, repo_root: Path) -> list[FileInfo]:
        """Recursively walk *root* and return :class:`FileInfo` for each match."""
        manifest: list[FileInfo] = []

        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            # Prune unwanted directories in-place to prevent os.walk descending
            dirnames[:] = [
                d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")
            ]

            for filename in filenames:
                ext = Path(filename).suffix.lower()
                language = self._ext_to_language.get(ext)
                if language is None:
                    continue  # not a parseable extension

                abs_path = Path(dirpath) / filename
                try:
                    size = abs_path.stat().st_size
                except OSError:
                    logger.debug("Could not stat %s – skipping", abs_path)
                    continue

                if size > self._max_file_size_bytes:
                    logger.debug(
                        "Skipping large file (%d bytes): %s", size, abs_path
                    )
                    continue

                relative = abs_path.relative_to(repo_root)
                manifest.append(
                    FileInfo(
                        file_path=str(abs_path),
                        language=language,
                        size_bytes=size,
                        relative_path=str(relative),
                    )
                )

        manifest.sort(key=lambda f: f.relative_path)
        return manifest

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _dir_size_mb(path: Path) -> float:
        """Return total disk usage of *path* in megabytes."""
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for fname in filenames:
                try:
                    total += (Path(dirpath) / fname).stat().st_size
                except OSError:
                    pass
        return total / (1024 * 1024)
