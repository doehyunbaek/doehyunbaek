# src/doehyunbaek/anon_to_zenodo.py
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable, Sequence

# Resolve package version (used for --version on this subcommand)
try:
    _PKG_VERSION = version("doehyunbaek")
except PackageNotFoundError:  # during dev/editable installs
    _PKG_VERSION = "0.0.0"

# Runtime deps from the original tool
from zenodo_client import Creator, Metadata, Zenodo  # type: ignore
import pystow  # type: ignore

__all__ = [
    "zip_directory",
    "upload_cwd",
    "register_subcommand",
    "UploadResult",
]

# -------------------------
# Ported core functionality
# -------------------------

DEFAULT_IGNORE = {
    ".git",
    ".venv",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".DS_Store",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
}

ALWAYS_IGNORE_SUFFIXES = {".pyc", ".pyo"}


def _iter_paths(root: Path, ignore_names: set[str]) -> Iterable[Path]:
    for path in root.rglob("*"):
        if any(part in ignore_names for part in path.parts):
            continue
        if path.is_file() and path.suffix in ALWAYS_IGNORE_SUFFIXES:
            continue
        if path.is_file():
            yield path


def zip_directory(
    directory: str | os.PathLike[str] = ".",
    *,
    output: str | None = None,
    ignore: Sequence[str] | None = None,
) -> Path:
    """Create a zip archive of the given directory and return its path."""
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    ignore_names = DEFAULT_IGNORE.union(ignore or [])

    if output is None:
        fd, tmp_path = tempfile.mkstemp(prefix=f"{root.name}-", suffix=".zip")
        os.close(fd)
        archive_path = Path(tmp_path)
    else:
        archive_path = Path(output).resolve()

    print(f"Creating archive {archive_path} from {root} (ignoring: {', '.join(sorted(ignore_names)) or '<none>'})")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in _iter_paths(root, ignore_names):
            if file_path == archive_path:  # avoid including the archive itself
                continue
            arcname = file_path.relative_to(root)
            zf.write(file_path, arcname.as_posix())
    try:
        size = archive_path.stat().st_size
        print(f"Archive created at {archive_path} ({size} bytes)")
    except OSError:
        print(f"Archive created at {archive_path}")
    return archive_path


@dataclass
class UploadResult:
    deposition_id: str
    response_json: dict
    archive_path: Path


def upload_cwd(
    *,
    title: str,
    description: str,
    creators: Sequence[str],
    sandbox: bool = False,
    directory: str | os.PathLike[str] = ".",
    publish: bool = True,
    license: str | None = "MIT",
) -> UploadResult:
    """Zip the target directory and upload to Zenodo."""
    archive = zip_directory(directory)

    def _slugify(text: str) -> str:
        # Lowercase, replace non-alphanumeric with hyphens, collapse repeats, strip edges
        return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-") or "untitled"

    slug = _slugify(title)
    config_module = "anon_to_zenodo"  # keep same namespace for continuity

    existing_id = pystow.get_config(config_module, slug)
    md_creators = [Creator(name=name) for name in creators]
    metadata = Metadata(
        title=title,
        upload_type="dataset",
        description=description,
        creators=md_creators,
        license=license,
    )
    zen = Zenodo(sandbox=sandbox)
    if existing_id:
        print(f"Reusing existing deposition {existing_id} for title '{title}' (slug={slug})")
        if publish:
            try:
                dep_json = zen._get_deposition(existing_id).json()  # type: ignore[attr-defined]
                if not dep_json.get("submitted"):
                    print(f"Existing deposition {existing_id} not published yet; publishing to enable versioning")
                    zen.publish(existing_id)
            except Exception as e:  # best effort
                print(f"Warning: could not verify/publish existing deposition {existing_id}: {e}")
        res = zen.update(existing_id, paths=[archive], publish=publish)
    else:
        print(f"Creating new deposition for title '{title}' (slug={slug})")
        res = zen.create(data=metadata, paths=[archive], publish=publish)
        pystow.write_config(config_module, slug, str(res.json()["id"]))

    return UploadResult(deposition_id=str(res.json()["id"]), response_json=res.json(), archive_path=archive)


# -------------------------
# Subcommand registration
# -------------------------

def _handle_anon_to_zenodo(args: argparse.Namespace) -> int:
    # Dynamic defaults (match original tool)
    if not args.title:
        target_dir = getattr(args, "directory", ".") or "."
        args.title = Path(target_dir).resolve().name
    if not args.description:
        args.description = "automatic anonymization and uplaod to zenodo"
    if not args.creators:
        args.creators = ["Authors, Anonymous"]

    result = upload_cwd(
        title=args.title,
        description=args.description,
        creators=args.creators,
        sandbox=args.sandbox,
        directory=args.directory,
        publish=args.publish,
        license=args.license,
    )
    if args.json_out:
        print(json.dumps(result.response_json, indent=2))
    else:
        print(result.response_json["links"]["html"])
    return 0


def register_subcommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the `anon_to_zenodo` subcommand on a parent ArgumentParser."""
    p = subparsers.add_parser(
        "anon_to_zenodo",
        help="Zip directory and upload to Zenodo using existing credentials.",
        description="Zip a directory and create (optionally publish) a Zenodo deposition.",
    )
    p.add_argument(
        "--title",
        help="Title for deposition (default: directory name)",
    )
    p.add_argument(
        "--description",
        help="Description / abstract (default: 'automatic anonymization and uplaod to zenodo')",
    )
    p.add_argument(
        "--creator",
        dest="creators",
        action="append",
        help="Repeatable: creator 'Family, Given' (default: 'Authors, Anonymous')",
    )
    p.add_argument(
        "--sandbox",
        action="store_true",
        help="Use Zenodo sandbox (https://sandbox.zenodo.org)",
    )
    p.add_argument(
        "--publish",
        action="store_true",
        help="Create draft but do not publish",
    )
    p.add_argument(
        "--dir",
        dest="directory",
        default=".",
        help="Directory to archive (default: %(default)s)",
    )
    p.add_argument(
        "--license",
        default="MIT",
        help="SPDX license identifier (default: %(default)s)",
    )
    p.add_argument(
        "--json",
        dest="json_out",
        action="store_true",
        help="Print JSON response instead of URL",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (repeat for more)",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"anon_to_zenodo (doehyunbaek) {_PKG_VERSION}",
    )

    # Attach handler
    p.set_defaults(_handler=_handle_anon_to_zenodo)
