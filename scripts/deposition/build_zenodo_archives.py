"""Validate the Zenodo manifest and build deterministic tiered archives."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


READY_ACTIONS = {
    "stage_and_upload",
    "upload",
    "upload_analysis_ready_input",
}
REFERENCE_ACTIONS = {"link_only"}
TIER_DIRECTORIES = {"raw": "raw", "intermediate": "intermediate", "output": "outputs"}
IGNORED_NAMES = {
    ".DS_Store",
    ".git",
    ".ipynb_checkpoints",
    ".venv",
    "__MACOSX",
    "venv",
    "__pycache__",
    "step_source_snapshot.tar.zst",
}
IGNORED_SUFFIXES = {
    ".cloupe",
    ".docx",
    ".html",
    ".ipynb",
    ".log",
    ".md",
    ".py",
    ".qmd",
    ".r",
    ".rmd",
    ".sh",
    ".zip",
}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "tier",
        "dataset",
        "archive_path",
        "source_path",
        "size_bytes",
        "status",
        "zenodo_action",
        "producer_or_source",
    }
    if not rows:
        raise ValueError("The Zenodo manifest is empty")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Manifest columns are missing: {sorted(missing)}")
    return rows


def is_safe_relative(path: Path) -> bool:
    return not path.is_absolute() and ".." not in path.parts


def split_sources(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


def resolve_sources(repo_root: Path, value: str) -> list[Path]:
    resolved = []
    for item in split_sources(value):
        pattern = Path(item)
        if pattern.is_absolute():
            matches = [pattern] if pattern.exists() else []
        elif any(char in item for char in "*?["):
            matches = sorted(repo_root.glob(item))
        else:
            candidate = repo_root / pattern
            matches = [candidate] if candidate.exists() else []
        if not matches:
            raise FileNotFoundError(f"Manifest source does not exist: {item}")
        resolved.extend(matches)
    return resolved


def hardlink_or_copy(source: str, destination: str) -> str:
    source_path = Path(source).resolve()
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        raise FileExistsError(f"Archive destination is duplicated: {destination_path}")
    try:
        os.link(source_path, destination_path)
    except OSError:
        shutil.copy2(source_path, destination_path)
    return str(destination_path)


def should_ignore_name(name: str) -> bool:
    return name in IGNORED_NAMES or Path(name).suffix.lower() in IGNORED_SUFFIXES


def ignore_paths(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if should_ignore_name(name)}


def copy_directory_contents(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for child in sorted(source.resolve().iterdir()):
        if should_ignore_name(child.name):
            continue
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(
                child,
                target,
                dirs_exist_ok=True,
                symlinks=False,
                copy_function=hardlink_or_copy,
                ignore=ignore_paths,
            )
        else:
            hardlink_or_copy(str(child), str(target))


def copy_source(source: Path, destination: Path) -> None:
    source = source.resolve()
    if source.is_dir():
        shutil.copytree(
            source,
            destination,
            dirs_exist_ok=True,
            symlinks=False,
            copy_function=hardlink_or_copy,
            ignore=ignore_paths,
        )
    else:
        if destination.exists():
            raise FileExistsError(f"Archive destination is duplicated: {destination}")
        hardlink_or_copy(str(source), str(destination))


def stage_row(repo_root: Path, stage_root: Path, row: dict[str, str]) -> int:
    relative_destination = Path(row["archive_path"])
    if not is_safe_relative(relative_destination):
        raise ValueError(f"Unsafe archive path: {relative_destination}")
    destination = stage_root / relative_destination

    sources = resolve_sources(repo_root, row["source_path"])
    destination_is_directory = row["archive_path"].endswith("/")
    if not destination_is_directory and len(sources) != 1:
        raise ValueError(
            f"File destination requires exactly one source: {row['archive_path']}"
        )

    if destination_is_directory and len(sources) == 1 and sources[0].is_dir():
        copy_directory_contents(sources[0], destination)
    elif destination_is_directory:
        destination.mkdir(parents=True, exist_ok=True)
        for source in sources:
            copy_source(source, destination / source.name)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy_source(sources[0], destination)

    total = 0
    targets = [destination] if destination.exists() else []
    for target in targets:
        if target.is_file():
            total += target.stat().st_size
        else:
            total += sum(path.stat().st_size for path in target.rglob("*") if path.is_file())
    return total


def write_subset_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_archive_summary(path: Path, inventory: dict[str, object]) -> None:
    fields = (
        "tier",
        "manifest_rows",
        "staged_bytes",
        "archive_bytes",
        "sha256",
        "archive_file",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for tier, result in inventory["tiers"].items():
            writer.writerow(
                {
                    "tier": tier,
                    "manifest_rows": result["rows"],
                    "staged_bytes": result["staged_bytes"],
                    "archive_bytes": result["archive_bytes"],
                    "sha256": result["sha256"],
                    "archive_file": Path(result["archive"]).name,
                }
            )


def human_size(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "external"
    value = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def write_item_summary(path: Path, items: list[dict[str, object]]) -> None:
    fields = (
        "tier",
        "dataset_or_analysis",
        "packaged",
        "staged_bytes",
        "staged_size",
        "archive_path",
        "source_path",
        "producer_or_source",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(items)


def build_archive(stage_root: Path, archive_root: str, output_path: Path) -> None:
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    command = [
        "tar",
        "--sort=name",
        "--mtime=@0",
        "--owner=0",
        "--group=0",
        "--numeric-owner",
        "-I",
        "zstd -T0 -10",
        "-cf",
        str(temporary_path),
        archive_root,
        "MANIFEST.tsv",
    ]
    try:
        subprocess.run(command, cwd=stage_root, check=True)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the STEP-analysis manifest and build Zenodo archives."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest", type=Path, default=Path("docs/zenodo_manifest.tsv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/zenodo"))
    parser.add_argument(
        "--tiers", nargs="+", default=["raw", "intermediate", "output"]
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--allow-deferred",
        action="store_true",
        help="build only the ready rows even when the manifest still has deferred rows",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(manifest_path)
    requested_tiers = set(args.tiers)
    selected = [
        row
        for row in rows
        if row["tier"] in requested_tiers and row["zenodo_action"] in READY_ACTIONS
    ]
    deferred = [
        row
        for row in rows
        if row["tier"] in requested_tiers
        and row["zenodo_action"] not in READY_ACTIONS | REFERENCE_ACTIONS
    ]
    external_references = [
        row
        for row in rows
        if row["tier"] in requested_tiers
        and row["zenodo_action"] in REFERENCE_ACTIONS
    ]
    if not selected:
        raise ValueError("No ready manifest rows match the requested tiers")
    if deferred and not args.validate_only and not args.allow_deferred:
        raise RuntimeError(
            f"Manifest has {len(deferred)} deferred rows. Resolve or remove them "
            "before building the final deposit, or pass --allow-deferred for an "
            "explicit partial build."
        )

    with tempfile.TemporaryDirectory(
        prefix="step-analysis-zenodo-", dir=output_dir
    ) as temporary:
        stage_root = Path(temporary)
        inventory = {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "tiers": {},
            "external_references": external_references,
            "deferred": deferred,
        }
        staged_row_sizes: dict[tuple[str, str], int] = {}
        for tier in args.tiers:
            tier_rows = [row for row in selected if row["tier"] == tier]
            if not tier_rows:
                continue
            if tier not in TIER_DIRECTORIES:
                raise ValueError(f"Unknown manifest tier: {tier}")
            archive_root = TIER_DIRECTORIES[tier]
            staged_bytes = 0
            for row in tier_rows:
                row_bytes = stage_row(repo_root, stage_root, row)
                staged_bytes += row_bytes
                staged_row_sizes[(row["tier"], row["archive_path"])] = row_bytes
            write_subset_manifest(stage_root / "MANIFEST.tsv", tier_rows)
            archive_path = output_dir / f"STEP_analysis_{tier}.tar.zst"
            tier_result = {
                "rows": len(tier_rows),
                "staged_bytes": staged_bytes,
                "archive": str(archive_path),
            }
            if not args.validate_only:
                build_archive(stage_root, archive_root, archive_path)
                tier_result["archive_bytes"] = archive_path.stat().st_size
                tier_result["sha256"] = sha256(archive_path)
            inventory["tiers"][tier] = tier_result
            shutil.rmtree(stage_root / archive_root)
            (stage_root / "MANIFEST.tsv").unlink()

        inventory["items"] = []
        for row in rows:
            if row["tier"] not in requested_tiers:
                continue
            staged_bytes = staged_row_sizes.get((row["tier"], row["archive_path"]))
            inventory["items"].append(
                {
                    "tier": row["tier"],
                    "dataset_or_analysis": row["dataset"],
                    "packaged": "yes" if staged_bytes is not None else "no",
                    "staged_bytes": staged_bytes if staged_bytes is not None else "",
                    "staged_size": human_size(staged_bytes),
                    "archive_path": row["archive_path"],
                    "source_path": row["source_path"],
                    "producer_or_source": row["producer_or_source"],
                }
            )

        write_item_summary(output_dir / "item_summary.tsv", inventory["items"])

        if not args.validate_only:
            inventory_path = output_dir / "archive_inventory.json"
            inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")
            checksum_lines = []
            for tier_result in inventory["tiers"].values():
                archive = Path(tier_result["archive"])
                checksum_lines.append(f"{tier_result['sha256']}  {archive.name}")
            (output_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")
            write_archive_summary(output_dir / "archive_summary.tsv", inventory)
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
