"""Create or update a Zenodo draft and upload prepared archives."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import requests


DEFAULT_QUOTA_BYTES = 50 * 1024**3
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def api_request(
    method: str,
    url: str,
    token: str,
    *,
    json_payload: dict | None = None,
) -> dict:
    error: Exception | None = None
    for attempt in range(1, 9):
        try:
            response = requests.request(
                method,
                url,
                headers={"Authorization": f"Bearer {token}"},
                json=json_payload,
                timeout=(30, 120),
            )
            if response.status_code not in TRANSIENT_STATUS_CODES:
                response.raise_for_status()
                return response.json()
            error = requests.HTTPError(
                f"Transient HTTP {response.status_code} for {method} {url}",
                response=response,
            )
        except (requests.ConnectionError, requests.Timeout) as caught:
            error = caught
        if attempt < 8:
            time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"Zenodo API request failed after 8 attempts: {method} {url}") from error


def upload_file(
    bucket_url: str,
    path: Path,
    token: str,
    attempts: int = 5,
) -> dict:
    error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with path.open("rb") as handle:
                response = requests.put(
                    f"{bucket_url}/{path.name}",
                    headers={"Authorization": f"Bearer {token}"},
                    data=handle,
                    timeout=(30, 900),
                )
            if response.status_code not in TRANSIENT_STATUS_CODES:
                response.raise_for_status()
                return response.json()
            error = requests.HTTPError(
                f"Transient HTTP {response.status_code} while uploading {path.name}",
                response=response,
            )
        except requests.RequestException as caught:
            error = caught
        if attempt == attempts:
            break
        time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"Failed to upload {path.name} after {attempts} attempts") from error


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload STEP-analysis archives to a Zenodo draft record."
    )
    parser.add_argument("--archives-dir", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--deposit-id", type=int)
    parser.add_argument("--api-url", default="https://zenodo.org/api")
    parser.add_argument("--state-json", type=Path)
    parser.add_argument("--allow-over-default-quota", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    archive_parts = sorted(
        args.archives_dir.glob("STEP_analysis_*.part*.tar.zst")
    )
    files = archive_parts or sorted(
        path
        for path in args.archives_dir.glob("STEP_analysis_*.tar.zst")
        if ".part" not in path.name
    )
    files.extend(
        path
        for path in [
            args.archives_dir / "SHA256SUMS",
            args.archives_dir / "SHA256SUMS.parts",
            args.archives_dir / "archive_inventory.json",
            args.archives_dir / "README_REASSEMBLY.txt",
        ]
        if path.exists()
    )
    if not files:
        raise FileNotFoundError(f"No prepared archives found in {args.archives_dir}")

    inventory_path = args.archives_dir / "archive_inventory.json"
    if not inventory_path.exists():
        raise FileNotFoundError("Missing archive_inventory.json; rebuild the archives")
    inventory = json.loads(inventory_path.read_text())
    manifest_digest = inventory.get("manifest_sha256")
    if not manifest_digest:
        raise ValueError("Archive inventory predates manifest hashing; rebuild the archives")
    manifest_path = Path(inventory["manifest"])
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest recorded by archive build is missing: {manifest_path}")
    if sha256(manifest_path) != manifest_digest:
        raise ValueError("Zenodo manifest changed after archive build; rebuild the archives")
    for tier, tier_result in inventory.get("tiers", {}).items():
        if "archive_bytes" not in tier_result or "sha256" not in tier_result:
            raise ValueError(
                f"Archive inventory for {tier} lacks completed-build metadata; "
                "rebuild the archives"
            )
        archive_path = args.archives_dir / Path(tier_result["archive"]).name
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive recorded for {tier} is missing: {archive_path}")
        if archive_path.stat().st_size != int(tier_result["archive_bytes"]):
            raise ValueError(f"Archive size changed after build: {archive_path}")

    total_bytes = sum(path.stat().st_size for path in files)
    if total_bytes > DEFAULT_QUOTA_BYTES and not args.allow_over_default_quota:
        raise ValueError(
            f"Prepared files total {total_bytes} bytes, above Zenodo's default "
            "50 GiB quota. Request an increased quota or pass "
            "--allow-over-default-quota after it is approved."
        )

    metadata = json.loads(args.metadata_json.read_text())
    if "metadata" not in metadata:
        metadata = {"metadata": metadata}
    plan = {
        "api_url": args.api_url,
        "deposit_id": args.deposit_id,
        "total_bytes": total_bytes,
        "files": [{"name": path.name, "bytes": path.stat().st_size} for path in files],
        "publish": False,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        raise RuntimeError("Set ZENODO_TOKEN in the environment before uploading")

    if args.deposit_id is None:
        deposition = api_request(
            "POST", f"{args.api_url}/deposit/depositions", token, json_payload={}
        )
        deposit_id = int(deposition["id"])
    else:
        deposit_id = args.deposit_id
        deposition = api_request(
            "GET", f"{args.api_url}/deposit/depositions/{deposit_id}", token
        )

    deposition = api_request(
        "PUT",
        f"{args.api_url}/deposit/depositions/{deposit_id}",
        token,
        json_payload=metadata,
    )
    bucket_url = deposition["links"]["bucket"]
    remote_files = {
        item.get("filename"): item
        for item in deposition.get("files", [])
        if item.get("filename")
    }
    uploaded = []
    skipped = []
    state_path = args.state_json or args.archives_dir / "zenodo_draft_state.json"
    state = {
        **plan,
        "deposit_id": deposit_id,
        "record_url": deposition["links"].get("html"),
        "uploaded": uploaded,
        "skipped": skipped,
        "publish": False,
    }
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    for path in files:
        remote = remote_files.get(path.name)
        if remote and int(remote.get("filesize", -1)) == path.stat().st_size:
            skipped.append(remote)
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            continue
        uploaded.append(upload_file(bucket_url, path, token))
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
