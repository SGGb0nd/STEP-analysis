"""Validate and publish a complete Zenodo draft deposition."""

import argparse
import json
import os
import time
from pathlib import Path

import requests


TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


def request_json(method: str, url: str, token: str, attempts: int = 8) -> dict:
    error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.request(
                method,
                url,
                headers={"Authorization": f"Bearer {token}"},
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
        if attempt < attempts:
            time.sleep(min(60, 2**attempt))
    raise RuntimeError(
        f"Zenodo API request failed after {attempts} attempts: {method} {url}"
    ) from error


def process_exists(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def remote_files(deposition: dict) -> dict[str, int]:
    return {
        item["filename"]: int(item["filesize"])
        for item in deposition.get("files", [])
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wait for a complete Zenodo draft, validate it, and publish it."
    )
    parser.add_argument("--deposit-id", type=int, required=True)
    parser.add_argument("--state-json", type=Path, required=True)
    parser.add_argument("--upload-pid", type=int)
    parser.add_argument("--api-url", default="https://zenodo.org/api")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=36)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        raise RuntimeError("Set ZENODO_TOKEN before publishing")

    state = json.loads(args.state_json.read_text())
    expected = {item["name"]: int(item["bytes"]) for item in state["files"]}
    if not expected:
        raise RuntimeError("The upload state contains no expected files")

    deposition_url = f"{args.api_url}/deposit/depositions/{args.deposit_id}"
    deadline = time.monotonic() + args.timeout_hours * 3600

    while True:
        deposition = request_json("GET", deposition_url, token)
        observed = remote_files(deposition)
        missing = sorted(expected.keys() - observed.keys())
        extra = sorted(observed.keys() - expected.keys())
        mismatched = sorted(
            name
            for name in expected.keys() & observed.keys()
            if expected[name] != observed[name]
        )
        observed_bytes = sum(observed.get(name, 0) for name in expected)
        print(
            f"remote_files={len(observed)}/{len(expected)} "
            f"remote_bytes={observed_bytes}/{sum(expected.values())}",
            flush=True,
        )

        if not missing and not extra and not mismatched:
            break
        if extra or mismatched:
            raise RuntimeError(
                f"Zenodo file-set mismatch: extra={extra}, size_mismatch={mismatched}"
            )
        if args.upload_pid and not process_exists(args.upload_pid):
            raise RuntimeError(
                f"Uploader PID {args.upload_pid} exited with {len(missing)} files missing"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out with {len(missing)} files missing")
        time.sleep(args.poll_seconds)

    if deposition.get("submitted"):
        published = deposition
    else:
        if deposition.get("metadata", {}).get("access_right") != "open":
            raise RuntimeError("Zenodo metadata is not configured for open access")
        publish_url = deposition.get("links", {}).get("publish")
        if not publish_url:
            raise RuntimeError("Zenodo draft does not expose a publish action")
        published = request_json("POST", publish_url, token)

    result = {
        "deposit_id": args.deposit_id,
        "submitted": published.get("submitted"),
        "state": published.get("state"),
        "doi": published.get("doi"),
        "record_url": published.get("links", {}).get("record_html")
        or published.get("links", {}).get("html"),
        "file_count": len(remote_files(published)),
        "total_bytes": sum(remote_files(published).values()),
    }
    result_path = args.result_json or args.state_json.with_name(
        "zenodo_publish_result.json"
    )
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
