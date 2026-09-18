#!/usr/bin/env python3
"""Download one experiment video safely, with resume and a local manifest."""
import argparse
import hashlib
import http.cookiejar
from http.client import IncompleteRead
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import HTTPCookieProcessor, Request, build_opener


class DownloadError(RuntimeError):
    pass


def manifest_path(output):
    return output.with_name(output.name + ".download.json")


def part_path(output):
    return output.with_name(output.name + ".part")


def part_manifest_path(output):
    return output.with_name(output.name + ".part.json")


def url_id(url):
    return hashlib.sha256(url.encode()).hexdigest()


def _read_json(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _safe_name(value):
    return value.replace("\n", " ").replace("\r", " ")[:180]


def _opener():
    return build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()))


def _is_synology_share(url):
    parts = [p for p in urlparse(url).path.split("/") if p]
    return len(parts) == 2 and parts[0] == "sharing" and bool(parts[1])


def _resolve_share(opener, shared_url):
    """Resolve a public, single-file Synology share without logging its token."""
    parsed = urlparse(shared_url)
    token = parsed.path.rstrip("/").split("/")[-1]
    try:
        with opener.open(Request(shared_url, headers={"User-Agent": "aim-video-bootstrap/1"}), timeout=30) as response:
            page = response.read(2_000_000).decode("utf-8", "replace")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise DownloadError("Could not open the sharing page.") from exc
    match = re.search(r'"filename"\s*:\s*"((?:[^"\\]|\\.)*)"', page)
    if not match:
        if re.search(r'sharing_status\s*=\s*"password"|password.*protect', page, re.I):
            raise DownloadError("The sharing link is password-protected; password authentication is not implemented. Use an authorized direct file URL.")
        raise DownloadError("The sharing page did not expose a downloadable single file.")
    try:
        filename = json.loads('"' + match.group(1) + '"')
    except json.JSONDecodeError as exc:
        raise DownloadError("The sharing page returned an invalid filename.") from exc
    if not filename or "/" in filename or "\\" in filename:
        raise DownloadError("The sharing page returned an unsafe filename.")
    # The sharing page's loaded File Browser client changes `/sharing/<id>` to
    # `/fsdownload/<id>/<filename>`. Keep the opaque id out of output/manifests.
    base = f"{parsed.scheme}://{parsed.netloc}"
    return f"{base}/fsdownload/{quote(token)}/{quote(filename)}"


def _request(opener, url, start, validator):
    headers = {"User-Agent": "aim-video-bootstrap/1", "Accept": "*/*", "Accept-Encoding": "identity"}
    if start:
        headers["Range"] = f"bytes={start}-"
        if validator:
            headers["If-Range"] = validator
    return opener.open(Request(url, headers=headers), timeout=60)


def _content_total(response, start):
    content_range = response.headers.get("Content-Range", "")
    if response.status == 206:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
        if not match:
            raise DownloadError("Missing or invalid Content-Range; partial file was kept.")
        first, last, total = map(int, match.groups())
        if first != start or not first <= last < total:
            raise DownloadError("Unexpected Content-Range; partial file was kept.")
        return total
    length = response.headers.get("Content-Length")
    return start + int(length) if length and length.isdigit() else None


def _reject_non_media(response, first):
    kind = response.headers.get_content_type().lower()
    prefix = first.lstrip().lower()
    if kind.startswith("text/") or kind == "application/json" or prefix.startswith((b"<html", b"<!doctype html", b"{", b"[")):
        raise DownloadError("Server returned HTML or JSON instead of a media file; check the sharing link or its password.")


def _digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_state(path, state):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _finish(output, state, sha256):
    part = part_path(output)
    digest = _digest(part)
    if sha256 and digest.lower() != sha256.lower():
        raise DownloadError("Downloaded file SHA-256 does not match the requested value.")
    state.update(size=part.stat().st_size, sha256=digest)
    # The final digest is a recovery journal if interrupted between the two renames.
    _write_state(part_manifest_path(output), state)
    part.replace(output)
    part_manifest_path(output).replace(manifest_path(output))
    return "downloaded"


def download(source_url, output, sha256=None, retries=3):
    parsed = urlparse(source_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise DownloadError("Only HTTP(S) video/share URLs are supported.")
    if sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", sha256):
        raise DownloadError("Expected SHA-256 must contain 64 hexadecimal characters.")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = url_id(source_url)
    existing = _read_json(manifest_path(output))
    if output.exists():
        if not existing:
            journal = _read_json(part_manifest_path(output))
            if journal and journal.get("url_id") == identity and journal.get("sha256") == _digest(output):
                existing = journal
                _write_state(manifest_path(output), existing)
                part_manifest_path(output).unlink(missing_ok=True)
        if existing and existing.get("url_id") == identity and existing.get("sha256") == _digest(output):
            if sha256 and existing["sha256"].lower() != sha256.lower():
                raise DownloadError("Existing file SHA-256 differs from the requested value.")
            return "skipped"
        raise DownloadError("Output already exists for a different source or lacks a valid manifest.")

    part = part_path(output)
    part_state = _read_json(part_manifest_path(output))
    if part.exists() and (not part_state or part_state.get("url_id") != identity):
        raise DownloadError("Incomplete file belongs to a different source; choose another output path.")
    start = part.stat().st_size if part.exists() else 0
    validator = (part_state or {}).get("validator")
    expected = (part_state or {}).get("size")
    if start and expected == start:
        return _finish(output, part_state, sha256)
    opener = _opener()
    media_url = _resolve_share(opener, source_url) if _is_synology_share(source_url) else source_url

    for attempt in range(retries + 1):
        # Re-evaluate after EVERY interrupted attempt, not just on process startup.
        if not validator or validator.startswith("W/") or (expected is not None and start > expected):
            start, validator, expected = 0, None, None
        try:
            with _request(opener, media_url, start, validator) as response:
                if start and response.status == 200:
                    # If-Range failed or Range is unsupported: use this full response.
                    start, validator, expected = 0, None, None
                if response.status not in (200, 206):
                    raise DownloadError(f"Server returned HTTP {response.status}.")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise DownloadError("Compressed HTTP transfer cannot be safely resumed.")
                total = _content_total(response, start)
                new_validator = response.headers.get("ETag") or response.headers.get("Last-Modified")
                if start and ((expected is not None and total != expected) or
                              (new_validator and validator != new_validator)):
                    raise DownloadError("Remote file changed during resume; choose a new output path.")
                first = response.read(8192)
                _reject_non_media(response, first)
                expected = total
                validator = new_validator or validator
                if expected is not None and shutil.disk_usage(output.parent).free < expected - start:
                    raise DownloadError("Not enough free disk space for the video.")
                with part.open("ab" if start else "wb") as stream:
                    _write_state(part_manifest_path(output), {"url_id": identity, "size": expected, "validator": validator})
                    stream.write(first)
                    last_progress = time.monotonic()
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(block)
                        if time.monotonic() - last_progress >= 10:
                            print(f"download_video: {stream.tell() / 1024**3:.2f} GiB received", file=sys.stderr)
                            last_progress = time.monotonic()
        except DownloadError:
            raise
        except (HTTPError, URLError, TimeoutError, OSError, IncompleteRead) as exc:
            if attempt >= retries:
                raise DownloadError("Download failed after retries; the .part file was kept for resume.") from exc
            time.sleep(min(2 ** attempt, 4))
            start = part.stat().st_size if part.exists() else 0
            continue

        actual = part.stat().st_size
        if expected is not None and actual != expected:
            if attempt >= retries:
                raise DownloadError("Download ended early; the .part file was kept for resume.")
            start = actual
            time.sleep(min(2 ** attempt, 4))
            continue
        if actual == 0:
            raise DownloadError("Server returned an empty file.")
        return _finish(output, {"url_id": identity, "size": actual, "validator": validator}, sha256)
    raise DownloadError("Download could not complete within the retry limit.")


def main():
    parser = argparse.ArgumentParser(description="Download one video with safe resume support.")
    parser.add_argument("--url", default=os.environ.get("VIDEO_URL"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sha256", help="Expected SHA-256 digest")
    args = parser.parse_args()
    if not args.url:
        parser.error("--url or VIDEO_URL is required")
    try:
        result = download(args.url, args.output, args.sha256)
    except DownloadError as exc:
        print(f"download_video: {_safe_name(str(exc))}", file=sys.stderr)
        return 1
    print(f"download_video: {result} ({args.output.name}, {args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
