import hashlib
import http.server
import importlib.util
import json
import socketserver
import tempfile
import threading
import unittest
from pathlib import Path


MODULE = Path(__file__).with_name("download_video.py")
SPEC = importlib.util.spec_from_file_location("download_video", MODULE)
download_video = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download_video)


class Handler(http.server.BaseHTTPRequestHandler):
    payload = b"video bytes for downloader tests"
    requests = []
    interrupt_once = False

    def log_message(self, *_args):
        pass

    def do_GET(self):
        type(self).requests.append((self.path, self.headers.get("Range")))
        if self.path == "/sharing/public-token":
            body = b'SYNO.SDS.ExtraSession = {"filename" : "clip.mov"};'
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Set-Cookie", "sharing_sid=test-session; Path=/")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/sharing/private-token":
            body = b'SYNO.SDS.Session={sharing_status = "password"}'
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/fsdownload/") and "sharing_sid=test-session" not in self.headers.get("Cookie", ""):
            self.send_error(403)
            return
        if self.path.startswith("/html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", "22")
            self.end_headers()
            self.wfile.write(b"<html>login</html>")
            return
        if self.path.startswith("/ignore-range"):
            if self.headers.get("Range") and self.headers.get("If-Range") != '"old-version"':
                self.send_error(412)
                return
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("ETag", '"new-version"')
            self.send_header("Content-Length", str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)
            return
        if self.path.startswith("/interrupt") and type(self).interrupt_once:
            type(self).interrupt_once = False
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            if self.path != "/interrupt-no-validator":
                self.send_header("ETag", '"test-video-v1"')
            self.send_header("Content-Length", str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload[:8])
            self.wfile.flush()
            self.connection.close()
            return
        start = 0
        if self.headers.get("Range"):
            start = int(self.headers["Range"].split("=")[1].split("-")[0])
            self.send_response(206)
            offset = 0 if self.path == "/bad-range" else start
            self.send_header("Content-Range", f"bytes {offset}-{len(self.payload)-1}/{len(self.payload)}")
        else:
            self.send_response(200)
        body = self.payload[start:]
        self.send_header("Content-Type", "video/mp4")
        self.send_header("ETag", '"test-video-v1"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class DownloadVideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        Handler.requests = []
        Handler.interrupt_once = False
        self.tmp = tempfile.TemporaryDirectory()
        self.output = Path(self.tmp.name) / "clip.mp4"

    def tearDown(self):
        self.tmp.cleanup()

    def test_downloads_and_writes_manifest(self):
        download_video.download(self.base + "/video", self.output)
        self.assertEqual(self.output.read_bytes(), Handler.payload)
        manifest = json.loads(download_video.manifest_path(self.output).read_text())
        self.assertEqual(manifest["size"], len(Handler.payload))
        self.assertNotIn(self.base, json.dumps(manifest))

    def test_rejects_html_response_without_writing_output(self):
        with self.assertRaisesRegex(download_video.DownloadError, "HTML"):
            download_video.download(self.base + "/html", self.output)
        self.assertFalse(self.output.exists())

    def test_resumes_after_interrupted_stream(self):
        Handler.interrupt_once = True
        download_video.download(self.base + "/interrupt", self.output, retries=2)
        self.assertEqual(self.output.read_bytes(), Handler.payload)
        self.assertIn(("/interrupt", "bytes=8-"), Handler.requests)

    def test_ignored_range_restarts_instead_of_appending(self):
        part = download_video.part_path(self.output)
        part.write_bytes(Handler.payload[:8])
        download_video.part_manifest_path(self.output).write_text(json.dumps({
            "url_id": download_video.url_id(self.base + "/ignore-range"),
            "size": len(Handler.payload), "validator": '"old-version"',
        }))
        download_video.download(self.base + "/ignore-range", self.output, retries=0)
        self.assertEqual(self.output.read_bytes(), Handler.payload)
        self.assertEqual(Handler.requests, [("/ignore-range", "bytes=8-")])

    def test_no_validator_restarts_on_retry(self):
        Handler.interrupt_once = True
        download_video.download(self.base + "/interrupt-no-validator", self.output, retries=1)
        self.assertEqual(self.output.read_bytes(), Handler.payload)
        self.assertEqual(Handler.requests, [("/interrupt-no-validator", None)] * 2)

    def test_wrong_content_range_cannot_corrupt_partial(self):
        partial = download_video.part_path(self.output)
        partial.write_bytes(Handler.payload[:8])
        download_video.part_manifest_path(self.output).write_text(json.dumps({
            "url_id": download_video.url_id(self.base + "/bad-range"),
            "size": len(Handler.payload), "validator": '"test-video-v1"',
        }))
        with self.assertRaises(download_video.DownloadError):
            download_video.download(self.base + "/bad-range", self.output, retries=0)
        self.assertFalse(self.output.exists())
        self.assertEqual(partial.read_bytes(), Handler.payload[:8])

    def test_rejects_local_file_url(self):
        source = Path(self.tmp.name) / "local-file"
        source.write_bytes(Handler.payload)
        with self.assertRaisesRegex(download_video.DownloadError, "HTTP"):
            download_video.download(source.as_uri(), self.output)

    def test_completed_partial_is_promoted_without_eof_range(self):
        url = self.base + "/video"
        download_video.part_path(self.output).write_bytes(Handler.payload)
        download_video.part_manifest_path(self.output).write_text(json.dumps({
            "url_id": download_video.url_id(url), "size": len(Handler.payload),
            "validator": '"test-video-v1"',
        }))
        download_video.download(url, self.output)
        self.assertEqual(self.output.read_bytes(), Handler.payload)
        self.assertEqual(Handler.requests, [])

    def test_final_rename_crash_recovers_from_digest_journal(self):
        url = self.base + "/video"
        self.output.write_bytes(Handler.payload)
        download_video.part_manifest_path(self.output).write_text(json.dumps({
            "url_id": download_video.url_id(url), "size": len(Handler.payload),
            "sha256": hashlib.sha256(Handler.payload).hexdigest(),
        }))
        self.assertEqual(download_video.download(url, self.output), "skipped")
        self.assertTrue(download_video.manifest_path(self.output).exists())
        self.assertEqual(Handler.requests, [])

    def test_weak_etag_restarts_instead_of_resuming(self):
        url = self.base + "/video"
        download_video.part_path(self.output).write_bytes(Handler.payload[:8])
        download_video.part_manifest_path(self.output).write_text(json.dumps({
            "url_id": download_video.url_id(url), "size": len(Handler.payload),
            "validator": 'W/"weak"',
        }))
        download_video.download(url, self.output)
        self.assertEqual(Handler.requests, [("/video", None)])
        self.assertEqual(self.output.read_bytes(), Handler.payload)

    def test_rerun_skips_validated_same_source(self):
        url = self.base + "/video"
        download_video.download(url, self.output)
        before = len(Handler.requests)
        self.assertEqual(download_video.download(url, self.output), "skipped")
        self.assertEqual(len(Handler.requests), before)

    def test_collision_does_not_overwrite_a_different_source_or_log_token(self):
        download_video.download(self.base + "/video?token=private", self.output)
        with self.assertRaisesRegex(download_video.DownloadError, "different source") as error:
            download_video.download(self.base + "/other?token=secret", self.output)
        self.assertNotIn("secret", str(error.exception))
        self.assertEqual(self.output.read_bytes(), Handler.payload)

    def test_sha256_is_checked(self):
        digest = hashlib.sha256(Handler.payload).hexdigest()
        download_video.download(self.base + "/video", self.output, sha256=digest)
        with self.assertRaisesRegex(download_video.DownloadError, "SHA-256"):
            download_video.download(self.base + "/other", Path(self.tmp.name) / "other.mp4", sha256="0" * 64)

    def test_synology_single_file_share_resolves_without_recording_token(self):
        download_video.download(self.base + "/sharing/public-token", self.output)
        self.assertEqual(self.output.read_bytes(), Handler.payload)
        self.assertIn(("/fsdownload/public-token/clip.mov", None), Handler.requests)
        self.assertNotIn("public-token", download_video.manifest_path(self.output).read_text())

    def test_synology_password_gate_is_actionable(self):
        with self.assertRaisesRegex(download_video.DownloadError, "password-protected"):
            download_video.download(self.base + "/sharing/private-token", self.output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
