"""CPU-only broker tests. No sudo, dmesg, CUDA, SAPIEN or privileged socket."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import stat
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_kernel_log as broker


class KernelBrokerTests(unittest.TestCase):
    def exchange(self, payload, allowed_uid=None, reader=None):
        client, server = socket.socketpair()
        self.addCleanup(client.close)
        self.addCleanup(server.close)
        client.settimeout(1)
        client.sendall(payload)
        reader = reader or Mock(return_value="kernel\n")
        stop = broker._handle_connection(server, os.getuid() if allowed_uid is None else allowed_uid, reader)
        return stop, broker._read_line(client, broker.MAX_RESPONSE), reader

    def test_read_uses_fixed_reader(self):
        stop, response, reader = self.exchange(b'{"op":"read"}\n')
        self.assertFalse(stop)
        self.assertEqual(response, {"ok": True, "kernel": "kernel\n"})
        reader.assert_called_once_with()

    def test_stop_never_reads_kernel(self):
        stop, response, reader = self.exchange(b'{"op":"stop"}\n')
        self.assertTrue(stop)
        self.assertEqual(response, {"ok": True, "stopping": True})
        reader.assert_not_called()

    def test_unknown_and_extra_fields_fail_closed(self):
        for payload in [b'{"op":"exec"}\n', b'{"op":"read","command":"touch /no"}\n',
                        b'[]\n', b'not-json\n', b'{"op":"read"}\n{"op":"stop"}\n']:
            with self.subTest(payload=payload):
                stop, response, reader = self.exchange(payload)
                self.assertFalse(stop)
                self.assertFalse(response["ok"])
                reader.assert_not_called()

    @unittest.skipIf(os.getuid() == 0, "real root peer is intentionally authorized")
    def test_other_uid_denied_by_kernel_peer_credentials(self):
        stop, response, reader = self.exchange(b'{"op":"read"}\n', allowed_uid=os.getuid() + 1)
        self.assertFalse(stop)
        self.assertFalse(response["ok"])
        self.assertIn("peer uid", response["error"])
        reader.assert_not_called()

    def test_oversize_request_rejected(self):
        stop, response, reader = self.exchange(b"x" * (broker.MAX_REQUEST + 1))
        self.assertFalse(stop)
        self.assertFalse(response["ok"])
        reader.assert_not_called()

    def test_reader_failure_does_not_stop_server(self):
        reader = Mock(side_effect=RuntimeError("dmesg unavailable"))
        stop, response, _ = self.exchange(b'{"op":"read"}\n', reader=reader)
        self.assertFalse(stop)
        self.assertFalse(response["ok"])

    def test_fixed_dmesg_argv_and_environment(self):
        result = Mock(returncode=0, stdout="sample\n")
        with patch.object(broker.subprocess, "run", return_value=result) as call:
            self.assertEqual(broker._kernel_text(), "sample\n")
        args, kwargs = call.call_args
        self.assertEqual(args, (["/usr/bin/dmesg", "--color=never"],))
        self.assertEqual(kwargs["env"], {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"})
        self.assertNotIn("shell", kwargs)

    def test_no_env_uses_original_read_only_path(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(broker, "_kernel_text", return_value="direct") as reader:
            self.assertEqual(broker.read_kernel_log(), "direct")
            reader.assert_called_once_with()

    def test_evidence_schema_and_hash_compatible(self):
        kernel = "\n".join(["normal", "  NVRM: assertion IOVAS failed  ",
                            "AER: Fatal test", "GPU fault sample", "normal end"])
        with patch.object(broker, "read_kernel_log", return_value=kernel):
            evidence = broker.driver_evidence()
        anomalies = ["NVRM: assertion IOVAS failed", "AER: Fatal test", "GPU fault sample"]
        self.assertEqual(set(evidence), {"host", "boot_id", "latest_driver_anomaly",
            "latest_driver_anomaly_sha256", "anomaly_count", "driver_anomalies_sha256", "dmesg_tail"})
        self.assertEqual(evidence["anomaly_count"], 3)
        self.assertEqual(evidence["driver_anomalies_sha256"], hashlib.sha256("\n".join(anomalies).encode()).hexdigest())
        self.assertEqual(evidence["latest_driver_anomaly_sha256"], hashlib.sha256(anomalies[-1].encode()).hexdigest())
        self.assertEqual(evidence["dmesg_tail"], kernel.splitlines())

    def test_socket_path_restricted(self):
        self.assertEqual(str(broker._socket_path("/run/route1-kernel-1024-run01.sock", 1024)),
                         "/run/route1-kernel-1024-run01.sock")
        for value in ["/tmp/x.sock", "/run/sub/x.sock", "/run/../run/route1-kernel-1024-a.sock",
                      "/run/route1-kernel-1024-../../x.sock", "/run/route1-kernel-1024-.sock"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                broker._socket_path(value, 1024)
        with self.assertRaises(ValueError):
            broker._socket_path("/run/route1-kernel-1000-run01.sock", 1024)

    def test_real_unprivileged_unix_loop_read_stop_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.sock"
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(path)); server.listen(2); server.settimeout(.1)
            info = path.lstat()
            reader = Mock(return_value="real socket\n")
            thread = threading.Thread(target=broker._serve_loop,
                args=(server, path, (info.st_dev, info.st_ino), os.getuid(), 5, reader),
                kwargs={"install_signals": False})
            thread.start()
            try:
                with patch.object(broker, "_socket_path", return_value=path):
                    response = broker._request(path, "read", expected_server_uid=os.getuid())
                    self.assertEqual(response["kernel"], "real socket\n")
                    response = broker._request(path, "stop", expected_server_uid=os.getuid())
                    self.assertTrue(response["stopping"])
                thread.join(2)
                self.assertFalse(thread.is_alive())
                self.assertFalse(path.exists())
                reader.assert_called_once_with()
            finally:
                thread.join(6)

    @unittest.skipIf(os.getuid() == 0, "requires non-root fake server")
    def test_client_rejects_nonroot_broker(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.sock"
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(str(path)); server.listen(1)
                with patch.object(broker, "_socket_path", return_value=path), self.assertRaises(PermissionError):
                    broker._request(path, "read", timeout=1)

    def test_owned_cleanup_preserves_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.sock"
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(str(path))
                info = path.lstat()
                path.unlink()
                path.write_text("replacement")
                broker._unlink_owned(path, (info.st_dev, info.st_ino))
                self.assertEqual(path.read_text(), "replacement")

    def test_lifetime_closes_idle_listener(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.sock"
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(path)); server.listen(1); server.settimeout(.01)
            info = path.lstat()
            broker._serve_loop(server, path, (info.st_dev, info.st_ino), os.getuid(), .02,
                               Mock(), install_signals=False)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
