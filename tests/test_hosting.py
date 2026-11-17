"""Verify real deployment entry points, assigned ports, and health checks."""
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.error import URLError
from urllib.request import Request, urlopen

from src.data import EXAMPLE_RECORD, ROOT
from src.hosting import health_url, hosting_settings, server_command
from src.pipeline import MODEL_PATH


class HostingConfigurationTests(unittest.TestCase):
    def test_default_is_dashboard(self):
        self.assertEqual(hosting_settings({}), ("dashboard", 8501))
        self.assertIn("streamlit", server_command({}))
        self.assertTrue(health_url({}).endswith(":8501/_stcore/health"))

    def test_api_uses_its_own_health_path(self):
        env = {"APP_MODE": "api", "PORT": "10000"}
        self.assertIn("uvicorn", server_command(env))
        self.assertIn("10000", server_command(env))
        self.assertEqual(health_url(env), "http://127.0.0.1:10000/health")

    def test_invalid_mode_and_port_fail_before_binding(self):
        for env in ({"APP_MODE": "invalid"}, {"PORT": "abc"}, {"PORT": "0"}, {"PORT": "65536"}):
            with self.subTest(environment=env), self.assertRaises(ValueError):
                server_command(env)


@unittest.skipUnless(MODEL_PATH.exists(), "Train a model before deployment smoke tests.")
class HostingSmokeTests(unittest.TestCase):
    def _run_service(self, mode):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        token = secrets.token_urlsafe(32)
        environment = {**os.environ, "APP_MODE": mode, "PORT": str(port),
                       "APNEA_API_KEY": token}
        with tempfile.TemporaryFile(mode="w+") as log:
            process = subprocess.Popen([sys.executable, "-m", "src.hosting"], cwd=ROOT,
                                       env=environment, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        log.seek(0)
                        self.fail(f"{mode} exited before startup:\n{log.read()}")
                    try:
                        with urlopen(health_url(environment), timeout=1) as response:
                            self.assertEqual(response.status, 200)
                        break
                    except (URLError, TimeoutError):
                        time.sleep(0.1)
                else:
                    log.seek(0)
                    self.fail(f"{mode} did not become healthy:\n{log.read()}")
                check = subprocess.run([sys.executable, "-m", "src.hosting", "healthcheck"],
                                       env=environment, cwd=ROOT, capture_output=True, text=True, timeout=10)
                self.assertEqual(check.returncode, 0, check.stderr)
                if mode == "api":
                    request = Request(f"http://127.0.0.1:{port}/v1/screen",
                                      data=json.dumps(EXAMPLE_RECORD).encode(),
                                      headers={"Content-Type": "application/json",
                                               "Authorization": f"Bearer {token}"})
                    with urlopen(request, timeout=5) as response:
                        self.assertIn(json.load(response)["status"], ("lower_signal", "raised_signal"))
                else:
                    with urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
                        self.assertIn(b"<html", response.read().lower())
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

    def test_dashboard_on_host_assigned_port(self):
        self._run_service("dashboard")

    def test_api_on_host_assigned_port(self):
        self._run_service("api")
