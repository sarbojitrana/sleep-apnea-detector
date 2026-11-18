"""Container entry point shared by the dashboard and device API."""
import os
import sys
from urllib.request import urlopen

from src.data import ROOT
from src.combined import load_combined_model


def hosting_settings(environment):
    mode = environment.get("APP_MODE", "dashboard")
    if mode not in ("dashboard", "api"):
        raise ValueError("APP_MODE must be dashboard or api.")
    try:
        port = int(environment.get("PORT", "8501" if mode == "dashboard" else "8000"))
    except (TypeError, ValueError) as error:
        raise ValueError("PORT must be an integer between 1 and 65535.") from error
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be an integer between 1 and 65535.")
    return mode, port


def server_command(environment):
    mode, port = hosting_settings(environment)
    if mode == "api":
        return [sys.executable, "-m", "uvicorn", "api:app", "--host", "0.0.0.0",
                "--port", str(port), "--no-access-log"]
    return [sys.executable, "-m", "streamlit", "run", str(ROOT / "ui.py"),
            "--server.address", "0.0.0.0", "--server.port", str(port),
            "--server.headless", "true", "--server.fileWatcherType", "none",
            "--browser.gatherUsageStats", "false"]


def health_url(environment):
    mode, port = hosting_settings(environment)
    path = "/health" if mode == "api" else "/_stcore/health"
    return f"http://127.0.0.1:{port}{path}"


def main():
    if sys.argv[1:] == ["healthcheck"]:
        with urlopen(health_url(os.environ), timeout=3) as response:
            if response.status != 200:
                raise RuntimeError("Service health check failed.")
        return
    if sys.argv[1:]:
        raise SystemExit("Usage: python -m src.hosting [healthcheck]")
    mode, _ = hosting_settings(os.environ)
    if mode == "dashboard":
        # Fail startup before declaring the Streamlit server healthy if the model is unusable.
        load_combined_model()
    os.chdir(ROOT)
    command = server_command(os.environ)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
