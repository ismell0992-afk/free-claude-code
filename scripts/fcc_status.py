"""Print the current FCC + Ollama connection status (read-only inspection).

Shows the configured ``OLLAMA_BASE_URL``, whether ``fcc-server`` is healthy, and
whether each Claude tier (``MODEL*``) actually exists on the target Ollama.

Run with: ``uv run scripts/fcc_status.py``
"""

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FCC_ENV = REPO_ROOT / ".env"
OLLAMA_PORT = 11434
DEFAULT_PORT = 8082


def _env_val(lines: list[str], key: str) -> str:
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{key}="):
            return stripped.split("=", 1)[1].strip().strip('"')
    return ""


def _proxy_port() -> int:
    try:
        from config.settings import get_settings

        return get_settings().port or DEFAULT_PORT
    except Exception:
        return DEFAULT_PORT


def _health_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return 200 <= resp.getcode() < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def main() -> int:
    if not FCC_ENV.is_file():
        print("  ! FCC .env not found; run a /connect-* command first.")
        return 1

    lines = FCC_ENV.read_text(encoding="utf-8").splitlines()
    base = _env_val(lines, "OLLAMA_BASE_URL")
    print(f"OLLAMA_BASE_URL: {base or '(unset)'}")

    port = _proxy_port()
    if _health_ok(f"http://127.0.0.1:{port}/health"):
        print(f"fcc-server:       healthy (:{port})")
    else:
        print(f"fcc-server:       not running (expected :{port})")

    host = "127.0.0.1"
    if base:
        match = re.match(r"https?://([^:/]+)", base)
        if match:
            host = match.group(1)
    target_models: list[str] = []
    try:
        with urllib.request.urlopen(
            f"http://{host}:{OLLAMA_PORT}/api/tags", timeout=5
        ) as resp:
            target_models = [
                m["name"] for m in json.loads(resp.read()).get("models", [])
            ]
        print(f"Ollama ({host}):   reachable, {len(target_models)} model(s)")
    except (urllib.error.URLError, OSError, ValueError):
        print(f"Ollama ({host}):   not reachable")

    print("Tier mappings:")
    for key in ("MODEL", "MODEL_OPUS", "MODEL_SONNET", "MODEL_HAIKU"):
        val = _env_val(lines, key)
        if not val:
            print(f"  {key}: (unset)")
            continue
        name = val[len("ollama/") :] if val.startswith("ollama/") else val
        mark = "ok" if name in target_models else "MISSING on target"
        print(f"  {key}: {val}  [{mark}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
