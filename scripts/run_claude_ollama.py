"""One-shot launcher: connect FCC to Ollama, start the proxy, launch Claude Code.

Wraps ``connect_ollama.py`` to configure the repo ``.env`` + opencode config,
then starts ``fcc-server`` (background) and drops you into ``fcc-claude``.

Run with:
    uv run scripts/run_claude_ollama.py                 # Windows PC via Tailscale
    uv run scripts/run_claude_ollama.py --local         # local Ollama on this Mac
    uv run scripts/run_claude_ollama.py --host 1.2.3.4  # explicit IP
    uv run scripts/run_claude_ollama.py [--local] --filter coder

Flags ``--no-server`` / ``--no-client`` skip those steps (used by tests / when
you prefer to manage the server yourself).
"""

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import connect_ollama

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PORT = 8082
HEALTH_PATH = "/health"
HEALTH_TIMEOUT = 30.0
HEALTH_POLL = 0.5


def resolve_ip(args: argparse.Namespace) -> str:
    if args.host:
        return args.host
    if args.local:
        return "127.0.0.1"
    return connect_ollama.discover_windows_ip()


def _proxy_port() -> int:
    try:
        from config.settings import get_settings

        return get_settings().port or DEFAULT_PORT
    except Exception:
        return DEFAULT_PORT


def _health_ok(port: int) -> bool:
    url = f"http://127.0.0.1:{port}{HEALTH_PATH}"
    try:
        with urllib.request.urlopen(url, timeout=1.0) as resp:
            return 200 <= resp.getcode() < 300
    except urllib.error.URLError, OSError, ValueError:
        return False


def ensure_server(port: int) -> subprocess.Popen[bytes] | None:
    """Start ``fcc-server`` in the background if the health check is not already up."""

    if _health_ok(port):
        print(f"  fcc-server already healthy on port {port}.")
        return None

    print(f"==> Starting fcc-server on port {port}...")
    proc = subprocess.Popen(
        ["uv", "run", "fcc-server"],
        cwd=REPO_ROOT,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + HEALTH_TIMEOUT
    while time.monotonic() < deadline:
        if _health_ok(port):
            print(f"  fcc-server is up on http://127.0.0.1:{port}.")
            return proc
        time.sleep(HEALTH_POLL)
    print(
        "  WARNING: fcc-server did not become healthy in time. "
        "Start it manually with: uv run fcc-server",
        file=sys.stderr,
    )
    return proc


def launch_client() -> int:
    """Launch ``fcc-claude`` in the foreground (Ctrl-C / exit returns here)."""

    print("==> Launching Claude Code via fcc-claude...")
    proc = subprocess.Popen(["uv", "run", "fcc-claude"], cwd=REPO_ROOT)
    return proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local", action="store_true", help="Target local Ollama on this Mac."
    )
    parser.add_argument("--host", default="", help="Explicit Ollama host IP.")
    parser.add_argument(
        "--filter", default="", help="Keep only models containing this substring."
    )
    parser.add_argument(
        "--opus", default=None, help="Pin the Opus tier to this Ollama model."
    )
    parser.add_argument(
        "--sonnet", default=None, help="Pin the Sonnet tier to this Ollama model."
    )
    parser.add_argument(
        "--haiku", default=None, help="Pin the Haiku tier to this Ollama model."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Pin the plain fallback MODEL to this Ollama model.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Load a saved connection profile by name (target + pins + filter).",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Save the current connection (target + pins + filter) as a named profile.",
    )
    parser.add_argument(
        "--last",
        action="store_true",
        help="Reuse the last connected target (falls back to Windows if none saved).",
    )
    parser.add_argument(
        "--no-server", action="store_true", help="Do not start fcc-server."
    )
    parser.add_argument(
        "--no-client", action="store_true", help="Do not launch fcc-claude."
    )
    args = parser.parse_args()

    profiles = connect_ollama._load_profiles()
    if args.profile:
        prof = profiles.get(args.profile)
        if not prof:
            print(f"  ERROR: no profile named '{args.profile}'.", file=sys.stderr)
            return 1
        connect_ollama._apply_profile(args, prof)
        print(f"==> Loaded profile '{args.profile}'.")
    elif args.last:
        prof = profiles.get("last")
        if prof:
            connect_ollama._apply_profile(args, prof)
            print("==> Reusing last target.")

    print("==> Resolving Ollama host...")
    try:
        ip = resolve_ip(args)
    except RuntimeError as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"  Ollama host: {ip}")

    print("==> Fetching models...")
    try:
        models = connect_ollama.fetch_models_ensuring(ip, args.local)
    except RuntimeError as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        return 1
    if not models:
        print("  ERROR: Ollama responded but listed no models.", file=sys.stderr)
        return 1
    print(f"  Ollama reachable with {len(models)} model(s).")

    if args.filter:
        keep = [m for m in models if args.filter.lower() in m.lower()]
        if not keep:
            print(f"  ERROR: No models match filter '{args.filter}'.", file=sys.stderr)
            return 1
        models = keep
        print(f"  Filtered to {len(models)} model(s) matching '{args.filter}'.")

    print("==> Reading model details (context + capabilities)...")
    details = connect_ollama.fetch_all_details(ip, models)
    capabilities = {name: caps for name, (_, caps) in details.items()}

    print("==> Updating opencode config + FCC .env...")
    overrides = {
        "MODEL": args.model,
        "MODEL_OPUS": args.opus,
        "MODEL_SONNET": args.sonnet,
        "MODEL_HAIKU": args.haiku,
    }
    connect_ollama.update_opencode(ip, models, details)
    connect_ollama.update_fcc_env(ip, models, overrides, capabilities)

    profiles["last"] = connect_ollama._target_descriptor(args, ip)
    if args.save:
        profiles[args.save] = connect_ollama._target_descriptor(args, ip)
        print(f"  Saved profile '{args.save}'.")
    connect_ollama._save_profiles(profiles)

    if not args.no_server:
        ensure_server(_proxy_port())

    if not args.no_client:
        return launch_client()

    print("\n=== Ready ===")
    print("  Start the proxy if needed: uv run fcc-server")
    print("  Then launch Claude Code:   uv run fcc-claude")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
