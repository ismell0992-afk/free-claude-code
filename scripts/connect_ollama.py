"""Connect opencode + the FCC proxy to an Ollama instance.

By default it auto-discovers the Windows tailnet peer over Tailscale. With
``--local`` it targets a provider already running on this Mac (``127.0.0.1``).
Either way it verifies Ollama is reachable, refreshes the live model list into
both ``~/.config/opencode/opencode.json`` and the repo-local ``.env``, then
reminds the user to pick a model with ``/model``.

Run with:
    uv run scripts/connect_ollama.py                 # Windows PC via Tailscale
    uv run scripts/connect_ollama.py --local         # local Ollama on this Mac
    uv run scripts/connect_ollama.py --host 1.2.3.4  # explicit IP
    uv run scripts/connect_ollama.py [--local] --filter coder
    uv run scripts/connect_ollama.py --local --opus qwen2.5-coder:7b --sonnet qwen3:14b --haiku qwen3:8b
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OPENCODE_CONFIG = Path.home() / ".config" / "opencode" / "opencode.json"
FCC_ENV = REPO_ROOT / ".env"
OLLAMA_PORT = 11434

# Sane defaults for models we auto-register in opencode.json. Ollama models vary
# wildly in real context limits; opencode needs *a* value, so we pick a generous
# ceiling and let the user tighten per-model later.
DEFAULT_CONTEXT = 128000
DEFAULT_OUTPUT = 16384

# Named connection profiles + the remembered "last" target live here (user-global,
# never committed — they can contain machine-specific IPs and pins).
PROFILES_PATH = Path.home() / ".config" / "opencode" / "ollama_profiles.json"


def _load_profiles() -> dict:
    if PROFILES_PATH.is_file():
        try:
            return json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _save_profiles(data: dict) -> None:
    PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILES_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _apply_profile(args: argparse.Namespace, prof: dict) -> None:
    """Copy a saved profile's fields onto ``args``."""

    args.local = bool(prof.get("local", False))
    args.host = prof.get("host", "") or ""
    args.opus = prof.get("opus")
    args.sonnet = prof.get("sonnet")
    args.haiku = prof.get("haiku")
    args.model = prof.get("model")
    args.filter = prof.get("filter", "") or ""


def _target_descriptor(args: argparse.Namespace, ip: str) -> dict:
    """Capture the resolved connection so it can be reused later via ``--last``."""

    return {
        "local": args.local,
        "host": ip,
        "opus": args.opus,
        "sonnet": args.sonnet,
        "haiku": args.haiku,
        "model": args.model,
        "filter": args.filter,
    }


def discover_windows_ip() -> str:
    """Return the IPv4 Tailscale address of the online Windows peer."""

    try:
        raw = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            "Could not run `tailscale status`. Is Tailscale installed and on PATH?"
        ) from exc

    try:
        status = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("`tailscale status --json` returned invalid JSON.") from exc

    peers = list(status.get("Peer", {}).values())
    for peer in peers:
        if peer.get("OS") == "windows" and peer.get("Online"):
            for addr in peer.get("TailscaleIPs", []):
                if ":" not in addr:
                    return addr
    hostnames = ", ".join(
        str(p.get("HostName", "?")) for p in peers if p.get("OS") == "windows"
    )
    raise RuntimeError(
        "No online Windows peer found on the tailnet."
        + (f" Found offline Windows peer(s): {hostnames}." if hostnames else "")
        + " Make sure the Windows PC is signed into Tailscale and Ollama is running."
    )


def fetch_models(ip: str) -> list[str]:
    """Return the sorted list of Ollama model names reachable at ``ip``."""

    url = f"http://{ip}:{OLLAMA_PORT}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            payload = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(
            f"Ollama at {ip}:{OLLAMA_PORT} is not reachable: {exc}"
        ) from exc

    models = [m["name"] for m in payload.get("models", []) if "name" in m]
    return sorted(models)


def start_local_ollama() -> bool:
    """Best-effort: start ``ollama serve`` in the background and wait for it.

    Returns True if Ollama becomes reachable on localhost, else False.
    """

    if not shutil.which("ollama"):
        return False
    print("  Starting local Ollama (ollama serve)...")
    subprocess.Popen(
        ["ollama", "serve"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        try:
            fetch_models("127.0.0.1")
        except RuntimeError:
            time.sleep(0.5)
        else:
            print("  Local Ollama is up.")
            return True
    return False


def fetch_models_ensuring(ip: str, local: bool) -> list[str]:
    """Fetch models, auto-starting a local Ollama when ``local`` and unreachable."""

    try:
        return fetch_models(ip)
    except RuntimeError:
        if local and start_local_ollama():
            return fetch_models(ip)
        raise


def fetch_model_details(ip: str, name: str) -> tuple[int, list[str]]:
    """Return ``(context_length, capabilities)`` for one model via ``/api/show``.

    Falls back to a conservative ``(8192, [])`` when the endpoint is unavailable
    (e.g. a model that vanished mid-scan) so the rest of the connect still works.
    """

    url = f"http://{ip}:{OLLAMA_PORT}/api/show"
    req = urllib.request.Request(
        url,
        data=json.dumps({"name": name}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return (8192, [])
    context = 8192
    for key, value in data.get("model_info", {}).items():
        if key.endswith(".context_length") and isinstance(value, int):
            context = value
            break
    return (context, list(data.get("capabilities", []) or []))


def fetch_all_details(ip: str, models: list[str]) -> dict[str, tuple[int, list[str]]]:
    """Fetch ``(context, capabilities)`` for every model in parallel."""

    details: dict[str, tuple[int, list[str]]] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_model_details, ip, m): m for m in models}
        for future in futures:
            name = futures[future]
            try:
                details[name] = future.result()
            except Exception:
                details[name] = (8192, [])
    return details


def _rebuild_opencode_models(
    existing: dict, models: list[str], details: dict[str, tuple[int, list[str]]] | None = None
) -> dict:
    """Preserve custom entries; fill in any newly discovered models.

    When ``details`` (name -> (context_length, capabilities)) is supplied, the
    real context length is used instead of the flat default.
    """

    details = details or {}
    rebuilt: dict[str, object] = {}
    for name in models:
        prior = existing.get(name)
        if isinstance(prior, dict):
            rebuilt[name] = prior
            continue
        context = details.get(name, (DEFAULT_CONTEXT, []))[0] or DEFAULT_CONTEXT
        rebuilt[name] = {
            "name": name,
            "limit": {"context": context, "output": min(context, DEFAULT_OUTPUT)},
        }
    return rebuilt


def update_opencode(
    ip: str,
    models: list[str],
    details: dict[str, tuple[int, list[str]]] | None = None,
) -> bool:
    """Point opencode's Ollama provider at the target and refresh its models."""

    if not OPENCODE_CONFIG.is_file():
        print(
            f"  ! opencode config not found at {OPENCODE_CONFIG}; skipping opencode update."
        )
        return False

    with OPENCODE_CONFIG.open(encoding="utf-8") as fh:
        config = json.load(fh)

    provider = config.setdefault("provider", {}).setdefault("ollama", {})
    provider["options"] = dict(provider.get("options", {}))
    provider["options"]["baseURL"] = f"http://{ip}:{OLLAMA_PORT}/v1"
    provider["models"] = _rebuild_opencode_models(
        provider.get("models", {}) or {}, models, details
    )

    tmp = OPENCODE_CONFIG.with_suffix(OPENCODE_CONFIG.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=4)
        fh.write("\n")
    tmp.replace(OPENCODE_CONFIG)
    return True


def _set_env_line(lines: list[str], key: str, value: str) -> list[str]:
    """Return ``lines`` with ``key`` set to ``value``, preserving comments/order."""

    target = f'{key}="{value}"'
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
            lines[i] = target
            return lines
    lines.append(target)
    return lines


def _parse_billions(name: str) -> int:
    """Return the largest ``NNb`` parameter count in a model name, else 0."""

    sizes = [int(m) for m in re.findall(r"(\d+)\s*[bB]", name)]
    return max(sizes) if sizes else 0


# Target model size (billions of params) per Claude tier. On a local Mac we keep
# Opus modest (16GB RAM); on a GPU box we can afford the most capable model.
TIER_TARGET_B = {
    "MODEL": 14,
    "MODEL_OPUS": 14,
    "MODEL_SONNET": 14,
    "MODEL_HAIKU": 8,
}
TIER_TARGET_B_GPU = {
    "MODEL": 32,
    "MODEL_OPUS": 32,
    "MODEL_SONNET": 14,
    "MODEL_HAIKU": 8,
}


def _best_model_for_tier(
    tier: str,
    models: list[str],
    lightweight: bool,
    capabilities: dict[str, list[str]] | None = None,
) -> str:
    """Pick the available model whose size is closest to the tier's target.

    On a lightweight (local Mac) target, ``-cloud`` models are excluded because
    they are remote API endpoints, not local compute. When ``capabilities`` is
    supplied, tool-calling models are preferred (FCC's translation needs them).
    """

    capabilities = capabilities or {}
    pool = [m for m in models if not (lightweight and m.endswith("-cloud"))]
    if not pool:
        pool = models
    if not pool:
        return ""
    tool_ok = [m for m in pool if "tools" in capabilities.get(m, [])]
    if tool_ok:
        pool = tool_ok
    target = (TIER_TARGET_B_GPU if not lightweight else TIER_TARGET_B).get(tier, 14)
    # Tie-break: closest size, then prefer non-context-limited variants, then shorter name.
    return min(
        pool,
        key=lambda m: (abs(_parse_billions(m) - target), "ctx" in m, len(m)),
    )


def update_fcc_env(
    ip: str,
    models: list[str],
    overrides: dict[str, str] | None = None,
    capabilities: dict[str, list[str]] | None = None,
) -> bool:
    """Set ``OLLAMA_BASE_URL`` and point the Claude tier mappings at Ollama.

    Resolution order per tier: an explicit ``--opus/--sonnet/--haiku/--model``
    override wins; otherwise a current ``ollama/<name>`` that exists on the
    target is kept; otherwise the tier is remapped to the closest-size model.
    ``capabilities`` (name -> list) enables tool-calling preference and warnings.
    """

    if not FCC_ENV.is_file():
        print(f"  ! FCC .env not found at {FCC_ENV}; skipping FCC update.")
        return False

    overrides = overrides or {}
    capabilities = capabilities or {}
    lightweight = ip == "127.0.0.1"

    with FCC_ENV.open(encoding="utf-8") as fh:
        lines = fh.readlines()

    lines = _set_env_line(lines, "OLLAMA_BASE_URL", f"http://{ip}:{OLLAMA_PORT}")

    for key in ("MODEL", "MODEL_OPUS", "MODEL_SONNET", "MODEL_HAIKU"):
        explicit = overrides.get(key)
        if explicit:
            name = (
                explicit[len("ollama/") :]
                if explicit.startswith("ollama/")
                else explicit
            )
            if name in models:
                if "tools" not in capabilities.get(name, []):
                    print(f"  ! {key}: '{name}' may lack tool calling; FCC needs it.")
                lines = _set_env_line(lines, key, f"ollama/{name}")
                continue
            print(f"  ! {key}: '{name}' not on target; ignoring override.")

        current = None
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(f"{key}="):
                current = stripped.split("=", 1)[1].strip().strip('"')
                break

        if current and current.startswith("ollama/"):
            name = current[len("ollama/") :]
            if name in models:
                if "tools" not in capabilities.get(name, []):
                    print(f"  ! {key}: '{name}' may lack tool calling; FCC needs it.")
                continue
            best = _best_model_for_tier(key, models, lightweight, capabilities)
            lines = _set_env_line(lines, key, f"ollama/{best}")
            print(f"  {key}: '{name}' not on target; remapped to '{best}'.")
        else:
            best = _best_model_for_tier(key, models, lightweight, capabilities)
            if best:
                lines = _set_env_line(lines, key, f"ollama/{best}")
                print(f"  {key}: set to '{best}' (target default).")

    with FCC_ENV.open("w", encoding="utf-8") as fh:
        fh.writelines(lines)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local",
        action="store_true",
        help="Target a local Ollama on this Mac (127.0.0.1) instead of Tailscale.",
    )
    parser.add_argument(
        "--host",
        default="",
        help="Connect to an explicit Ollama host IP (overrides --local / discovery).",
    )
    parser.add_argument(
        "--filter",
        default="",
        help="Only keep models whose name contains this substring (case-insensitive).",
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
    args = parser.parse_args()

    profiles = _load_profiles()
    if args.profile:
        prof = profiles.get(args.profile)
        if not prof:
            print(f"  ERROR: no profile named '{args.profile}'.", file=sys.stderr)
            return 1
        _apply_profile(args, prof)
        print(f"==> Loaded profile '{args.profile}'.")
    elif args.last:
        prof = profiles.get("last")
        if prof:
            _apply_profile(args, prof)
            print("==> Reusing last target.")

    if args.host:
        ip = args.host
        print(f"==> Using explicit host {ip}...")
    elif args.local:
        ip = "127.0.0.1"
        print("==> Targeting local Ollama on this Mac (127.0.0.1)...")
    else:
        print("==> Discovering Windows PC on the tailnet...")
        try:
            ip = discover_windows_ip()
        except RuntimeError as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"  Windows PC found at {ip} (Tailscale).")

    print("==> Checking Ollama...")
    try:
        models = fetch_models_ensuring(ip, args.local)
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
            print(
                f"  ERROR: No models match filter '{args.filter}'.",
                file=sys.stderr,
            )
            return 1
        models = keep
        print(f"  Filtered to {len(models)} model(s) matching '{args.filter}'.")

    print("==> Reading model details (context + capabilities)...")
    details = fetch_all_details(ip, models)
    capabilities = {name: caps for name, (_, caps) in details.items()}

    print("==> Updating opencode config...")
    opencode_ok = update_opencode(ip, models, details)
    if opencode_ok:
        print(f"  Updated {OPENCODE_CONFIG}")

    print("==> Updating FCC .env...")
    overrides = {
        "MODEL": args.model,
        "MODEL_OPUS": args.opus,
        "MODEL_SONNET": args.sonnet,
        "MODEL_HAIKU": args.haiku,
    }
    fcc_ok = update_fcc_env(ip, models, overrides, capabilities)
    if fcc_ok:
        print(f"  Updated {FCC_ENV}")

    profiles["last"] = _target_descriptor(args, ip)
    if args.save:
        profiles[args.save] = _target_descriptor(args, ip)
        print(f"  Saved profile '{args.save}'.")
    _save_profiles(profiles)

    print("\n=== Done ===")
    print(f"Ollama: http://{ip}:{OLLAMA_PORT}")
    if opencode_ok:
        print("  opencode: Ollama provider + model list refreshed (real context windows).")
        print("  Next: run /model in opencode to pick a model (restart if needed).")
    if fcc_ok:
        print("  FCC: OLLAMA_BASE_URL set; Claude tiers map to your Ollama models.")
    print("\nAvailable models:")
    for name in models:
        flags = " [tools]" if "tools" in capabilities.get(name, []) else ""
        print(f"  - ollama/{name}{flags}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
