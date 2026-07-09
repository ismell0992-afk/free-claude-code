"""Tests for scripts/connect_windows_ollama.py (network + subprocess mocked)."""

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "connect_ollama.py"

spec = importlib.util.spec_from_file_location("connect_windows_ollama", SCRIPT)
assert spec is not None
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


TAILSCALE_ONLINE = {
    "Peer": {
        "k1": {
            "HostName": "DESKTOP-CRT1K2Q",
            "OS": "windows",
            "Online": True,
            "TailscaleIPs": ["100.98.79.73", "fd7a:115c:a1e0::6839:4f4a"],
        },
        "k2": {
            "HostName": "mac",
            "OS": "macOS",
            "Online": True,
            "TailscaleIPs": ["100.67.1.1"],
        },
    }
}
TAILSCALE_OFFLINE = {
    "Peer": {
        "k1": {
            "HostName": "DESKTOP",
            "OS": "windows",
            "Online": False,
            "TailscaleIPs": ["100.98.1.1"],
        }
    }
}
TAILSCALE_NONE = {
    "Peer": {
        "k1": {
            "HostName": "mac",
            "OS": "macOS",
            "Online": True,
            "TailscaleIPs": ["100.67.1.1"],
        }
    }
}


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _run_result(stdout: str):
    class _R:
        stdout: str

    r = _R()
    r.stdout = stdout
    return r


def test_discover_windows_ip_success(monkeypatch):
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *a, **k: _run_result(json.dumps(TAILSCALE_ONLINE)),
    )
    assert mod.discover_windows_ip() == "100.98.79.73"


def test_discover_windows_ip_offline(monkeypatch):
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *a, **k: _run_result(json.dumps(TAILSCALE_OFFLINE)),
    )
    with pytest.raises(RuntimeError, match="No online Windows peer"):
        mod.discover_windows_ip()


def test_discover_windows_ip_none(monkeypatch):
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *a, **k: _run_result(json.dumps(TAILSCALE_NONE)),
    )
    with pytest.raises(RuntimeError, match="No online Windows peer"):
        mod.discover_windows_ip()


def test_fetch_models(monkeypatch):
    payload = json.dumps(
        {"models": [{"name": "qwen3:8b"}, {"name": "qwen2.5-coder:32b"}]}
    ).encode()

    def fake_urlopen(url, timeout=0):
        return _FakeResponse(payload)

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    assert mod.fetch_models("100.98.79.73") == ["qwen2.5-coder:32b", "qwen3:8b"]


def test_fetch_models_unreachable(monkeypatch):
    def fake_urlopen(url, timeout=0):
        raise mod.urllib.error.URLError("boom")

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="not reachable"):
        mod.fetch_models("100.98.79.73")


def test_fetch_models_ensuring_boots_local(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(url, timeout=0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise mod.urllib.error.URLError("down")
        return _FakeResponse(json.dumps({"models": [{"name": "qwen3:8b"}]}).encode())

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    started = []
    monkeypatch.setattr(
        mod.subprocess,
        "Popen",
        lambda *a, **k: started.append(a) or object(),
    )
    models = mod.fetch_models_ensuring("127.0.0.1", local=True)
    assert models == ["qwen3:8b"]
    assert started and started[0][0][:2] == ["ollama", "serve"]


def test_fetch_models_ensuring_no_boot_when_remote(monkeypatch):
    def fake_urlopen(url, timeout=0):
        raise mod.urllib.error.URLError("down")

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    started = []
    monkeypatch.setattr(
        mod.subprocess,
        "Popen",
        lambda *a, **k: started.append(a) or object(),
    )
    with pytest.raises(RuntimeError, match="not reachable"):
        mod.fetch_models_ensuring("100.98.79.73", local=False)
    assert not started


def test_update_opencode(monkeypatch, tmp_path):
    cfg = tmp_path / "opencode.json"
    cfg.write_text(
        json.dumps(
            {
                "provider": {
                    "ollama": {"options": {"baseURL": "http://127.0.0.1:11434/v1"}}
                }
            }
        )
    )
    monkeypatch.setattr(mod, "OPENCODE_CONFIG", cfg)
    ok = mod.update_opencode("100.98.79.73", ["qwen3:8b", "qwen2.5-coder:32b"])
    assert ok
    data = json.loads(cfg.read_text())
    ollama = data["provider"]["ollama"]
    assert ollama["options"]["baseURL"] == "http://100.98.79.73:11434/v1"
    assert set(ollama["models"]) == {"qwen3:8b", "qwen2.5-coder:32b"}
    assert ollama["models"]["qwen3:8b"]["limit"]["context"] == mod.DEFAULT_CONTEXT


def test_update_opencode_preserves_custom(monkeypatch, tmp_path):
    cfg = tmp_path / "opencode.json"
    cfg.write_text(
        json.dumps(
            {
                "provider": {
                    "ollama": {
                        "models": {
                            "qwen3:8b": {
                                "name": "Qwen3 8B",
                                "limit": {"context": 4096, "output": 4096},
                            }
                        }
                    }
                }
            }
        )
    )
    monkeypatch.setattr(mod, "OPENCODE_CONFIG", cfg)
    mod.update_opencode("100.98.79.73", ["qwen3:8b"])
    data = json.loads(cfg.read_text())
    assert data["provider"]["ollama"]["models"]["qwen3:8b"]["limit"]["context"] == 4096


def test_update_fcc_env(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        'OLLAMA_BASE_URL="http://localhost:11434"\n'
        'MODEL="nvidia_nim/nvidia/nemotron"\n'
        'MODEL_OPUS="ollama/qwen2.5-coder:32b"\n'
    )
    monkeypatch.setattr(mod, "FCC_ENV", env)
    ok = mod.update_fcc_env("100.98.79.73", ["qwen2.5-coder:32b", "qwen3:8b"])
    assert ok
    text = env.read_text()
    assert 'OLLAMA_BASE_URL="http://100.98.79.73:11434"' in text
    assert 'MODEL="ollama/qwen2.5-coder:32b"' in text
    assert 'MODEL_OPUS="ollama/qwen2.5-coder:32b"' in text


def test_update_fcc_env_remaps_missing(monkeypatch, tmp_path):
    # Local Mac list lacks the Windows-only 32b coder; Opus/Sonnet/Haiku remap.
    env = tmp_path / ".env"
    env.write_text(
        'MODEL_OPUS="ollama/qwen2.5-coder:32b-instruct-q4_K_M"\n'
        'MODEL_SONNET="ollama/qwen3:14b"\n'
        'MODEL_HAIKU="ollama/qwen3:8b"\n'
    )
    monkeypatch.setattr(mod, "FCC_ENV", env)
    models = ["qwen2.5-coder:7b", "qwen3-coder:30b", "qwen3:14b", "qwen3:8b"]
    ok = mod.update_fcc_env("127.0.0.1", models)
    assert ok
    text = env.read_text()
    assert 'MODEL_OPUS="ollama/qwen3:14b"' in text  # closest-to-14B on Mac
    assert 'MODEL_SONNET="ollama/qwen3:14b"' in text  # already present, kept
    assert 'MODEL_HAIKU="ollama/qwen3:8b"' in text


def test_update_fcc_env_explicit_override(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text('MODEL_OPUS="ollama/qwen2.5-coder:32b-instruct-q4_K_M"\n')
    monkeypatch.setattr(mod, "FCC_ENV", env)
    models = ["qwen2.5-coder:7b", "qwen3:14b", "qwen3:8b"]
    ok = mod.update_fcc_env("127.0.0.1", models, overrides={"MODEL_OPUS": "qwen3:14b"})
    assert ok
    assert 'MODEL_OPUS="ollama/qwen3:14b"' in env.read_text()


def test_best_model_for_tier():
    models = ["qwen2.5-coder:7b", "qwen3-coder:30b", "qwen3:14b", "qwen3:8b"]
    # Closest-to-target sizing: Opus Mac (14B) -> 14b; Haiku (8B) -> 8b;
    # Opus GPU (32B) -> 30b (closest available).
    assert mod._best_model_for_tier("MODEL_OPUS", models, True) == "qwen3:14b"
    assert mod._best_model_for_tier("MODEL_HAIKU", models, True) == "qwen3:8b"
    assert mod._best_model_for_tier("MODEL_OPUS", models, False) == "qwen3-coder:30b"
    # Cloud models are excluded on the lightweight target.
    cloud = ["qwen2.5-coder:32b", "qwen3-coder:480b-cloud", "qwen3:14b", "qwen3:8b"]
    assert mod._best_model_for_tier("MODEL_OPUS", cloud, True) == "qwen3:14b"
    assert mod._best_model_for_tier("MODEL_OPUS", cloud, False) == "qwen2.5-coder:32b"


def test_main_local_uses_localhost(monkeypatch, capsys):
    captured = {}

    def fake_fetch(ip: str) -> list[str]:
        captured["ip"] = ip
        return ["qwen3:8b"]

    monkeypatch.setattr(mod, "fetch_models", fake_fetch)
    monkeypatch.setattr(mod, "fetch_all_details", lambda ip, models: {})
    monkeypatch.setattr(mod, "update_opencode", lambda ip, models, details=None: True)
    monkeypatch.setattr(
        mod, "update_fcc_env", lambda ip, models, overrides=None, capabilities=None: True
    )
    monkeypatch.setattr("sys.argv", ["connect_ollama.py", "--local"])

    assert mod.main() == 0
    assert captured["ip"] == "127.0.0.1"
    assert "127.0.0.1" in capsys.readouterr().out


def test_main_host_override(monkeypatch, capsys):
    captured: dict[str, object] = {}

    def fake_fetch(ip: str) -> list[str]:
        captured["ip"] = ip
        return ["qwen3:8b"]

    monkeypatch.setattr(mod, "fetch_models", fake_fetch)
    monkeypatch.setattr(mod, "fetch_all_details", lambda ip, models: {})
    monkeypatch.setattr(mod, "update_opencode", lambda ip, models, details=None: True)
    monkeypatch.setattr(
        mod, "update_fcc_env", lambda ip, models, overrides=None, capabilities=None: True
    )
    monkeypatch.setattr("sys.argv", ["connect_ollama.py", "--host", "10.0.0.5"])

    assert mod.main() == 0
    assert captured["ip"] == "10.0.0.5"


def test_fetch_model_details(monkeypatch):
    payload = json.dumps(
        {
            "model_info": {"qwen3.context_length": 40960},
            "capabilities": ["completion", "tools", "thinking"],
        }
    ).encode()

    def fake_urlopen(req, timeout=0):
        return _FakeResponse(payload)

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    context, caps = mod.fetch_model_details("127.0.0.1", "qwen3:14b")
    assert context == 40960
    assert "tools" in caps


def test_update_opencode_uses_real_context(monkeypatch, tmp_path):
    cfg = tmp_path / "opencode.json"
    cfg.write_text(json.dumps({"provider": {"ollama": {"options": {"baseURL": "x"}}}}))
    monkeypatch.setattr(mod, "OPENCODE_CONFIG", cfg)
    ok = mod.update_opencode(
        "127.0.0.1", ["qwen3:14b"], details={"qwen3:14b": (40960, ["tools"])}
    )
    assert ok
    data = json.loads(cfg.read_text())
    model = data["provider"]["ollama"]["models"]["qwen3:14b"]
    assert model["limit"]["context"] == 40960
    assert model["limit"]["output"] == min(40960, mod.DEFAULT_OUTPUT)


def test_best_model_for_tier_prefers_tools():
    models = ["nano:1b", "coder:7b"]
    caps = {"nano:1b": [], "coder:7b": ["tools"]}
    # Opus targets ~14B, but tool-capable models are preferred regardless of size.
    assert mod._best_model_for_tier("MODEL_OPUS", models, True, caps) == "coder:7b"
    # Without capability info, falls back to closest size.
    assert mod._best_model_for_tier("MODEL_OPUS", models, True, {}) == "coder:7b" or "nano:1b"


def test_profiles_roundtrip(monkeypatch, tmp_path):
    prof_file = tmp_path / "profiles.json"
    monkeypatch.setattr(mod, "PROFILES_PATH", prof_file)

    args = argparse.Namespace(
        local=True, host="", opus="a", sonnet="b", haiku="c", model="d", filter="f"
    )
    desc = mod._target_descriptor(args, "127.0.0.1")
    profiles = {"last": desc, "work": desc}
    mod._save_profiles(profiles)
    assert mod._load_profiles() == profiles

    loaded = mod._load_profiles()["work"]
    applied = argparse.Namespace()
    mod._apply_profile(applied, loaded)
    assert applied.local is True
    assert applied.opus == "a"
    assert applied.filter == "f"
