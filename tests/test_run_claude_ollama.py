"""Tests for scripts/run_claude_ollama.py (network + subprocess mocked)."""

import importlib.util
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

SPEC_C = importlib.util.spec_from_file_location(
    "connect_ollama", SCRIPTS / "connect_ollama.py"
)
assert SPEC_C is not None and SPEC_C.loader is not None
connect_ollama = importlib.util.module_from_spec(SPEC_C)
SPEC_C.loader.exec_module(connect_ollama)
sys.modules["connect_ollama"] = connect_ollama

SPEC_R = importlib.util.spec_from_file_location(
    "run_claude_ollama", SCRIPTS / "run_claude_ollama.py"
)
assert SPEC_R is not None and SPEC_R.loader is not None
run_claude_ollama = importlib.util.module_from_spec(SPEC_R)
SPEC_R.loader.exec_module(run_claude_ollama)
sys.modules["run_claude_ollama"] = run_claude_ollama


def _args(local=False, host="", filter="", no_server=False, no_client=False):
    return Namespace(
        local=local,
        host=host,
        filter=filter,
        no_server=no_server,
        no_client=no_client,
    )


def _patch_connect(monkeypatch, models=("qwen3:8b",)):
    monkeypatch.setattr(connect_ollama, "fetch_models", lambda ip: list(models))
    monkeypatch.setattr(connect_ollama, "discover_windows_ip", lambda: "100.98.79.73")
    updated = {}

    def fake_opencode(ip, models):
        updated["opencode_ip"] = ip
        return True

    def fake_fcc(ip, models, overrides=None):
        updated["fcc_ip"] = ip
        return True

    monkeypatch.setattr(connect_ollama, "update_opencode", fake_opencode)
    monkeypatch.setattr(connect_ollama, "update_fcc_env", fake_fcc)
    return updated


def test_resolve_ip_local(monkeypatch):
    monkeypatch.setattr(connect_ollama, "discover_windows_ip", lambda: "100.98.79.73")
    assert run_claude_ollama.resolve_ip(_args(local=True)) == "127.0.0.1"


def test_resolve_ip_host(monkeypatch):
    monkeypatch.setattr(connect_ollama, "discover_windows_ip", lambda: "100.98.79.73")
    assert run_claude_ollama.resolve_ip(_args(host="10.0.0.5")) == "10.0.0.5"


def test_resolve_ip_windows(monkeypatch):
    monkeypatch.setattr(connect_ollama, "discover_windows_ip", lambda: "100.98.79.73")
    assert run_claude_ollama.resolve_ip(_args()) == "100.98.79.73"


def test_ensure_server_skips_when_healthy(monkeypatch):
    monkeypatch.setattr(run_claude_ollama, "_health_ok", lambda port: True)
    started = []
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **k: started.append(a) or _fake_proc()
    )
    assert run_claude_ollama.ensure_server(8082) is None
    assert not started


def test_ensure_server_starts_and_waits(monkeypatch):
    calls = {"n": 0}

    def fake_health(port):
        calls["n"] += 1
        return calls["n"] >= 2

    monkeypatch.setattr(run_claude_ollama, "_health_ok", fake_health)
    popped = []
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **k: popped.append(a) or _fake_proc()
    )
    proc = run_claude_ollama.ensure_server(8082)
    assert proc is not None
    assert popped and popped[0][0][:2] == ["uv", "run"]


def test_main_connect_only(monkeypatch, capsys):
    updated = _patch_connect(monkeypatch)
    monkeypatch.setattr(run_claude_ollama, "ensure_server", lambda port: None)
    monkeypatch.setattr(run_claude_ollama, "launch_client", lambda: 0)
    monkeypatch.setattr(
        "sys.argv",
        ["run_claude_ollama.py", "--local", "--no-server", "--no-client"],
    )

    assert run_claude_ollama.main() == 0
    assert updated["opencode_ip"] == "127.0.0.1"
    assert updated["fcc_ip"] == "127.0.0.1"
    assert "Ready" in capsys.readouterr().out


def test_main_launch_client(monkeypatch, capsys):
    _patch_connect(monkeypatch)
    monkeypatch.setattr(run_claude_ollama, "ensure_server", lambda port: None)
    popped = []
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **k: popped.append(a) or _fake_proc()
    )
    monkeypatch.setattr("sys.argv", ["run_claude_ollama.py", "--local"])

    assert run_claude_ollama.main() == 0
    assert any("fcc-claude" in call[0] for call in popped)
    assert "Launching Claude Code" in capsys.readouterr().out


class _fake_proc:
    pid = 1234

    def wait(self):
        return 0
