#!/bin/bash
set -e

echo "=== FCC Connection Diagnostic ==="

echo "\n1. Tailscale status:"
tailscale status

echo -e "\n2. .env ANTHROPIC_AUTH_TOKEN:"
grep "^ANTHROPIC_AUTH_TOKEN" .env || echo "  (not found)"

echo -e "\n3. .env MODEL_ entries:"
grep "^MODEL_" .env || echo "  (no MODEL_ entries found)"

echo -e "\n4. .env OLLAMA_BASE_URL:"
grep "^OLLAMA_BASE_URL" .env || echo "  (not found)"

echo -e "\n5. Running connection check..."
# First try to fix any malformed lines in .env
sed -i.bak '/^MODEL_.*/ {s/"ollama/qwen3:14b""ollama/qwen3:14b"/}' .env

echo "\n6. Checking Tailscale peers:"
tailscale status --json | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    peers = data.get('Peer', {})
    for peer in peers.values():
        if peer.get('OS') == 'windows' and peer.get('Online'):
            print(f'Windows PC: {peer.get(\"HostName\", \"?")}')
            print(f'Online: {peer.get(\"Online\", False)}')
            for addr in peer.get('TailscaleIPs', []):
                if ':' not in addr:
                    print(f'IP: {addr}')
except Exception as e:
    print(f'Error parsing Tailscale status: {e}')
"

echo -e "\n7. Running connect_ollama.py..."
uv run scripts/connect_ollama.py

echo -e "\n8. Running status check..."
uv run scripts/fcc_status.py
