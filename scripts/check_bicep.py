"""Compile-check Bicep files without the Bicep CLI, through the language server that ships
with the VS Code Bicep extension (needs the .NET runtime it targets). Use `bicep build` or
`az bicep build` instead where the CLI is available; CI compiles with the CLI.

Usage:
    python scripts/check_bicep.py                     # infra/**/*.bicep + *.bicepparam
    python scripts/check_bicep.py infra/main.bicep    # specific files
Prints every compiler/linter diagnostic; exits 1 on errors (and on warnings with --strict).
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

_SERVERS = sorted(
    (Path.home() / ".vscode/extensions").glob(
        "ms-azuretools.vscode-bicep-*/bicepLanguageServer/Bicep.LangServer.dll"
    )
)
if not _SERVERS:
    sys.exit("VS Code Bicep extension not found; install it or use the Bicep CLI")
SERVER = _SERVERS[-1]
STRICT = "--strict" in sys.argv
args = [a for a in sys.argv[1:] if a != "--strict"]
infra = Path(__file__).resolve().parents[1] / "infra"
files = [Path(f).resolve() for f in args] or sorted(
    [*infra.rglob("*.bicep"), *infra.glob("*.bicepparam")]
)

proc = subprocess.Popen(
    ["dotnet", str(SERVER), "--stdio"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)
diagnostics: dict[str, list] = {}
received: dict[str, float] = {}


def send(msg: dict) -> None:
    body = json.dumps(msg).encode("utf-8")
    proc.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
    proc.stdin.flush()


def reader() -> None:
    while True:
        length = 0
        while True:
            line = proc.stdout.readline()
            if not line:
                return
            line = line.strip()
            if not line:
                break
            if line.lower().startswith(b"content-length:"):
                length = int(line.split(b":")[1])
        msg = json.loads(proc.stdout.read(length))
        if msg.get("method") == "textDocument/publishDiagnostics":
            p = msg["params"]
            diagnostics[p["uri"].lower()] = p["diagnostics"]
            received[p["uri"].lower()] = time.time()
        elif "id" in msg and "method" in msg:  # server -> client request: acknowledge
            send({"jsonrpc": "2.0", "id": msg["id"], "result": None})


threading.Thread(target=reader, daemon=True).start()
root = files[0].parent.parent.as_uri() if files else Path.cwd().as_uri()
send(
    {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"processId": None, "rootUri": root, "capabilities": {}},
    }
)
time.sleep(15)
send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
for f in files:
    lang = "bicep-params" if f.suffix == ".bicepparam" else "bicep"
    send(
        {
            "jsonrpc": "2.0",
            "method": "textDocument/didOpen",
            "params": {
                "textDocument": {
                    "uri": f.as_uri(),
                    "languageId": lang,
                    "version": 1,
                    "text": f.read_text(encoding="utf-8"),
                }
            },
        }
    )

# Wait until every file has diagnostics and nothing new arrived for a few seconds.
deadline = time.time() + 180
uris = {f.as_uri().lower() for f in files}
while time.time() < deadline:
    time.sleep(2)
    if uris <= set(received) and time.time() - max(received.values()) > 8:
        break

severity = {1: "ERROR", 2: "WARNING", 3: "INFO", 4: "HINT"}
errors = 0
warnings = 0
for f in files:
    found = diagnostics.get(f.as_uri().lower())
    if found is None:
        print(f"{f.name}: NO DIAGNOSTICS RECEIVED")
        errors += 1
        continue
    print(f"{f.name}: {len(found)} diagnostic(s)")
    for d in found:
        sev = severity.get(d.get("severity", 1))
        errors += sev == "ERROR"
        warnings += sev == "WARNING"
        line = d["range"]["start"]["line"] + 1
        print(f"  {sev} line {line} [{d.get('code', '')}]: {d['message']}")
proc.kill()
sys.exit(1 if errors or (STRICT and warnings) else 0)
