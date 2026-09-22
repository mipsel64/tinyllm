#!/bin/bash
# Isolated smoke test of the native Anthropic provider:
#   pi -> tinyllm /anthropic -> loopback Anthropic fixture
# It covers a streamed tool call and the tool_result continuation that follows.
#
# No real provider, no credentials, no installed service: the fixture, the
# gateway and pi all run on ephemeral loopback ports in a temp directory that is
# removed on exit.
#
#   scripts/smoke-anthropic-pi.sh          # rebuilds target/debug/tinyllm first
#   BIN=target/release/tinyllm scripts/...  # use an existing binary as is
set -uo pipefail

MODEL="claude-sonnet-4-6"
BIN="${BIN:-}"
WORK="$(mktemp -d)"
FIXTURE_PID=""; GATEWAY_PID=""
pass=0; fail=0
ok()  { printf '  \033[32mPASS\033[0m  %s\n' "$1"; pass=$((pass+1)); }
bad() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; fail=$((fail+1)); [ -n "${2:-}" ] && printf '        %s\n' "${2:0:400}"; }
cleanup() {
  [ -n "$FIXTURE_PID" ] && kill "$FIXTURE_PID" 2>/dev/null
  [ -n "$GATEWAY_PID" ] && kill "$GATEWAY_PID" 2>/dev/null
  wait 2>/dev/null
  rm -rf "$WORK"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
trap 'exit 141' PIPE

# python3 is already a dependency here and doubles as the process timeout,
# which macOS has no `timeout` binary for.
run_timeout() {
  python3 -c 'import subprocess, sys
try:
    sys.exit(subprocess.run(sys.argv[2:], timeout=float(sys.argv[1])).returncode)
except subprocess.TimeoutExpired:
    sys.exit(124)' "$@"
}

command -v pi >/dev/null || { echo "pi is not installed"; exit 1; }
if [ -z "$BIN" ]; then
  cargo build --locked || exit 1
  BIN=target/debug/tinyllm
fi
[ -x "$BIN" ] || { echo "$BIN is not executable"; exit 1; }
BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"

# ---------------------------------------------------------------- fixture
printf '%s\n' "the smoke fixture file says SMOKE_FILE_MARKER" > "$WORK/fixture.txt"
cat > "$WORK/fixture.py" <<'PY'
"""Minimal Anthropic Messages upstream: one streamed tool call, then a reply."""
import json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WORK, TARGET = sys.argv[1], sys.argv[2]
count, lock = 0, threading.Lock()


def has_tool_result(body):
    for message in body.get("messages", []):
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(block, dict) and block.get("type") == "tool_result" for block in content
        ):
            return True
    return False


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        global count
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        with lock:
            count += 1
            index = count
        with open(os.path.join(WORK, "upstream-%d.json" % index), "w") as handle:
            json.dump(
                {
                    "path": self.path,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": body,
                },
                handle,
            )
        # Clients may add query parameters, which tinyllm forwards verbatim.
        if self.path.split("?")[0] != "/v1/messages":
            self.send_error(404)
            return
        model = body.get("model", "")
        start = {
            "type": "message_start",
            "message": {
                "id": "msg_%d" % index, "type": "message", "role": "assistant",
                "model": model, "content": [], "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 12, "output_tokens": 1},
            },
        }
        if not body.get("stream"):
            payload = json.dumps({
                **start["message"],
                "content": [{"type": "text", "text": "SMOKE_OK"}],
                "stop_reason": "end_turn",
            }).encode()
            content_type = "application/json"
        else:
            if has_tool_result(body):
                blocks = [
                    {"type": "content_block_start", "index": 0,
                     "content_block": {"type": "text", "text": ""}},
                    {"type": "content_block_delta", "index": 0,
                     "delta": {"type": "text_delta", "text": "SMOKE_OK"}},
                    {"type": "content_block_stop", "index": 0},
                    {"type": "message_delta",
                     "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                     "usage": {"output_tokens": 5}},
                ]
            else:
                blocks = [
                    {"type": "content_block_start", "index": 0,
                     "content_block": {"type": "text", "text": ""}},
                    {"type": "content_block_delta", "index": 0,
                     "delta": {"type": "text_delta", "text": "Reading the file."}},
                    {"type": "content_block_stop", "index": 0},
                    {"type": "content_block_start", "index": 1,
                     "content_block": {"type": "tool_use", "id": "toolu_smoke",
                                       "name": "read", "input": {}}},
                    {"type": "content_block_delta", "index": 1,
                     "delta": {"type": "input_json_delta",
                               "partial_json": json.dumps({"path": TARGET})}},
                    {"type": "content_block_stop", "index": 1},
                    {"type": "message_delta",
                     "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                     "usage": {"output_tokens": 9}},
                ]
            events = [start] + blocks + [{"type": "message_stop"}]
            payload = "".join(
                "event: %s\ndata: %s\n\n" % (event["type"], json.dumps(event)) for event in events
            ).encode()
            content_type = "text/event-stream"
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
with open(os.path.join(WORK, "fixture.port"), "w") as handle:
    handle.write(str(server.server_address[1]))
server.serve_forever()
PY
python3 "$WORK/fixture.py" "$WORK" "$WORK/fixture.txt" > "$WORK/fixture.log" 2>&1 & FIXTURE_PID=$!
for _ in $(seq 50); do [ -s "$WORK/fixture.port" ] && break; sleep 0.1; done
UPSTREAM="$(cat "$WORK/fixture.port" 2>/dev/null)"
[ -n "$UPSTREAM" ] || { echo "fixture did not start"; cat "$WORK/fixture.log"; exit 1; }

# ---------------------------------------------------------------- gateway
cat > "$WORK/config.toml" <<EOF
[server]
bind = "127.0.0.1:0"
auth_token = "smoke-token"
state_dir = "$WORK/state"

[providers.anthropic]
type = "anthropic"
base_url = "http://127.0.0.1:$UPSTREAM/v1"

[providers.anthropic.auth]
type = "ApiKey"
options = "fixture-key"

[providers.anthropic.models."$MODEL"]
EOF
"$BIN" -c "$WORK/config.toml" > "$WORK/gateway.log" 2>&1 & GATEWAY_PID=$!
GATEWAY=""; READY=""
for _ in $(seq 100); do
  GATEWAY="$(sed -n 's|.*address=http://127.0.0.1:\([0-9]*\)/anthropic.*|\1|p' "$WORK/gateway.log" | head -1)"
  if [ -n "$GATEWAY" ] \
    && curl -fsS -m 2 -o /dev/null "http://127.0.0.1:$GATEWAY/anthropic/api/hello"; then
    READY=1; break
  fi
  kill -0 "$GATEWAY_PID" 2>/dev/null || break
  sleep 0.1
done
[ -n "$READY" ] || { echo "gateway never answered the hello probe"; cat "$WORK/gateway.log"; exit 1; }
BASE="http://127.0.0.1:$GATEWAY/anthropic"
printf '\n\033[1mNative streaming through the gateway\033[0m\n'

curl -sS -N -m 10 -o "$WORK/stream.sse" \
  -H "x-api-key: smoke-token" -H 'content-type: application/json' \
  -H 'anthropic-beta: smoke-beta-2026-01-01' \
  --data-binary "{\"model\":\"anthropic/$MODEL\",\"max_tokens\":64,\"stream\":true,\"thinking\":{\"type\":\"adaptive\"},\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}" \
  "$BASE/v1/messages"
stream_status=$?
{ [ "$stream_status" -eq 0 ] \
    && grep -q 'event: message_start' "$WORK/stream.sse" \
    && grep -q 'event: message_stop' "$WORK/stream.sse"; } \
  && ok "gateway returns a complete SSE stream" \
  || bad "gateway returns a complete SSE stream (curl exit $stream_status)" \
         "$(cat "$WORK/stream.sse")"
grep -q "\"model\":\"anthropic/$MODEL\"" "$WORK/stream.sse" && ok "client sees the prefixed model ID" \
  || bad "client sees the prefixed model ID" "$(head -2 "$WORK/stream.sse")"

# ---------------------------------------------------------------- pi leg
mkdir -p "$WORK/pi"
cat > "$WORK/pi/models.json" <<EOF
{
  "providers": {
    "tinyllm": {
      "baseUrl": "$BASE",
      "api": "anthropic-messages",
      "apiKey": "smoke-token",
      "models": [
        { "id": "anthropic/$MODEL", "reasoning": true, "input": ["text"],
          "compat": { "forceAdaptiveThinking": true, "supportsStrictTools": true } }
      ]
    }
  }
}
EOF
printf '\n\033[1mpi streaming tool call and tool_result continuation\033[0m\n'
run_timeout "${PI_TIMEOUT:-120}" \
  env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN -u ANTHROPIC_BASE_URL -u ANTHROPIC_MODEL \
      -u CLAUDE_CODE_SUBAGENT_MODEL \
      PI_CODING_AGENT_DIR="$WORK/pi" PI_OFFLINE=1 PI_SKIP_VERSION_CHECK=1 \
    pi -p --no-extensions --no-skills --no-prompt-templates --no-context-files \
       --no-session --no-approve --tools read \
       --provider tinyllm --model "tinyllm/anthropic/$MODEL" \
       "Read $WORK/fixture.txt and report what it says." \
  > "$WORK/pi.out" 2> "$WORK/pi.err"
pi_status=$?
{ [ "$pi_status" -eq 0 ] && grep -q SMOKE_OK "$WORK/pi.out"; } \
  && ok "pi completed the turn after the tool result" \
  || bad "pi completed the turn after the tool result (exit $pi_status)" \
         "$(tail -5 "$WORK/pi.out" "$WORK/pi.err")"

# ---------------------------------------------------------------- upstream
printf '\n\033[1mWhat the upstream actually received\033[0m\n'
python3 - "$WORK" "$MODEL" <<'PY'
import glob, json, os, sys
work, model = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(work, "upstream-*.json")),
               key=lambda p: int(p.rsplit("-", 1)[1].split(".")[0]))
checks = []
requests = [json.load(open(path)) for path in files]
checks.append((len(requests) >= 3, "upstream saw the curl turn plus both pi turns",
               "saw %d requests" % len(requests)))
paths = {request["path"].split("?")[0] for request in requests}
checks.append((paths == {"/v1/messages"}, "every upstream request hit /v1/messages", str(paths)))
headers = [request["headers"] for request in requests]
checks.append((all(h.get("x-api-key") == "fixture-key" for h in headers),
               "upstream received the provider key as x-api-key",
               str([h.get("x-api-key") for h in headers])))
checks.append((not any("authorization" in h for h in headers),
               "no authorization header reached the upstream", ""))
checks.append((all("anthropic-version" in h for h in headers),
               "anthropic-version was sent on every request", ""))
checks.append((headers[0].get("anthropic-beta") == "smoke-beta-2026-01-01",
               "anthropic-beta passed through unchanged", str(headers[0].get("anthropic-beta"))))
raw = "".join(open(path).read() for path in files)
checks.append(("smoke-token" not in raw, "the local gateway token never went upstream", ""))
checks.append((all(request["body"]["model"] == model for request in requests),
               "upstream saw the bare native model ID",
               str([request["body"]["model"] for request in requests])))
checks.append((requests[0]["body"].get("thinking") == {"type": "adaptive"},
               "native thinking controls passed through unchanged",
               str(requests[0]["body"].get("thinking"))))
pi_requests = requests[1:]
checks.append((bool(pi_requests) and all(
    request["body"].get("thinking", {}).get("type") == "adaptive"
    and request["body"].get("output_config")
    and request["body"].get("tools")
    and all(tool.get("strict") and tool.get("cache_control")
            for tool in request["body"]["tools"])
    for request in pi_requests),
    "pi's own thinking, output_config and strict cached tools reached the upstream",
    json.dumps([[request["body"].get("thinking"), request["body"].get("output_config"),
                 request["body"].get("tools", [])[:1]] for request in pi_requests])[:300]))
results = []
for request in requests[1:]:
    for message in request["body"].get("messages", []):
        content = message.get("content")
        if isinstance(content, list):
            results += [block for block in content
                        if isinstance(block, dict) and block.get("type") == "tool_result"]
checks.append((any(block.get("tool_use_id") == "toolu_smoke" for block in results),
               "the continuation carried the tool_result for the streamed tool_use",
               json.dumps(results)[:200]))
checks.append(("SMOKE_FILE_MARKER" in json.dumps(results),
               "the tool result contained the file pi actually read", ""))
failed = 0
for good, label, detail in checks:
    print("  \033[32mPASS\033[0m  %s" % label if good else "  \033[31mFAIL\033[0m  %s\n        %s" % (label, detail))
    failed += 0 if good else 1
sys.exit(1 if failed else 0)
PY
upstream=$?
printf '\n'
if [ "$fail" = 0 ] && [ "$upstream" = 0 ]; then
  printf '\033[32msmoke passed\033[0m (%d shell checks + upstream checks)\n' "$pass"
  exit 0
fi
printf '\033[31msmoke failed\033[0m (%d shell failures)\n' "$fail"
exit 1
