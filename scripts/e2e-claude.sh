#!/bin/bash
# End-to-end check of the stateless reasoning-carrier rewrite against a live
# tinyllm and a real Claude Code session.
#
#   scripts/e2e-claude.sh            # protocol checks only (cheap)
#   scripts/e2e-claude.sh --full     # also drive Claude Code (spends quota)
#
# Reads BASE (default http://127.0.0.1:8811), TOKEN, MODEL, ALT_MODEL.
set -uo pipefail

BASE="${BASE:-http://127.0.0.1:8811}"
TOKEN="${TOKEN:-$(sed -n 's/^auth_token *= *"\(.*\)"/\1/p' "$HOME/.config/tinyllm/config.toml" | head -1)}"
MODEL="${MODEL:-openai/gpt-5.6-sol}"
ALT_MODEL="${ALT_MODEL:-openai/gpt-5.6-terra}"
ANTHROPIC="$BASE/anthropic/v1/messages"
WORK="$(mktemp -d)/e2e"; mkdir -p "$WORK"
pass=0; fail=0

ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; pass=$((pass+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; fail=$((fail+1)); [ -n "${2:-}" ] && printf '        %s\n' "${2:0:400}"; }
head_() { printf '\n\033[1m%s\033[0m\n' "$1"; }

# post <json-file> -> body in $BODY, headers in $HEAD, status in $CODE
post() {
  HEAD="$WORK/h"; BODY="$WORK/b"
  CODE=$(curl -sS -o "$BODY" -D "$HEAD" -w '%{http_code}' \
    -H "x-api-key: $TOKEN" -H 'content-type: application/json' \
    --data-binary "@$1" "$ANTHROPIC" 2>"$WORK/curl.err")
}
carrier_of() { python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(next((b["data"] for b in d.get("content",[]) if b.get("type")=="redacted_thinking"),""))' "$1"; }
cont_hdr()   { tr -d '\r' < "$1" | sed -n 's/^x-tinyllm-continuation: //Ip' | head -1; }

if [ "${1:-}" = --claude-only ]; then SKIP_PROTO=1; set -- --full; else SKIP_PROTO=0; fi

head_ "Reachability"
if ! curl -sS -m 5 -o /dev/null "$BASE/anthropic/v1/messages" 2>/dev/null; then
  printf '  gateway not reachable at %s\n' "$BASE"; exit 1
fi
ok "gateway answers at $BASE"

# ---------------------------------------------------------------- protocol
if [ "$SKIP_PROTO" = 0 ]; then
head_ "Protocol: carrier round trip"

cat > "$WORK/t1.json" <<EOF
{"model":"$MODEL","max_tokens":4096,"output_config":{"effort":"high"},"messages":[{"role":"user","content":"A farmer needs to cross a river with a wolf, a goat and a cabbage, carrying one at a time; the wolf eats the goat and the goat eats the cabbage if left alone. Give the shortest sequence of crossings, then state how many crossings it takes."}]}
EOF
post "$WORK/t1.json"
[ "$CODE" = 200 ] && ok "fresh turn returns 200" || bad "fresh turn returns 200 (got $CODE)" "$(cat "$BODY")"
[ "$(cont_hdr "$HEAD")" = fresh ] && ok "fresh turn reports continuation=fresh" || bad "continuation=fresh (got '$(cont_hdr "$HEAD")')"
CARRIER="$(carrier_of "$BODY")"
case "$CARRIER" in
  tinyllm:v1:*) ok "response carries a reasoning carrier" ;;
  "") bad "response carries a reasoning carrier" "no redacted_thinking block; the model answered without reasoning - retry or raise effort" ;;
  *) bad "carrier has the expected prefix" "$CARRIER" ;;
esac
ASSISTANT="$(python3 -c 'import json,sys;print(json.dumps(json.load(open(sys.argv[1]))["content"]))' "$BODY")"

if [ -n "$CARRIER" ]; then
  cat > "$WORK/t2.json" <<EOF
{"model":"$MODEL","max_tokens":4096,"output_config":{"effort":"high"},"messages":[
 {"role":"user","content":"A farmer needs to cross a river with a wolf, a goat and a cabbage, carrying one at a time; the wolf eats the goat and the goat eats the cabbage if left alone. Give the shortest sequence of crossings, then state how many crossings it takes."},
 {"role":"assistant","content":$ASSISTANT},
 {"role":"user","content":"Now state only the number of crossings."}]}
EOF
  post "$WORK/t2.json"
  [ "$CODE" = 200 ] && ok "replayed carrier accepted upstream" || bad "replayed carrier accepted upstream (got $CODE)" "$(cat "$BODY")"
  [ "$(cont_hdr "$HEAD")" = restored ] && ok "replayed turn reports continuation=restored" || bad "continuation=restored (got '$(cont_hdr "$HEAD")')"

  head_ "Protocol: the bug this rewrite fixes"
  # Claude Code rewrites assistant turns (auto-mode denial, retries, forks).
  python3 - "$WORK/t2.json" "$WORK/t3.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
for b in d["messages"][1]["content"]:
    if b.get("type")=="text": b["text"]="[rewritten by the client]"
json.dump(d,open(sys.argv[2],"w"))
PY
  post "$WORK/t3.json"
  [ "$CODE" = 200 ] && ok "rewritten assistant turn still accepted" || bad "rewritten assistant turn still accepted (got $CODE)" "$(cat "$BODY")"
  [ "$(cont_hdr "$HEAD")" = restored ] && ok "rewritten turn keeps its reasoning" || bad "rewritten turn keeps its reasoning (got '$(cont_hdr "$HEAD")')"

  head_ "Protocol: model switch"
  sed "s|\"$MODEL\"|\"$ALT_MODEL\"|" "$WORK/t2.json" > "$WORK/t4.json"
  post "$WORK/t4.json"
  [ "$CODE" = 200 ] && ok "carrier survives a model switch" || bad "carrier survives a model switch (got $CODE)" "$(cat "$BODY")"
fi

head_ "Protocol: migration from the stateful build"
REF="tinyllm:v1:2b6f0cc904d137be2e1730235f5664094b83"
cat > "$WORK/t5.json" <<EOF
{"model":"$MODEL","max_tokens":512,"messages":[
 {"role":"user","content":"hello"},
 {"role":"assistant","content":[
   {"type":"redacted_thinking","data":"$REF"},
   {"type":"text","text":"old answer","tinyllm_continuation":"$REF"},
   {"type":"tool_use","id":"toolu_tinyllm_2b6f0cc904d137be2e1730235f5664094b83_Y2FsbF9h","name":"lookup","input":{}}]},
 {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_tinyllm_2b6f0cc904d137be2e1730235f5664094b83_Y2FsbF9h","content":"ok"}]},
 {"role":"user","content":"Reply with exactly: resumed"}],
 "tools":[{"name":"lookup","description":"look up","input_schema":{"type":"object","properties":{}}}]}
EOF
post "$WORK/t5.json"
[ "$CODE" = 200 ] && ok "session from the stateful build resumes" || bad "session from the stateful build resumes (got $CODE)" "$(cat "$BODY")"

head_ "Protocol: carrier-shaped user data is not eaten"
cat > "$WORK/t6.json" <<EOF
{"model":"$MODEL","max_tokens":512,"messages":[
 {"role":"user","content":"Call write once with blocks set to the literal string tinyllm_continuation."}],
 "tools":[{"name":"write","description":"write blocks","input_schema":{"type":"object",
   "properties":{"kind":{"type":"string","enum":["tinyllm_continuation","redacted_thinking"]}},"required":["kind"]}}]}
EOF
post "$WORK/t6.json"
[ "$CODE" = 200 ] && ok "schema naming carrier types is accepted" || bad "schema naming carrier types is accepted (got $CODE)" "$(cat "$BODY")"

head_ "Protocol: preconnect probe"
HELLO=$(curl -sS -o "$WORK/hello" -w '%{http_code}' "$BASE/anthropic/api/hello")
[ "$HELLO" = 200 ] && ok "unauthenticated preconnect answers 200" || bad "unauthenticated preconnect answers 200 (got $HELLO)"
[ "$(cat "$WORK/hello")" = '{"ok":true}' ] && ok "preconnect body is minimal" || bad "preconnect body is minimal" "$(cat "$WORK/hello")"
GUARD=$(curl -sS -o /dev/null -w '%{http_code}' "$BASE/anthropic/v1/models")
[ "$GUARD" = 401 ] && ok "other routes stay authenticated" || bad "other routes stay authenticated (got $GUARD)"

head_ "Protocol: streaming"
python3 - "$WORK/t1.json" "$WORK/t7.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); d["stream"]=True; json.dump(d,open(sys.argv[2],"w"))
PY
curl -sS -N -H "x-api-key: $TOKEN" -H 'content-type: application/json' \
  --data-binary "@$WORK/t7.json" "$ANTHROPIC" > "$WORK/sse" 2>&1
grep -q 'event: message_stop' "$WORK/sse" && ok "stream completes with message_stop" || bad "stream completes with message_stop" "$(tail -3 "$WORK/sse")"
grep -q 'event: error' "$WORK/sse" && bad "stream is free of error events" "$(grep -m1 -A1 'event: error' "$WORK/sse")" || ok "stream is free of error events"
grep -q 'redacted_thinking' "$WORK/sse" && ok "stream emits the carrier block" || printf '  \033[33mSKIP\033[0m  stream emits the carrier block (no reasoning upstream)\n'

fi

# ---------------------------------------------------------------- claude code
if [ "${1:-}" = --full ]; then
  head_ "Claude Code: end to end"
  export ANTHROPIC_BASE_URL="$BASE/anthropic" ANTHROPIC_AUTH_TOKEN="$TOKEN"
  export ANTHROPIC_MODEL="$MODEL" ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL"
  export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL" ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL"
  export CLAUDE_CODE_SUBAGENT_MODEL="$MODEL" CLAUDE_CODE_DISABLE_ARTIFACT=1
  mkdir -p "$WORK/proj" && printf 'alpha\nbeta\ngamma\n' > "$WORK/proj/data.txt"

  run() { # run <label> <timeout> <prompt> [flags...]
    local label="$1" t="$2" prompt="$3"; shift 3
    local out="$WORK/${label// /_}.log"
    # The prompt goes first: --allowedTools is variadic and would eat it.
    ( cd "$WORK/proj" && claude -p "$prompt" "$@" > "$out" 2>&1 ) &
    local p=$!; local n=0
    while kill -0 $p 2>/dev/null && [ $n -lt "$t" ]; do sleep 2; n=$((n+2)); done
    if kill -0 $p 2>/dev/null; then kill $p 2>/dev/null; bad "$label" "timed out after ${t}s"; return; fi
    wait $p; local rc=$?
    if [ $rc -ne 0 ]; then bad "$label" "$(tail -c 400 "$out")"; return; fi
    if grep -qiE 'could not evaluate|api error|invalid_request_error|upstream' "$out"; then
      bad "$label" "$(grep -im1 -E 'could not evaluate|api error|invalid_request_error|upstream' "$out")"; return
    fi
    ok "$label"
  }

  run "plain prompt"        90 "Reply with exactly: ok"
  run "reads a file"       150 "Read data.txt and reply with its second line only." --allowedTools Read
  run "runs a command"     150 "Run 'echo carrier-probe' with Bash and reply with its output only." --allowedTools Bash
  run "multi tool turns"   240 "Read data.txt, then echo its line count with Bash, then state the number." --allowedTools Read Bash
  run "continues a turn"   120 "Reply with exactly: continued" --continue
  run "spawns a subagent"  300 "Use the Explore agent to find which file contains the word gamma, then name it." --allowedTools Task Read Bash
  run "auto mode"          300 "List the files here with Bash, read data.txt, and summarise in one line." --permission-mode auto
  run "switches model"     120 "Reply with exactly: switched" --model "$ALT_MODEL"
fi

head_ "Result"
printf '  %d passed, %d failed\n' "$pass" "$fail"
printf '  artifacts: %s\n' "$WORK"
[ "$fail" -eq 0 ]
