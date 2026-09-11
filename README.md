# tinyllm

A small Rust gateway that lets Claude Code use OpenAI, OpenRouter and Z.ai
models. Claude Code keeps control of tools, MCP connections, permissions,
background shells and subagents; tinyllm translates the API traffic.

## What it supports

| Provider | Authentication | Example model ID |
| --- | --- | --- |
| OpenAI | API key or ChatGPT subscription | `openai/gpt-5.6-sol` |
| OpenRouter | API key | `openrouter/deepseek/deepseek-4-pro` |
| Z.ai | API key | `zai/glm-5.3` |

- Anthropic Messages, OpenAI Chat Completions and Responses, with JSON or streaming SSE.
- Text, images, multiple tool calls and structured tool results.
- MCP tools, including deferred ToolSearch, concurrent subagents and background shells through Claude Code.
- Claude Code WebSearch through OpenAI native search, with domain filters and source links.
- OpenAI reasoning continuity through tool calls, restarts, model switches and compaction, with no gateway-side state.
- Subscription login and automatic token refresh; per-model reasoning effort and OpenAI service tier defaults.

Model access and native OpenRouter/Z.ai endpoint support depend on the upstream
account. Unsupported semantic controls return explicit errors. OpenAI conversion
does not support PDFs/audio, exact thinking budgets or Anthropic server compaction.
Token counting is answered locally with `o200k_base` plus estimates for images
and framing; it sizes context, it is not a billing count. Subscription mode does
not enforce `max_tokens` and rejects temperature/top-p. Monitor availability
remains unverified.

## How to use it

### Install

Download a [release](https://github.com/mipsel64/tinyllm/releases) for macOS or
Linux (AMD64/ARM64), verify it with `checksums.txt`, and put `tinyllm` on your PATH.
Or install from this checkout:

```sh
make install
```

This builds in release mode and installs to `~/.local/bin/tinyllm`. Add
`~/.local/bin` to your `PATH` if needed.

`tinyllm --version` prints `<cargo-version>+<short-commit> <UTC-timestamp>`,
prefixed by the binary name, for example `tinyllm 0.1.0+abc1234 2026-09-10T12:34:56Z`.

### Configure and start

From the checkout or extracted release directory:

```sh
mkdir -p ~/.config/tinyllm
cp -n tinyllm.example.toml ~/.config/tinyllm/config.toml
chmod 600 ~/.config/tinyllm/config.toml
tinyllm openai login
tinyllm
```

The example uses subscription auth and listens on `127.0.0.1:8080`. Login prints a
browser link; use `tinyllm openai login --device-auth` on a headless machine.
Stop the gateway before logging in again or running `tinyllm openai logout`.
For a renamed OpenAI provider, add `--provider NAME` to login/logout.

For an OpenAI API key, replace the auth section in your config:

```toml
[providers.openai.auth]
type = "ApiKey"
options = "${OPENAI_API_KEY}"
```

Export `OPENAI_API_KEY` and start `tinyllm`; skip login. OpenRouter and Z.ai use
the `api_key` settings shown in [tinyllm.example.toml](tinyllm.example.toml).
That file documents all server, logging, provider and model options.

Use `tinyllm --config /path/to/config.toml` to select another config; YAML is also
supported. Environment variables expand before parsing, including in comments.
Relative paths resolve beside the config file; use absolute paths instead of `~`.
Restart after editing. Set `server.auth_token` to require a local gateway token;
omitting it disables authentication. Upstream credentials stay on the server.

### Connect Claude Code

In another terminal, set these variables or their equivalents in your project's
Claude settings:

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080/anthropic
export ANTHROPIC_AUTH_TOKEN=tinyllm
export ANTHROPIC_MODEL=openai/gpt-5.6-sol
export ANTHROPIC_DEFAULT_OPUS_MODEL=openai/gpt-5.6-sol
export ANTHROPIC_DEFAULT_SONNET_MODEL=openai/gpt-5.6-sol
export ANTHROPIC_DEFAULT_HAIKU_MODEL=openai/gpt-5.6-sol
export CLAUDE_CODE_SUBAGENT_MODEL=openai/gpt-5.6-sol
export CLAUDE_CODE_DISABLE_ARTIFACT=1
claude
```

Use your `server.auth_token` if configured; otherwise `tinyllm` is a client
placeholder. Choose models your account can access. The prefix is the provider
table name; the rest is the upstream model ID, including any further slashes.
Client reasoning effort and service tier override configured model defaults.
For a configured OpenAI GPT model, append `-fast` (for example,
`openai/gpt-5.6-sol-fast`) to request the priority tier while sending the base model
upstream, overriding client and configured service tiers; subscription requests also
carry Codex's routing hint. The suffix does not stack, and upstream decides what it
delivers: responses may still report `default`. Fast/priority processing can consume
more credits or cost more.

In auto permission mode Claude Code checks every tool call with a short
subrequest, which by default runs on the session's model — so a reasoning model
is asked to judge each Bash command before it runs. Set
`server.auto_review_model` to a small fast model to make those checks quicker
and cheaper.

Claude Code asks for `high` reasoning effort on every turn, so a configured
`reasoning_effort` cannot bring it down. `max_reasoning_effort` is a ceiling
applied after the client's choice: it lowers a larger request and leaves a
smaller one alone. `server.model_aliases` maps client model IDs such as
`claude-sonnet-4-6` onto a configured provider/model.

Set `CLAUDE_CODE_DISABLE_ARTIFACT=1` to prevent Claude Code from calling its hosted
Artifact tool: gateway-token sessions cannot publish to Claude.ai Artifacts.
Ask Claude Code to save HTML locally and open it in your browser instead; disabling
Artifact does not automatically open local files.

ToolSearch is supported; `ENABLE_TOOL_SEARCH=false` loads all MCP definitions
upfront and uses more context. Experimental betas need no blanket disabling.
Avoid `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`: it disables Monitor.
tinyllm does not change global Claude settings.

### Connect other clients

Use `http://127.0.0.1:8080/v1` as an OpenAI client's base URL, a prefixed model ID,
and your local gateway token as its API key (`tinyllm` when auth is disabled).

| Format | Inference | Model discovery |
| --- | --- | --- |
| Anthropic | `POST /anthropic/v1/messages` | `GET /anthropic/v1/models` |
| Chat Completions | `POST /v1/chat/completions` | `GET /v1/models` |
| Responses | `POST /v1/responses` | `GET /v1/models` |

`GET /anthropic/api/hello` returns `{"ok":true}` and is the only route served
without a gateway token: Claude Code warms its connection pool against it before
sending credentials, and a refusal would reveal the same thing a reply does.

### Run as a service

From a source checkout, run these as your normal user:

| Command | Action |
| --- | --- |
| `make build` | Build in release mode. |
| `make install` | Build and install to `~/.local/bin/tinyllm`. |
| `make setup` | Check config, install, then enable and start the native service. |
| `make restart` | Restart the service gracefully, without rebuilding. |
| `make restart REBUILD=1` | Build, install, then restart. |
| `make status` | Print the current launchd/systemd service status. |
| `make clean` | Stop and remove the service and installed binary; keep config, credentials and logs. |

`setup` stops before building if `~/.config/tinyllm/config.toml` is missing and
prints instructions to copy the example and fill the required provider/auth fields.
Edit the config and complete subscription login (if used) before setup.
macOS uses the system LaunchDaemon and prompts for `sudo`; Linux uses a systemd
user service. Enable Linux lingering as described below for startup without login.
Existing private plist customizations are retained; `clean` does not disable lingering.

Use the [macOS launchd, Linux systemd and Docker Compose examples](examples/README.md)
for boot startup, logs, persistent state and graceful shutdown. The image
`ghcr.io/mipsel64/tinyllm` uses `nightly` and `main-<short-sha>` tags for main-branch
builds, and the Git tag (such as `v0.1.0`) for releases.

### Credentials and reasoning continuity

`~/.local/state/tinyllm` (or `$XDG_STATE_HOME/tinyllm`) holds OpenAI credentials
at `auth/openai.json`. Nothing else is stored there.

The gateway keeps no conversation state. OpenAI encrypted reasoning travels back
to the client inside `redacted_thinking` blocks (Chat Completions uses
`reasoning_details`), so restarts, model switches, subagents and multiple gateway
instances all resume without shared storage. A carrier the gateway cannot read —
a foreign signature, an older format, or one the client dropped — replays as
plain history rather than failing the request. Switching to another provider
drops the carriers, since encrypted reasoning is OpenAI-specific.

MIT licensed. Adapted from [m0n0x41d/anthropic-proxy-rs](https://github.com/m0n0x41d/anthropic-proxy-rs/tree/59eb97bc3150106c18589fa5785102ae7be81caa);
upstream and contributor attribution is preserved in [LICENSE](LICENSE).
