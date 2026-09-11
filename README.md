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
- OpenAI reasoning continuation through tool calls, restarts and retained history after compaction.
- Subscription login and automatic token refresh; per-model reasoning effort and OpenAI service tier defaults.

Model access and native OpenRouter/Z.ai endpoint support depend on the upstream
account. Unsupported semantic controls return explicit errors. OpenAI conversion
does not support PDFs/audio, exact thinking budgets or Anthropic server compaction.
Token counting is not implemented. Subscription mode does not enforce `max_tokens`
and rejects temperature/top-p. Monitor availability remains unverified.

## How to use it

### Install

Download a [release](https://github.com/mipsel64/tinyllm/releases) for macOS or
Linux (AMD64/ARM64), verify it with `checksums.txt`, and put `tinyllm` on your PATH.
Or install from this checkout:

```sh
cargo install --locked --path .
```

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

Set `CLAUDE_CODE_DISABLE_ARTIFACT=1` to prevent Claude Code from calling its hosted
Artifact tool: gateway-token sessions cannot publish to Claude.ai Artifacts.
Ask Claude Code to save HTML locally and open it in your browser instead; disabling
Artifact does not automatically open local files.

ToolSearch is supported; `ENABLE_TOOL_SEARCH=false` loads all MCP definitions
upfront and uses more context. Experimental betas need no blanket disabling.
Avoid `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`: it disables Monitor.
tinyllm does not change global Claude settings.

OpenAI WebSearch supports `web_search_20250305`, bare-domain allow/block lists and
approximate location. Answers include Markdown source links; native search state
is preserved for continuation. With an API key, `max_uses` maps to the upstream
`max_tool_calls` cap. Subscription access has no hard-cap parameter: the limit is
best-effort through model instructions, with a warning and an
`x-tinyllm-web-search` response header. Newer dynamic-filtering search versions,
domain paths/wildcards and search combined with stop sequences or structured output
are unsupported. Cited output cannot be combined with either output constraint.
Citation conversion also applies when no search tool was declared. Failed or
incomplete search attempts stay in native state without discarding an otherwise
valid answer; whole-response failures still return errors.

### Connect other clients

Use `http://127.0.0.1:8080/v1` as an OpenAI client's base URL, a prefixed model ID,
and your local gateway token as its API key (`tinyllm` when auth is disabled).

| Format | Inference | Model discovery |
| --- | --- | --- |
| Anthropic | `POST /anthropic/v1/messages` | `GET /anthropic/v1/models` |
| Chat Completions | `POST /v1/chat/completions` | `GET /v1/models` |
| Responses | `POST /v1/responses` | `GET /v1/models` |

### Run as a service

Use the [macOS launchd, Linux systemd and Docker Compose examples](examples/README.md)
for boot startup, logs, persistent state and graceful shutdown. The image
`ghcr.io/mipsel64/tinyllm` uses `nightly` and `main-<short-sha>` tags for main-branch
builds, and the Git tag (such as `v0.1.0`) for releases.

### Manage state

State defaults to `~/.local/state/tinyllm` (or `$XDG_STATE_HOME/tinyllm`).
Default OpenAI credentials live at `auth/openai.json` beneath it. Keep the state
directory to resume conversations; continuation records include plaintext
assistant text and tool arguments. Only one gateway can use a state directory.

The default continuation budget is 256 MiB, with no automatic eviction. Inspect
usage with `tinyllm state status`. Stop the gateway before previewing cleanup:

```sh
tinyllm state prune --older-than-days 30
```

Add `--apply` to delete the selected records. Credentials are preserved, but
deleted turns can no longer resume. Client compaction retains only the surviving
history; missing continuation records produce an error.

MIT licensed. Adapted from [m0n0x41d/anthropic-proxy-rs](https://github.com/m0n0x41d/anthropic-proxy-rs/tree/59eb97bc3150106c18589fa5785102ae7be81caa);
upstream and contributor attribution is preserved in [LICENSE](LICENSE).
