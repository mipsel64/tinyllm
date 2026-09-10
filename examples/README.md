# Run tinyllm as a service

Choose one deployment for port 8080. Native services run the installed binary
directly, reuse your subscription login and bind to localhost with the default
config. Docker has its own persistent state volume. These examples give active
requests up to 11 minutes to finish on stop; increase that limit if you increase
the gateway timeouts. Boot startup requires the home directory to be available
and the host to remain awake.

## Prepare the native binary

From an extracted release directory:

```sh
mkdir -p "$HOME/.local/bin" "$HOME/.config/tinyllm"
install -m 755 ./tinyllm "$HOME/.local/bin/tinyllm"
cp -n tinyllm.example.toml "$HOME/.config/tinyllm/config.toml"
chmod 700 "$HOME/.config/tinyllm"
chmod 600 "$HOME/.config/tinyllm/config.toml"
"$HOME/.local/bin/tinyllm" --version
```

From a source checkout, first run `cargo build --locked --release` and use
`target/release/tinyllm` in the install command. Edit the config before starting
the service. Keep `bind = "127.0.0.1:8080"` for local Claude Code.

For subscription auth, log in as your normal user before starting either native
service. Both examples set the same state-directory default:

```sh
XDG_STATE_HOME="$HOME/.local/state" "$HOME/.local/bin/tinyllm" \
  --config "$HOME/.config/tinyllm/config.toml" openai login
```

Use `--device-auth` for a headless machine. API-key auth skips login. Explicit
`server.state_dir` or `credentials_dir` settings override the defaults; keep those
paths consistent between login and the service. Stop the service before logging
in again, logging out or applying `tinyllm state prune`.

## macOS: launchd

[tinyllm.plist](tinyllm.plist) is a LaunchDaemon that starts at boot and runs as
your normal user. It remains available after logout. launchd does not expand
`~`, `$HOME` or other shell syntax in plist paths.

```sh
mkdir -p "$HOME/Library/Logs/tinyllm"
chmod 700 "$HOME/Library/Logs/tinyllm"
cp -n examples/tinyllm.plist "$HOME/.config/tinyllm/tinyllm.plist"
```

Edit that copy: replace every `__HOME__` with your absolute home path and
`__USER__` with the result of `id -un`. Keep credentials in the private config;
the installed plist is readable by other users. launchd does not inherit your
terminal's environment or load a shell profile. Config references to environment
variables need corresponding `EnvironmentVariables` entries, or literal values
in the private config.

Install and start:

```sh
plutil -lint "$HOME/.config/tinyllm/tinyllm.plist"
sudo install -o root -g wheel -m 644 "$HOME/.config/tinyllm/tinyllm.plist" \
  /Library/LaunchDaemons/io.github.mipsel64.tinyllm.plist
sudo launchctl bootstrap system /Library/LaunchDaemons/io.github.mipsel64.tinyllm.plist
sudo launchctl print system/io.github.mipsel64.tinyllm
tail -f "$HOME/Library/Logs/tinyllm/stderr.log"
```

Restart gracefully after a config or binary update:

```sh
sudo launchctl bootout system/io.github.mipsel64.tinyllm
sudo launchctl bootstrap system /Library/LaunchDaemons/io.github.mipsel64.tinyllm.plist
```

Stop with `bootout`; remove the installed plist to prevent startup on the next
boot. After changing the plist itself, stop, reinstall it and bootstrap again.
Log files append continuously; stop the service before rotating them, then
bootstrap again. `kickstart -k` forcibly kills the process, so use the sequence
above when active requests should finish.

See Apple's [launchd guide](https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html)
and the installed `man launchd.plist` / `man launchctl` for platform details.

## Linux: systemd

[tinyllm.service](tinyllm.service) is a **user** unit; `%h` resolves to your home
directory. Run the following as your normal user:

```sh
mkdir -p "$HOME/.config/systemd/user"
install -m 644 examples/tinyllm.service "$HOME/.config/systemd/user/tinyllm.service"
systemctl --user daemon-reload
systemctl --user enable --now tinyllm.service
sudo loginctl enable-linger "$(id -un)"
systemctl --user status tinyllm.service
journalctl --user -u tinyllm.service -f
```

Lingering starts your user manager at boot and keeps it after logout. Without
lingering, startup depends on a login session. The journal handles logs according
to the host's retention policy.

The unit optionally reads `~/.config/tinyllm/environment` for config variables:

```text
OPENAI_API_KEY=your-key
TINYLLM_TOKEN=your-local-token
```

Create that file only if needed, set mode 600, and use `NAME=value` lines without
`export`. systemd does not source shell profiles.

| Operation | Command |
| --- | --- |
| Apply config or environment changes | `systemctl --user restart tinyllm.service` |
| Stop | `systemctl --user stop tinyllm.service` |
| Stop and disable startup | `systemctl --user disable --now tinyllm.service` |

After editing the unit, run `daemon-reload` before restarting. Disabling this unit
does not require disabling lingering, which may serve other user services.
See the systemd [service reference](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html)
and [loginctl reference](https://www.freedesktop.org/software/systemd/man/latest/loginctl.html).

## Docker: Compose

[compose.yaml](compose.yaml) uses the GHCR image, `unless-stopped` restarts, a
named state volume and bounded Docker logs. Use Docker Compose 2.24 or later.
Docker itself must start on boot; desktop Docker engines may require login.

Copy the files into a private deployment directory:

```sh
mkdir -p "$HOME/.config/tinyllm/docker"
chmod 700 "$HOME/.config/tinyllm/docker"
cp -n examples/compose.yaml "$HOME/.config/tinyllm/docker/compose.yaml"
cp -n tinyllm.example.toml "$HOME/.config/tinyllm/docker/config.toml"
chmod 644 "$HOME/.config/tinyllm/docker/config.toml"
cd "$HOME/.config/tinyllm/docker"
```

The directory stays private; the config file must be readable by the container's
UID 65532. In `config.toml`, change the existing `server.bind` to
`"0.0.0.0:8080"`. Leave `state_dir` unset or use `"/var/lib/tinyllm"`; leave the
default subscription credentials directory unset too. The host port is still
bound to `127.0.0.1`, so Claude Code keeps its usual localhost base URL.

If the config expands environment variables, add a mode-600 `environment` file
beside `compose.yaml`, using the `NAME=value` format above. Compose passes it to
both login and serving containers. `TINYLLM_IMAGE` in your shell or Compose `.env`
can select a published release tag instead of the default `latest` image.

Log in with the shared volume, then start in the background:

```sh
docker compose config --quiet
docker compose pull
docker compose run --rm tinyllm --config /etc/tinyllm/config.toml openai login --device-auth
docker compose up -d
docker compose ps
docker compose logs --tail 100 -f tinyllm
```

Skip login for API keys. Stop the serving container before repeating login,
logging out or running state cleanup. One-off commands must include `--config`
because they replace the image's default command:

```sh
docker compose stop
docker compose run --rm tinyllm --config /etc/tinyllm/config.toml state status
docker compose run --rm tinyllm --config /etc/tinyllm/config.toml state prune --older-than-days 30
docker compose up -d
```

Pruning previews only; add `--apply` to delete selected continuation records.
For config changes use `docker compose restart`; for environment changes or an
image update, use `docker compose pull` and `docker compose up -d --force-recreate`.
`docker compose down` keeps the named volume; `down --volumes` deletes credentials
and continuation state. Keep the project name/volume when upgrading.

The image is distroless: run tinyllm commands directly; it has no shell or curl.
See Docker's [restart policy](https://docs.docker.com/engine/containers/start-containers-automatically/)
and [Compose service reference](https://docs.docker.com/reference/compose-file/services/).
