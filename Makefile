SHELL := /bin/sh
.DEFAULT_GOAL := build

OS := $(shell uname -s)
CARGO_TARGET_DIR ?= target
BIN_DIR := $(HOME)/.local/bin
CONFIG := $(HOME)/.config/tinyllm/config.toml
PLIST := $(HOME)/.config/tinyllm/tinyllm.plist
LAUNCHD_PLIST := /Library/LaunchDaemons/io.github.mipsel64.tinyllm.plist
LAUNCHD_SERVICE := system/io.github.mipsel64.tinyllm
BOOTOUT_TIMEOUT ?= 680
SYSTEMD_DIR := $(or $(XDG_CONFIG_HOME),$(HOME)/.config)/systemd/user

.PHONY: build install setup restart status clean ensure-config ensure-os

build:
	cargo build --locked --release --bin tinyllm --target-dir "$(CARGO_TARGET_DIR)"

install: build
	@set -eu; \
	mkdir -p "$(BIN_DIR)"; \
	tmp=$$(mktemp "$(BIN_DIR)/.tinyllm.XXXXXX"); \
	trap 'rm -f "$$tmp"' 0; \
	install -m 755 "$(CARGO_TARGET_DIR)/release/tinyllm" "$$tmp"; \
	mv -f "$$tmp" "$(BIN_DIR)/tinyllm"

ensure-config:
	@if ! test -f "$(CONFIG)"; then \
		printf '%s\n' \
			"Missing config: $(CONFIG)" \
			'Copy the example, then fill the required provider type and authentication fields:' \
			'  mkdir -p "$$HOME/.config/tinyllm"' \
			'  cp tinyllm.example.toml "$$HOME/.config/tinyllm/config.toml"' \
			'  chmod 600 "$$HOME/.config/tinyllm/config.toml"' \
			'For API-key auth, fill the API key; for Subscription, run tinyllm openai login after make install.' \
			'Edit the config, then rerun make setup.' >&2; \
		exit 1; \
	fi

ensure-os:
	@case "$(OS)" in Darwin|Linux) ;; *) printf 'Unsupported OS: %s\n' "$(OS)" >&2; exit 1 ;; esac

setup: ensure-config ensure-os
	$(MAKE) install
ifeq ($(OS),Darwin)
	@set -eu; umask 077; \
	mkdir -p "$(HOME)/Library/Logs/tinyllm"; \
	chmod 700 "$(HOME)/Library/Logs/tinyllm"; \
	test -e "$(PLIST)" || cp examples/tinyllm.plist "$(PLIST)"; \
	plutil -replace UserName -string "$$(id -un)" "$(PLIST)"; \
	plutil -remove ProgramArguments.0 "$(PLIST)"; \
	plutil -insert ProgramArguments.0 -string "$(BIN_DIR)/tinyllm" "$(PLIST)"; \
	plutil -remove ProgramArguments.2 "$(PLIST)"; \
	plutil -insert ProgramArguments.2 -string "$(CONFIG)" "$(PLIST)"; \
	plutil -replace EnvironmentVariables.HOME -string "$(HOME)" "$(PLIST)"; \
	plutil -replace EnvironmentVariables.XDG_STATE_HOME -string "$(HOME)/.local/state" "$(PLIST)"; \
	plutil -replace StandardOutPath -string "$(HOME)/Library/Logs/tinyllm/stdout.log" "$(PLIST)"; \
	plutil -replace StandardErrorPath -string "$(HOME)/Library/Logs/tinyllm/stderr.log" "$(PLIST)"; \
	plutil -lint "$(PLIST)"; \
	sudo install -o root -g wheel -m 644 "$(PLIST)" "$(LAUNCHD_PLIST)"
	sudo launchctl enable "$(LAUNCHD_SERVICE)"
else ifeq ($(OS),Linux)
	mkdir -p "$(SYSTEMD_DIR)"
	install -m 644 examples/tinyllm.service "$(SYSTEMD_DIR)/tinyllm.service"
	systemctl --user daemon-reload
	systemctl --user enable tinyllm.service
endif
	$(MAKE) restart REBUILD=0

restart: ensure-os
ifeq ($(REBUILD),1)
	$(MAKE) install
endif
ifeq ($(OS),Darwin)
	@set -eu; \
	if launchctl print "$(LAUNCHD_SERVICE)" >/dev/null 2>&1; then \
		sudo launchctl bootout "$(LAUNCHD_SERVICE)" || true; \
		waited=0; \
		while launchctl print "$(LAUNCHD_SERVICE)" >/dev/null 2>&1; do \
			if test "$$waited" -eq 0; then \
				printf 'Waiting for %s to finish draining in-flight requests...\n' "$(LAUNCHD_SERVICE)" >&2; \
			fi; \
			if test "$$waited" -ge "$(BOOTOUT_TIMEOUT)"; then \
				printf 'Still loaded after %ss; rerun make restart once it exits.\n' "$(BOOTOUT_TIMEOUT)" >&2; \
				exit 1; \
			fi; \
			sleep 1; \
			waited=$$((waited + 1)); \
		done; \
	fi; \
	sudo launchctl bootstrap system "$(LAUNCHD_PLIST)"
else ifeq ($(OS),Linux)
	systemctl --user restart tinyllm.service
endif

status: ensure-os
ifeq ($(OS),Darwin)
	launchctl print "$(LAUNCHD_SERVICE)"
else ifeq ($(OS),Linux)
	systemctl --user status --no-pager tinyllm.service
endif

clean: ensure-os
ifeq ($(OS),Darwin)
	@if launchctl print "$(LAUNCHD_SERVICE)" >/dev/null 2>&1; then \
		sudo launchctl bootout "$(LAUNCHD_SERVICE)"; \
	fi
	@if test -e "$(LAUNCHD_PLIST)"; then sudo rm -f "$(LAUNCHD_PLIST)"; fi
else ifeq ($(OS),Linux)
	@set -eu; \
	if test -e "$(SYSTEMD_DIR)/tinyllm.service" || systemctl --user is-active --quiet tinyllm.service; then \
		systemctl --user stop tinyllm.service; \
		if test -e "$(SYSTEMD_DIR)/tinyllm.service"; then \
			systemctl --user disable tinyllm.service; \
			rm -f "$(SYSTEMD_DIR)/tinyllm.service"; \
		fi; \
		systemctl --user daemon-reload; \
	fi
endif
	rm -f "$(BIN_DIR)/tinyllm"
