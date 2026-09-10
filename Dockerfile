FROM rust:1.93.1-bookworm AS build
WORKDIR /build
COPY Cargo.toml Cargo.lock rust-toolchain.toml build.rs ./
COPY src ./src
ARG GIT_SHA=unknown
ARG BUILD_DATE
RUN cargo build --locked --release --bin tinyllm \
    && install -d -m 0700 /var/lib/tinyllm

FROM gcr.io/distroless/cc-debian12:nonroot
COPY --from=build /build/target/release/tinyllm /usr/local/bin/tinyllm
COPY --from=build --chown=65532:65532 /var/lib/tinyllm /var/lib/tinyllm
COPY LICENSE /usr/share/licenses/tinyllm/LICENSE
ENV HOME=/home/nonroot XDG_STATE_HOME=/var/lib
USER 65532:65532
RUN ["/usr/local/bin/tinyllm", "--version"]
EXPOSE 8080
STOPSIGNAL SIGINT
ENTRYPOINT ["/usr/local/bin/tinyllm"]
CMD ["--config", "/etc/tinyllm/config.toml"]
