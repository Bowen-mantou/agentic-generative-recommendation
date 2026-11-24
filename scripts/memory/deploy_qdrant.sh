#!/usr/bin/env bash
# Deploy the dedicated Qdrant container used by the AGR Memory module (Phase 6 / Q7).
#
# Design constraints (see docs/phase6/D9_MEMORY_DEPLOYMENT.md):
#   * one project-owned container, never touching other teams' containers;
#   * HTTP/gRPC bound to loopback only (no public exposure);
#   * active data on local disk (named Docker volume or an explicit host path), never NFS;
#   * idempotent: re-running only starts/enables the existing container.
#
# Usage:
#   bash scripts/memory/deploy_qdrant.sh            # deploy or start
#   bash scripts/memory/deploy_qdrant.sh status     # inspect without changing anything
#   bash scripts/memory/deploy_qdrant.sh --dry-run  # print the docker command only
#
# Every knob can be overridden through environment variables; the defaults match the
# values validated for the S4000 host (192.0.2.10, agr) on 2026-11-24.

set -euo pipefail

QDRANT_CONTAINER="${QDRANT_CONTAINER:-agr-qdrant}"
QDRANT_IMAGE="${QDRANT_IMAGE:-qdrant/qdrant:v1.20.2}"
# Pull-through mirror used because registry-1.docker.io is unreachable from the host.
QDRANT_MIRROR_IMAGE="${QDRANT_MIRROR_IMAGE:-docker.m.daocloud.io/qdrant/qdrant:latest}"
QDRANT_VOLUME="${QDRANT_VOLUME:-agr-qdrant-data}"
QDRANT_HTTP_PORT="${QDRANT_HTTP_PORT:-6333}"
QDRANT_GRPC_PORT="${QDRANT_GRPC_PORT:-6334}"
QDRANT_BIND="${QDRANT_BIND:-127.0.0.1}"
QDRANT_RUNTIME="${QDRANT_RUNTIME:-runc}"
QDRANT_HTTP_URL="${QDRANT_HTTP_URL:-http://127.0.0.1:${QDRANT_HTTP_PORT}}"

log() { printf '[deploy_qdrant] %s\n' "$*"; }
die() { printf '[deploy_qdrant] ERROR: %s\n' "$*" >&2; exit 1; }

json_field() { # stdin JSON -> value of a top-level key
  python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get(sys.argv[1], ""))' "$1"
}

require_docker() {
  command -v docker >/dev/null 2>&1 || die "docker is not installed"
  docker info >/dev/null 2>&1 || die "docker daemon is not reachable (or the user is not in the docker group)"
}

# A port is considered free when no listener is bound to it. If the listener is our own
# container, deployment continues (the script is idempotent); otherwise we stop instead of
# interfering with somebody else's service.
check_port() {
  local port="$1"
  local line
  line="$({ ss -H -ltn 2>/dev/null || netstat -ltn 2>/dev/null; } | grep -E ":${port}[[:space:]]" || true)"
  if [ -z "$line" ]; then
    return 0
  fi
  if docker port "$QDRANT_CONTAINER" 2>/dev/null | grep -qE ":${port}$"; then
    log "port ${port} is held by this container (${QDRANT_CONTAINER})"
    return 0
  fi
  printf '%s\n' "$line" >&2
  die "port ${port} is already in use by another process; refusing to interfere (set QDRANT_HTTP_PORT/QDRANT_GRPC_PORT to change it)"
}

ensure_image() {
  if docker image inspect "$QDRANT_IMAGE" >/dev/null 2>&1; then
    return 0
  fi
  log "image ${QDRANT_IMAGE} missing; pulling via mirror ${QDRANT_MIRROR_IMAGE}"
  if ! docker pull "$QDRANT_MIRROR_IMAGE"; then
    log "mirror pull failed; trying the canonical name directly"
    docker pull "$QDRANT_IMAGE"
  else
    docker tag "$QDRANT_MIRROR_IMAGE" "$QDRANT_IMAGE"
  fi
  docker image inspect "$QDRANT_IMAGE" >/dev/null 2>&1 \
    || die "image ${QDRANT_IMAGE} is still unavailable after pulling"
}

ensure_volume() {
  docker volume inspect "$QDRANT_VOLUME" >/dev/null 2>&1 \
    || { log "creating docker volume ${QDRANT_VOLUME}"; docker volume create "$QDRANT_VOLUME" >/dev/null; }
}

runtime_flag() {
  # A long list of MUSA containers shares this daemon and mthreads is its default runtime;
  # Qdrant is a plain amd64 image, so pin the stock runc runtime when it is available.
  if [ -n "$QDRANT_RUNTIME" ] && docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q "\"${QDRANT_RUNTIME}\""; then
    printf '%s' "--runtime=${QDRANT_RUNTIME}"
  fi
}

wait_healthy() {
  local timeout_s="${1:-120}"
  local start now body
  start=$(date +%s)
  while :; do
    body="$(curl -fsS --max-time 3 "${QDRANT_HTTP_URL}/healthz" 2>/dev/null || true)"
    if [ -n "$body" ]; then
      log "Qdrant is healthy: ${body}"
      return 0
    fi
    now=$(date +%s)
    if [ $((now - start)) -ge "$timeout_s" ]; then
      docker logs --tail 40 "$QDRANT_CONTAINER" >&2 || true
      return 1
    fi
    sleep 2
  done
}

print_status() {
  log "container state:"
  docker ps -a --filter "name=^/${QDRANT_CONTAINER}$" \
    --format '  name={{.Names}} image={{.Image}} status={{.Status}} ports={{.Ports}}'
  log "image: ${QDRANT_IMAGE}"
  docker image inspect "$QDRANT_IMAGE" --format '  id={{.Id}} created={{.Created}}' 2>/dev/null || true
  local version_json collections_json
  version_json="$(curl -fsS --max-time 5 "${QDRANT_HTTP_URL}/" 2>/dev/null || true)"
  collections_json="$(curl -fsS --max-time 5 "${QDRANT_HTTP_URL}/collections" 2>/dev/null || true)"
  if [ -n "$version_json" ]; then
    log "version: $(printf '%s' "$version_json" | json_field version)"
  else
    log "version: unknown (server not answering)"
  fi
  log "collections: ${collections_json:-no response}"
  docker stats --no-stream --format '  mem_usage={{.MemUsage}} cpu={{.CPUPerc}}' "$QDRANT_CONTAINER" 2>/dev/null || true
  log "volume ${QDRANT_VOLUME} size: $(docker system df -v 2>/dev/null | awk -v v="$QDRANT_VOLUME" '$1==v {print $3, $4}' || true)"
}

main() {
  local action="${1:-deploy}"
  local dry_run="${QDRANT_DRY_RUN:-0}"
  for arg in "$@"; do [ "$arg" = "--dry-run" ] && dry_run=1; done
  [ "$action" = "--dry-run" ] && action=deploy

  require_docker
  check_port "$QDRANT_HTTP_PORT"
  check_port "$QDRANT_GRPC_PORT"

  if [ "$action" = "status" ]; then
    print_status
    return 0
  fi
  [ "$action" = "deploy" ] || die "unknown action '${action}' (use: deploy | status)"

  ensure_image
  ensure_volume

  local run_args=(-d
    --name "$QDRANT_CONTAINER"
    --restart unless-stopped
    -p "${QDRANT_BIND}:${QDRANT_HTTP_PORT}:6333"
    -p "${QDRANT_BIND}:${QDRANT_GRPC_PORT}:6334"
    -v "${QDRANT_VOLUME}:/qdrant/storage"
    -e QDRANT__TELEMETRY_DISABLED=true
    -e QDRANT__SERVICE__ENABLE_TLS=false
    --log-driver json-file --log-opt max-size=50m --log-opt max-file=3)
  local runtime
  runtime="$(runtime_flag)"
  [ -n "$runtime" ] && run_args+=("$runtime")
  run_args+=("$QDRANT_IMAGE")

  if [ "$dry_run" = "1" ]; then
    printf 'DRY RUN: docker run'
    printf ' %q' "${run_args[@]}"
    printf '\n'
    return 0
  fi

  if docker ps -a --format '{{.Names}}' | grep -qx "$QDRANT_CONTAINER"; then
    if docker ps --format '{{.Names}}' | grep -qx "$QDRANT_CONTAINER"; then
      log "container ${QDRANT_CONTAINER} already running"
    else
      log "starting existing container ${QDRANT_CONTAINER}"
      docker start "$QDRANT_CONTAINER" >/dev/null
    fi
  else
    log "creating container ${QDRANT_CONTAINER}"
    docker run "${run_args[@]}" >/dev/null
  fi

  wait_healthy 120 || die "container started but ${QDRANT_HTTP_URL}/healthz did not answer in time"
  print_status
  log "done; HTTP ${QDRANT_HTTP_URL}, gRPC ${QDRANT_BIND}:${QDRANT_GRPC_PORT}, volume ${QDRANT_VOLUME}"
}

main "$@"
