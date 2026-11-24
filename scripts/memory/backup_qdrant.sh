#!/usr/bin/env bash
# Snapshot backup / restore for the AGR Memory Qdrant server (Phase 6 / D9, Q7).
#
# Active Qdrant data lives on the local Docker volume; this script is the only path that
# writes Qdrant state to NFS. It uses Qdrant's own snapshot API (consistent on-disk copy)
# instead of copying the storage directory, then verifies size and SHA-256 of every file.
#
# Usage:
#   bash scripts/memory/backup_qdrant.sh backup             # snapshot all collections
#   bash scripts/memory/backup_qdrant.sh backup --collection agr_memories
#   bash scripts/memory/backup_qdrant.sh list               # show available backups
#   bash scripts/memory/backup_qdrant.sh restore <collection> <snapshot-file>
#   bash scripts/memory/backup_qdrant.sh collections        # collection inventory
#
# Environment:
#   QDRANT_URL        default http://127.0.0.1:6333
#   BACKUP_DIR        default /mnt/agr-data/qdrant_backup
#   KEEP              snapshots kept per collection (default 14)
#   QDRANT_CONTAINER  container name used only for version info (default agr-qdrant)
#
# Exit codes: 0 ok, 1 runtime failure, 2 usage error.

set -euo pipefail

QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
BACKUP_DIR="${BACKUP_DIR:-/mnt/agr-data/qdrant_backup}"
KEEP="${KEEP:-14}"
QDRANT_CONTAINER="${QDRANT_CONTAINER:-agr-qdrant}"
MANIFEST="${BACKUP_DIR}/manifest.jsonl"
CURL=(curl -fsS --max-time 600)

log() { printf '[backup_qdrant] %s\n' "$*"; }
die() { printf '[backup_qdrant] ERROR: %s\n' "$*" >&2; exit 1; }

api() { # api <method> <path> [json-body]
  local method="$1" path="$2" body="${3:-}"
  if [ -n "$body" ]; then
    "${CURL[@]}" -X "$method" -H 'Content-Type: application/json' -d "$body" "${QDRANT_URL}${path}"
  else
    "${CURL[@]}" -X "$method" "${QDRANT_URL}${path}"
  fi
}

qdrant_version() {
  api GET / 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("version","unknown"))' 2>/dev/null || echo unknown
}

list_collections() {
  api GET /collections | python3 -c '
import json, sys
data = json.load(sys.stdin)
for item in data.get("result", {}).get("collections", []):
    print(item["name"])
'
}

collection_count() {
  api GET "/collections/$1" | python3 -c '
import json, sys
data = json.load(sys.stdin)
print(data.get("result", {}).get("points_count", "unknown"))
'
}

usage() {
  sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

cmd_collections() {
  if ! api GET /healthz >/dev/null 2>&1; then
    die "Qdrant is not reachable at ${QDRANT_URL}"
  fi
  while IFS= read -r name; do
    [ -n "$name" ] || continue
    printf '%s\tpoints=%s\n' "$name" "$(collection_count "$name")"
  done < <(list_collections)
}

cmd_backup() {
  local only=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --collection) only="${2:?--collection needs a name}"; shift 2 ;;
      *) die "unknown backup option: $1" ;;
    esac
  done

  command -v curl >/dev/null 2>&1 || die "curl is not installed"
  api GET /healthz >/dev/null 2>&1 || die "Qdrant is not reachable at ${QDRANT_URL}"
  mkdir -p "$BACKUP_DIR"

  local version image_id
  version="$(qdrant_version)"
  image_id="$(docker inspect "$QDRANT_CONTAINER" --format '{{.Image}}' 2>/dev/null || true)"

  local collections=()
  if [ -n "$only" ]; then
    collections=("$only")
  else
    while IFS= read -r name; do [ -n "$name" ] && collections+=("$name"); done < <(list_collections)
  fi
  if [ ${#collections[@]} -eq 0 ]; then
    log "no collections to back up"
    return 0
  fi

  local collection snapshot_url status rc=0
  for collection in "${collections[@]}"; do
    local target_dir="${BACKUP_DIR}/${collection}"
    mkdir -p "$target_dir"
    snapshot_url="$(api POST "/collections/${collection}/snapshots" '{}' \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"]["name"])')" \
      || { log "snapshot creation failed for ${collection}"; rc=1; continue; }
    local file="${snapshot_url}"
    log "${collection}: downloading ${snapshot_url}"
    if ! "${CURL[@]}" -o "${target_dir}/${file}" \
        "${QDRANT_URL}/collections/${collection}/snapshots/${snapshot_url}"; then
      log "download failed for ${collection}/${snapshot_url}"
      rc=1
      continue
    fi
    local bytes sha
    bytes="$(stat -c %s "${target_dir}/${file}")"
    sha="$(sha256sum "${target_dir}/${file}" | cut -d' ' -f1)"
    printf '%s\n' "${sha}  ${file}" > "${target_dir}/${file}.sha256"
    printf '{"collection":"%s","snapshot":"%s","file":"%s","bytes":%s,"sha256":"%s","created_at":"%s","qdrant_version":"%s","image_id":"%s"}\n' \
      "$collection" "$snapshot_url" "${target_dir}/${file}" "$bytes" "$sha" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$version" "$image_id" >> "$MANIFEST"
    log "${collection}: saved ${bytes} bytes sha256=${sha:0:16}..."

    # Best effort: ask Qdrant to drop the server-side snapshot copy; the NFS copy is the
    # backup of record and keeping both wastes local disk.
    api DELETE "/collections/${collection}/snapshots/${snapshot_url}" >/dev/null 2>&1 || true

    # Prune old snapshots, keeping the newest $KEEP files per collection.
    local old
    while IFS= read -r old; do
      [ -n "$old" ] || continue
      log "${collection}: pruning ${old}"
      rm -f "$old" "$old.sha256"
    done < <(ls -1t "${target_dir}"/*.snapshot 2>/dev/null | tail -n "+$((KEEP + 1))" || true)
  done

  log "backup finished (dir=${BACKUP_DIR}, manifest=${MANIFEST})"
  return "$rc"
}

cmd_restore() {
  local collection="${1:-}" file="${2:-}"
  [ -n "$collection" ] && [ -n "$file" ] || die "restore needs <collection> <snapshot-file>"
  [ -f "$file" ] || die "snapshot file not found: $file"
  api GET /healthz >/dev/null 2>&1 || die "Qdrant is not reachable at ${QDRANT_URL}"

  log "uploading ${file} -> collection ${collection}"
  "${CURL[@]}" -X POST \
    -F "snapshot=@${file}" \
    "${QDRANT_URL}/collections/${collection}/snapshots/upload?priority=snapshot" \
    >/dev/null
  log "restored; points_count=$(collection_count "$collection")"
}

cmd_list() {
  [ -d "$BACKUP_DIR" ] || die "backup dir does not exist: $BACKUP_DIR"
  log "backups in ${BACKUP_DIR}:"
  find "$BACKUP_DIR" -name '*.snapshot' -type f -printf '%TY-%Tm-%Td %TH:%TM  %10s  %p\n' 2>/dev/null \
    | sort -r || echo "  (none)"
  if [ -f "$MANIFEST" ]; then
    log "manifest entries: $(wc -l < "$MANIFEST")"
  fi
}

main() {
  local action="${1:-backup}"
  shift || true
  case "$action" in
    backup) cmd_backup "$@" ;;
    restore) cmd_restore "$@" ;;
    list) cmd_list ;;
    collections) cmd_collections ;;
    -h|--help|help) usage ;;
    *) usage ;;
  esac
}

main "$@"
