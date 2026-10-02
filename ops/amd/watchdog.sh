#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# The on-droplet watchdog. Armed by entrypoint.sh; stops wasted GPU time from the inside.
#
#   watchdog.sh --arm                      loop forever
#   watchdog.sh --once [--dry-run]         one tick (the tests use this)
#
# Three independent trip conditions, any one sufficient:
#   wall clock  elapsed >= --max-minutes          (default 720; the laptop's dead-man switch is stricter)
#   idle        no work process for --idle-minutes (default 30): nothing is using the GPU
#   no laptop   no established ssh session for --ssh-minutes (default 20): the tunnel and every
#               driver step are ssh sessions, so none at all means the laptop is gone
#
# What a trip does: push everything to the Hub (the copy that survives), then power the droplet
# off. A droplet cannot destroy itself and a powered-off one still bills, so the power-off is a
# SIGNAL: the laptop's deadman.py destroys any tagged droplet it finds powered off. The two
# switches are deliberately different in kind: this one lives inside what it protects, so it only
# ever saves the work; the one outside is the one that stops the meter.

set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MAX_MIN="${AMD_WATCHDOG_MAX_MIN:-720}"
IDLE_MIN="${AMD_IDLE_LIMIT_MIN:-30}"
SSH_MIN="${AMD_SSH_LIMIT_MIN:-20}"
TICK="${AMD_WATCHDOG_TICK:-60}"
STATE_DIR="${AMD_STATE_DIR:-$AMD_REMOTE_ROOT/.watchdog}"
WORK_PATTERN='train\.sft_lora|ops\.amd\.sft_run|vllm\.entrypoints|vllm serve|smoke\.sh|run_sft\.sh|serve\.sh|sync_back\.sh|bench\.py|merge_adapter'
ONCE=0; ARM=0; DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-minutes)  MAX_MIN="$2"; shift 2 ;;
    --idle-minutes) IDLE_MIN="$2"; shift 2 ;;
    --ssh-minutes)  SSH_MIN="$2"; shift 2 ;;
    --tick)         TICK="$2"; shift 2 ;;
    --state-dir)    STATE_DIR="$2"; shift 2 ;;
    --once)         ONCE=1; shift ;;
    --arm)          ARM=1; shift ;;
    --dry-run)      DRY=1; shift ;;
    *) amd_die "unknown argument '$1'" ;;
  esac
done
mkdir -p "$STATE_DIR" "$AMD_REMOTE_LOG"

# Minutes since a counter file was last reset, kept as a count of ticks-in-a-row.
bump() { local f="$STATE_DIR/$1" n=0; [[ -f "$f" ]] && read -r n < "$f"; n=$((n + 1)); printf '%s\n' "$n" > "$f"; printf '%s\n' "$n"; }
reset() { printf '0\n' > "$STATE_DIR/$1"; }

tick() {
  local start now elapsed reason="" busy ssh_n
  start="$(cat "$STATE_DIR/start" 2>/dev/null || date +%s)"
  printf '%s\n' "$start" > "$STATE_DIR/start"
  now="$(date +%s)"
  elapsed=$(( (now - start) / 60 ))

  (( elapsed >= MAX_MIN )) && reason="wall clock: ${elapsed} min >= ${MAX_MIN} min"

  busy="$(pgrep -fc "$WORK_PATTERN" 2>/dev/null || true)"
  if [[ -z "$reason" ]]; then
    if (( ${busy:-0} > 0 )); then reset idle; else
      local n; n="$(bump idle)"
      (( n * TICK / 60 >= IDLE_MIN )) && reason="idle: no work process for ${IDLE_MIN} min"
    fi
  fi

  ssh_n="$(ss -Htn state established '( sport = :22 )' 2>/dev/null | wc -l)"
  if [[ -z "$reason" ]]; then
    if (( ssh_n > 0 )); then reset ssh; else
      local m; m="$(bump ssh)"
      (( m * TICK / 60 >= SSH_MIN )) && reason="no laptop: no ssh session for ${SSH_MIN} min"
    fi
  fi

  if [[ -z "$reason" ]]; then
    amd_log "tick: ${elapsed}/${MAX_MIN} min, busy=${busy:-0}, ssh=${ssh_n}: no trip"
    return 0
  fi
  amd_log "FIRING: $reason"
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$reason" >> "$AMD_REMOTE_LOG/watchdog.trips"
  if (( DRY )); then amd_log "dry run: would push to the Hub and power off"; return 1; fi
  # sync_back needs torch/hf (the container); the poweroff below needs the host
  amd_in_container bash "$AMD_REMOTE_ROOT/ops/amd/sync_back.sh" --push-hub || amd_log "final push failed"
  amd_log "powering off: the laptop's deadman.py destroys a droplet found powered off"
  shutdown -h now || amd_log "shutdown failed"
  return 1
}

if (( ONCE )); then tick; exit $?; fi
(( ARM )) || amd_die "give --arm or --once"
trap 'amd_log "watchdog stopping"; exit 0' TERM INT
amd_log "watchdog armed: wall ${MAX_MIN} min, idle ${IDLE_MIN} min, no-laptop ${SSH_MIN} min"
while true; do
  tick || exit 0
  sleep "$TICK"
done
