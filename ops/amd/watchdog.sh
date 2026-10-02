#!/usr/bin/env bash
# The dead-man switch. Runs on the droplet, on a timer, and stops the meter.
#
#   watchdog.sh [--once] --idle-minutes N --max-minutes N
#
# A powered-off GPU droplet still bills: AMD and DigitalOcean both say the disk, CPU, RAM and IP
# stay reserved and charges accrue until the instance is *destroyed*. So the switch powers off
# first (which stops the GPU doing work and is recoverable from a snapshot) and then destroys.
#
# Three independent reasons to fire, any one of which is sufficient:
#   wall clock  --max-minutes   the budget ceiling, derived from the price and the cap
#   idle        --idle-minutes  no job process and no ssh session for this long
#   heartbeat   --heartbeat-file  a file the laptop touches; if it goes stale the laptop is gone
#
# The heartbeat is the one that matters for the failure nobody sees: the laptop sleeps, the ssh
# connection drops, the sweep keeps running, and the only signal is that nothing has touched the
# heartbeat. Every `driver.py` step touches it.

set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

IDLE_MIN="${AMD_IDLE_LIMIT_MIN}"
MAX_MIN="${AMD_WALLCLOCK_LIMIT_MIN}"
HEARTBEAT="${AMD_HEARTBEAT_FILE:-/opt/smol-ladder/.heartbeat}"
ONCE=0
DESTROY=1
STATE_FILE="${AMD_STATE_FILE:-/opt/smol-ladder/.watchdog-state}"
IDLE_FILE="${AMD_IDLE_FILE:-/opt/smol-ladder/.watchdog-idle}"
LOG="$AMD_REMOTE_LOG/watchdog.log"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --idle-minutes)   IDLE_MIN="$2"; shift 2 ;;
    --max-minutes)    MAX_MIN="$2"; shift 2 ;;
    --heartbeat-file) HEARTBEAT="$2"; shift 2 ;;
    --no-destroy)     DESTROY=0; shift ;;
    --once)           ONCE=1; shift ;;
    *) amd_die "unknown argument '$1'" ;;
  esac
done

mkdir -p "$(dirname "$LOG")"

START_EPOCH="$(date +%s)"
if [[ -f "$STATE_FILE" ]]; then
  # Read back across invocations, so a --once timer every minute accumulates one budget rather
  # than restarting it every time.
  read -r START_EPOCH < "$STATE_FILE" || START_EPOCH="$(date +%s)"
fi
printf '%s\n' "$START_EPOCH" > "$STATE_FILE"

elapsed_min=$(( ( $(date +%s) - START_EPOCH ) / 60 ))
amd_log "watchdog: elapsed ${elapsed_min}m, cap ${MAX_MIN}m, idle cap ${IDLE_MIN}m"

reason=""

if (( elapsed_min >= MAX_MIN )); then
  spent="$(python3 -c "print(f'{${elapsed_min}/60*${AMD_PRICE_PER_GPU_HOUR}:.2f}')")"
  reason="wall-clock cap reached: ${elapsed_min}m of ${MAX_MIN}m (~\$${spent} at \$${AMD_PRICE_PER_GPU_HOUR}/h)"
fi

# Idle: no sft_lora / run_ladder / vllm process, and no ssh session.
if [[ -z "$reason" ]]; then
  BUSY="$(pgrep -f 'train\.sft_lora|smol_ladder\.run_ladder|vllm serve' | wc -l)"
  if (( BUSY == 0 )); then
    IDLE_FOR=0
    if [[ -f "$IDLE_MINUTES_FILE" ]]; then
      read -r IDLE_FOR < "$IDLE_FILE" || IDLE_FOR=0
    fi
    IDLE_FOR=$(( IDLE_FOR + 1 ))
    printf '%s\n' "$IDLE_FOR" > "$IDLE_FILE"
    if (( IDLE_FOR >= IDLE_MIN )); then
      reason="idle ${IDLE_FOR}m with no training, eval or vLLM process running"
    fi
  else
    printf '0\n' > "$IDLE_FILE"
  fi
fi

# Heartbeat: the laptop touches this file between steps. Stale means the orchestrator is gone.
if [[ -z "$reason" && -f "$HEARTBEAT" ]]; then
  HB_AGE=$(( ( $(date +%s) - $(stat -c %Y "$HEARTBEAT") ) / 60 ))
  if (( HB_AGE > IDLE_MIN )); then
    reason="heartbeat stale for ${HB_AGE}m: the orchestrating laptop is unreachable"
  fi
fi

if [[ -n "$reason" ]]; then
  amd_log "FIRING: $reason"
  # Everything that is not the Hub push or the rsync is lost. Push and sync before poweroff, in
  # that order: the Hub copy is the one that survives if the droplet dies during the sync.
  if [[ -x "$AMD_REMOTE_ROOT/ops/amd/sync_back.sh" ]]; then
    amd_log "attempting a final sync before poweroff"
    "$AMD_REMOTE_ROOT/ops/amd/sync_back.sh" --push-hub --to-laptop || \
      amd_log "sync_back failed; the run is lost"
  fi
  shutdown -h now &
  sleep 20
  if (( DESTROY )); then
    # Nothing to destroy from inside: a droplet cannot destroy itself. The laptop does that with
    # the API token (driver.py --destroy) or the owner does it in the console. Powering off is the
    # best an on-instance switch can do, which is why the runbook's teardown is the last step.
    amd_log "powered off. DESTROY THE DROPLET IN THE CONSOLE OR VIA 'driver.py --destroy'."
  fi
else
  amd_log "no trip condition; staying up"
fi

(( ONCE )) || exit 0
