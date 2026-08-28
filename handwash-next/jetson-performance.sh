#!/bin/bash
# Jetson performance helper for Argus.
#
# TensorRT throughput on a Jetson is gated as much by the board's power
# mode and clock governor as by anything in the code: default power modes
# and unlocked clocks routinely cost 30-50% of sustained inference rate,
# and thermal throttling takes the rest on a hot board. This script makes
# that one-command instead of folklore:
#
#   ./jetson-performance.sh            report current mode, clocks, temps
#   sudo ./jetson-performance.sh --max     switch to the MAXN-class mode
#                                          (if this device has one) and
#                                          lock clocks with jetson_clocks
#   sudo ./jetson-performance.sh --set ID  switch to an explicit mode id
#                                          from the report, then lock clocks
#   ./jetson-performance.sh --check    quiet advisory: prints one hint if
#                                          a faster mode is available,
#                                          nothing otherwise (run_nanoowl.sh
#                                          calls this before launching)
#
# The right mode id is NOT the same on every Jetson (MAXN is 0 on an AGX
# Orin but not on other boards, and some devices have no MAXN at all), so
# nothing here hardcodes an id - modes are read from this device's own
# nvpmodel.conf. nvpmodel choices persist across reboots; jetson_clocks
# does NOT, so re-run --max after a reboot (or wire it into a systemd
# unit). On a non-Jetson machine every action is a clean no-op.
#
# NVPMODEL_CONF can override the conf path (used by the test suite).
set -u

CONF="${NVPMODEL_CONF:-/etc/nvpmodel.conf}"
ACTION="${1:-report}"

have() { command -v "$1" >/dev/null 2>&1; }

if ! have nvpmodel; then
    if [ "$ACTION" = "--check" ]; then
        exit 0
    fi
    echo "nvpmodel not found - this is not a Jetson (or the L4T tools"
    echo "are not on PATH), so there is nothing to tune here."
    exit 0
fi

# "ID NAME" per line, from this device's own mode table.
list_modes() {
    [ -r "$CONF" ] || return 0
    sed -n 's/^< *POWER_MODEL *ID=\([0-9][0-9]*\) *NAME=\([^ >]*\).*$/\1 \2/p' "$CONF"
}

current_name() { nvpmodel -q 2>/dev/null | sed -n '1s/^NV Power Mode: *//p'; }
current_id() { nvpmodel -q 2>/dev/null | sed -n '2p'; }

# The MAXN-class mode, if this device has one ("MAXN", "MAXN_SUPER", ...).
# When several match, the last listed wins (later entries are the newer,
# faster variants). Prints "ID NAME" or nothing.
best_maxn() {
    list_modes | awk 'toupper($2) ~ /MAXN/ {id=$1; name=$2}
                      END {if (id != "") print id " " name}'
}

lock_clocks() {
    if have jetson_clocks; then
        if jetson_clocks; then
            echo "Clocks locked to maximum (jetson_clocks)."
            echo "Note: this does not survive a reboot - re-run after booting."
        else
            echo "jetson_clocks failed - clocks are still governed."
        fi
    else
        echo "jetson_clocks not found - skipping clock lock."
    fi
}

show_temps() {
    local zone type temp
    for zone in /sys/class/thermal/thermal_zone*; do
        [ -r "$zone/type" ] && [ -r "$zone/temp" ] || continue
        type="$(cat "$zone/type")"
        temp="$(cat "$zone/temp")"
        printf '  %-16s %s.%s C\n' "$type" "$((temp / 1000))" "$(((temp % 1000) / 100))"
    done
}

case "$ACTION" in

report)
    echo "Current power mode: $(current_name || true) (id $(current_id || true))"
    echo
    echo "Available modes ($CONF):"
    list_modes | sed 's/^/  id /'
    echo
    if have jetson_clocks; then
        echo "Clock state (jetson_clocks --show; may need sudo):"
        jetson_clocks --show 2>/dev/null || echo "  (not readable without sudo)"
    fi
    echo
    echo "Thermal zones:"
    show_temps
    echo
    best="$(best_maxn)"
    if [ -n "$best" ]; then
        echo "For maximum sustained inference: sudo $0 --max"
    else
        echo "No MAXN-class mode on this device. Pick the highest-wattage"
        echo "mode your power supply and cooling support: sudo $0 --set ID"
    fi
    ;;

--check)
    best="$(best_maxn)"
    [ -n "$best" ] || exit 0
    best_id="${best%% *}"
    best_name="${best#* }"
    now_id="$(current_id)"
    if [ -n "$now_id" ] && [ "$now_id" != "$best_id" ]; then
        echo "Power mode is $(current_name) - NanoOWL will run well below its"
        echo "potential. For full speed: sudo ./jetson-performance.sh --max ($best_name + locked clocks)"
    fi
    exit 0
    ;;

--max)
    if [ "$(id -u)" -ne 0 ]; then
        echo "Needs root: sudo $0 --max"
        exit 1
    fi
    best="$(best_maxn)"
    if [ -z "$best" ]; then
        echo "This device has no MAXN-class mode. Available modes:"
        list_modes | sed 's/^/  id /'
        echo "Pick one explicitly: sudo $0 --set ID"
        exit 1
    fi
    best_id="${best%% *}"
    best_name="${best#* }"
    if nvpmodel -m "$best_id"; then
        echo "Power mode set to $best_name (id $best_id). Persists across reboots."
    else
        echo "nvpmodel -m $best_id failed."
        exit 1
    fi
    lock_clocks
    echo "Watch thermals under sustained load - throttling undoes all of"
    echo "this. Current temps:"
    show_temps
    ;;

--set)
    if [ "$(id -u)" -ne 0 ]; then
        echo "Needs root: sudo $0 --set ID"
        exit 1
    fi
    MODE_ID="${2:-}"
    if [ -z "$MODE_ID" ] || ! list_modes | awk '{print $1}' | grep -qx "$MODE_ID"; then
        echo "Pass a mode id from this device's table:"
        list_modes | sed 's/^/  id /'
        exit 1
    fi
    if nvpmodel -m "$MODE_ID"; then
        echo "Power mode set to id $MODE_ID. Persists across reboots."
    else
        echo "nvpmodel -m $MODE_ID failed."
        exit 1
    fi
    lock_clocks
    ;;

*)
    echo "Usage: $0 [--check | --max | --set ID]   (no action = report)"
    exit 1
    ;;
esac
