#!/bin/bash
# Pre-start script for bfi-capture.service. Idempotent: safe to re-run.
# Tears down wlan0, creates a monitor interface on the AX210 phy, and
# (optionally) auto-discovers the busiest 5 GHz channel.
#
# Channel selection order:
#   1. If $FREQ is set in the environment, use it (with $CENTER if set).
#   2. Otherwise, hop the candidate list and pick the channel with the most
#      Action management frames in $PROBE_S seconds.
#   3. Fall back to the prior default (channel 40, 80 MHz, center 5210 MHz).

set -e

PHY="${PHY:-phy1}"
IFACE="${IFACE:-mon0}"
WIDTH="${WIDTH:-80}"
PROBE_S="${PROBE_S:-3}"
DEFAULT_FREQ="${DEFAULT_FREQ:-5200}"
DEFAULT_CENTER="${DEFAULT_CENTER:-5210}"
CANDIDATE_CHANS="${CANDIDATE_CHANS:-36 40 44 48 149 153 157 161 165}"

center_for_chan() {
    case "$1" in
        36|40|44|48)         echo 5210 ;;
        52|56|60|64)         echo 5290 ;;
        100|104|108|112)     echo 5530 ;;
        116|120|124|128)     echo 5610 ;;
        132|136|140|144)     echo 5690 ;;
        149|153|157|161)     echo 5775 ;;
        *)                   echo "" ;;
    esac
}

freq_for_chan() {
    echo $((5000 + $1 * 5))
}

# 1. Tear down NetworkManager-managed wlan0 if present.
if iw dev wlan0 info >/dev/null 2>&1; then
    nmcli dev disconnect wlan0 2>/dev/null || true
    iw dev wlan0 del 2>/dev/null || true
fi

# 2. Create monitor interface on the AX210 phy if missing.
ACTUAL_PHY="$PHY"
if ! iw phy "$ACTUAL_PHY" info >/dev/null 2>&1; then
    ACTUAL_PHY="$(iw phy 2>/dev/null | awk '/Wiphy/ {print $2; exit}')"
fi
if ! iw dev "$IFACE" info >/dev/null 2>&1; then
    iw phy "$ACTUAL_PHY" interface add "$IFACE" type monitor
fi
ip link set "$IFACE" up

# 3. Pick a frequency.
if [ -n "$FREQ" ]; then
    PICK_FREQ="$FREQ"
    PICK_CENTER="${CENTER:-$DEFAULT_CENTER}"
    SOURCE="env"
elif command -v tcpdump >/dev/null 2>&1; then
    BEST_CH=""
    BEST_N=0
    for ch in $CANDIDATE_CHANS; do
        f="$(freq_for_chan "$ch")"
        c="$(center_for_chan "$ch")"
        [ -z "$c" ] && continue
        iw dev "$IFACE" set freq "$f" "$WIDTH" "$c" >/dev/null 2>&1 || continue
        n=$(timeout "$PROBE_S" tcpdump -i "$IFACE" -nn -c 200 \
                "type mgt subtype action" 2>/dev/null | wc -l)
        echo "  probe ch=$ch freq=$f action_frames=$n"
        if [ "$n" -gt "$BEST_N" ]; then
            BEST_N="$n"
            BEST_CH="$ch"
        fi
    done
    if [ -n "$BEST_CH" ] && [ "$BEST_N" -gt 0 ]; then
        PICK_FREQ="$(freq_for_chan "$BEST_CH")"
        PICK_CENTER="$(center_for_chan "$BEST_CH")"
        SOURCE="auto(ch=$BEST_CH action_frames=$BEST_N)"
    else
        PICK_FREQ="$DEFAULT_FREQ"
        PICK_CENTER="$DEFAULT_CENTER"
        SOURCE="default(no traffic seen)"
    fi
else
    PICK_FREQ="$DEFAULT_FREQ"
    PICK_CENTER="$DEFAULT_CENTER"
    SOURCE="default(tcpdump missing)"
fi

iw dev "$IFACE" set freq "$PICK_FREQ" "$WIDTH" "$PICK_CENTER" || true
echo "Monitor $IFACE up on phy $ACTUAL_PHY freq $PICK_FREQ ${WIDTH}MHz center $PICK_CENTER  source=$SOURCE"
