#!/bin/bash
# Quick status query for the BFI capture service.

DB_DIR="${DB_DIR:-/var/lib/bfi-capture}"
TODAY="$DB_DIR/bfi_$(date -I).sqlite"

echo "=== Service status ==="
systemctl status bfi-capture --no-pager 2>&1 | head -10

echo
echo "=== Today's database: $TODAY ==="
if [ -f "$TODAY" ]; then
    n=$(sqlite3 "$TODAY" "SELECT COUNT(*) FROM bfi")
    earliest=$(sqlite3 "$TODAY" "SELECT MIN(ts) FROM bfi" 2>/dev/null)
    latest=$(sqlite3 "$TODAY" "SELECT MAX(ts) FROM bfi" 2>/dev/null)
    echo "BFI rows: $n"
    if [ -n "$earliest" ] && [ -n "$latest" ] && [ "$earliest" != "" ]; then
        echo "first frame: $(date -d @$earliest 2>/dev/null || echo $earliest)"
        echo "last  frame: $(date -d @$latest 2>/dev/null || echo $latest)"
    fi
    echo
    echo "Top STAs (src_mac):"
    sqlite3 -column -header "$TODAY" \
        "SELECT src_mac, COUNT(*) AS n FROM bfi GROUP BY src_mac ORDER BY n DESC LIMIT 10"
    echo
    echo "MIMO Control distribution:"
    sqlite3 -column -header "$TODAY" \
        "SELECT mimo_ctrl, COUNT(*) AS n FROM bfi GROUP BY mimo_ctrl ORDER BY n DESC LIMIT 5"
else
    echo "(no captures yet today)"
fi

echo
echo "=== Disk ==="
ls -lah "$DB_DIR" 2>/dev/null | head -15
df -h "$DB_DIR" 2>/dev/null | tail -2
