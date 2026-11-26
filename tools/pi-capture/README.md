# Pi air-capture testbed

The over-the-air BFI Action frame sniffer described in Section 4 of the paper.
Runs on a Raspberry Pi 5 + Intel AX210 NIC in monitor mode on 5 GHz channel 40.
Decodes HE compressed beamforming Action frames from a stock 802.11ax router with
no firmware patch and no association credentials.

## Files

| File | Purpose |
|---|---|
| `bfi_capture.py` | Scapy-based sniffer. Decodes 802.11 management frames matching the HE/VHT compressed-beamforming Action category. Writes one row per BFI frame to a daily-rotated SQLite. |
| `setup_monitor.sh` | Idempotent script that creates `mon0` on the AX210, sets the channel, brings the interface up. Runs as `ExecStartPre` so every service start (cold or post-stall-exit) rebuilds the monitor cleanly. |
| `bfi_status.sh` | Quick status query: service state, today's frame count, top STAs by frame count, MIMO Control distribution, disk usage. |
| `bfi-capture.service` | Systemd unit. `Type=notify`, `Restart=always`, `WatchdogSec=120`, `User=root`. |

## Stall-exit watchdog (added 2026-05-14)

The original heartbeat thread sent `WATCHDOG=1` to systemd every 30 s unconditionally,
even when no 802.11 frames were reaching the sniffer. This produced a silent-failure
mode: the radio detuned (e.g., after a kernel update or a wake-from-suspend event),
`mon0` stayed `UP, LOWER_UP` from the kernel's perspective, but the driver delivered
zero frames. The python process kept reporting healthy, systemd never restarted it,
and capture was dead for 73 hours before being noticed.

`bfi_capture.py` now self-exits when zero 802.11 frames have arrived for
`--stall-exit-s` seconds (default 600). systemd's `Restart=always` plus the
`ExecStartPre=setup_monitor.sh` rebuilds `mon0` on the right channel on every
restart, so the failure mode self-heals within ~10 minutes.

The exit path is `os.kill(getpid(), SIGTERM)` from the heartbeat thread; it
sends `STOPPING=1` to systemd first so the journal records the stall and the
exit reason.

The cold-start case (zero frames since boot) is excluded from the trigger so a
slow first frame after channel hop does not loop the service.

## Deploy on a fresh Pi

The service file expects:
- The Pi user `pi` (adjust `User=` and `WorkingDirectory=` to taste)
- The Intel AX210 NIC on `wlan0` (the `setup_monitor.sh` script picks the
  parent interface and creates `mon0` as the monitor child)
- The DB directory at `/var/lib/bfi-capture` (created by the script if missing)

Install:

```bash
sudo cp tools/pi-capture/bfi-capture.service /etc/systemd/system/
sudo mkdir -p /home/pi/bfi-capture
sudo cp tools/pi-capture/{bfi_capture.py,setup_monitor.sh,bfi_status.sh} \
        /home/pi/bfi-capture/
sudo chown -R pi:pi /home/pi/bfi-capture
sudo chmod +x /home/pi/bfi-capture/{bfi_capture.py,setup_monitor.sh,bfi_status.sh}
sudo systemctl daemon-reload
sudo systemctl enable --now bfi-capture
```

Verify after ~30 s:

```bash
bash tools/pi-capture/bfi_status.sh
```

Expected output: BFI rows growing, last frame within the last few seconds,
non-empty Top STAs.

## Configuration knobs

`bfi_capture.py` flags (set on `ExecStart=` in the unit file):

| Flag | Default | Purpose |
|---|---|---|
| `--iface` | `mon0` | The monitor interface produced by `setup_monitor.sh`. |
| `--db-dir` | `/var/lib/bfi-capture` | Daily-rotated SQLite goes here as `bfi_YYYY-MM-DD.sqlite`. |
| `--log-every` | 100 | Print a `[stat]` line every N 802.11 frames seen. |
| `--heartbeat-s` | 30 | systemd `WATCHDOG=1` cadence. |
| `--stall-warn-s` | 300 | `[heartbeat:warn]` if no frames for this long. |
| `--stall-exit-s` | 600 | Self-exit if no frames for this long (lets systemd rebuild `mon0`). |
| `--commit-every` | 16 | SQLite commit batch size. |
