#!/usr/bin/env python3
"""Long-running BFI capture service.

Reads HE/VHT Compressed Beamforming Reports from a monitor-mode
interface and writes parsed metadata to daily-rotated SQLite databases.
Runs forever; relies on systemd Restart=always for resilience.

Output schema (per BFI frame):
    ts          REAL    Unix timestamp (radiotap mactime if available)
    src_mac     TEXT    transmitter (the STA sending BFI)
    ap_mac      TEXT    receiver (the AP)
    category    INTEGER 0x1e (HE) or 0x15 (VHT)
    subtype     INTEGER 13 (Action) or 14 (Action No-Ack)
    mimo_ctrl   TEXT    HE: 5-byte hex; VHT: 3-byte hex
    body_hex    TEXT    full action body hex
    body_len    INTEGER bytes
    rssi        INTEGER dBm (best effort, may be NULL)
    chan_freq   INTEGER MHz (from radiotap if available)
    mactime     INTEGER radiotap MAC timestamp counter (us since radio up)
"""
from __future__ import annotations

import argparse
import datetime
import os
import signal
import socket
import sqlite3
import sys
import threading
import time
from pathlib import Path

from scapy.all import sniff, Dot11, RadioTap


def _sd_notify(msg: str) -> None:
    """Send a sd_notify datagram if running under systemd notify type."""
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return
    if addr.startswith("@"):
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.sendto(msg.encode("utf-8"), addr)
    except OSError:
        pass

CATEGORY_HE = 0x1e
CATEGORY_VHT = 0x15
ACTION_COMPRESSED_BEAMFORMING = 0x00
HE_MIMO_CTRL_LEN = 5
VHT_MIMO_CTRL_LEN = 3

DEFAULT_LOG_EVERY = 100
DEFAULT_HEARTBEAT_S = 30
DEFAULT_STALL_WARN_S = 300
DEFAULT_STALL_EXIT_S = 600  # exit if zero frames for this long; systemd Restart=always + ExecStartPre rebuilds mon0
DEFAULT_COMMIT_EVERY = 16

HE_BW_MAP = {0: 20, 1: 40, 2: 80, 3: 160}
HE_NG_MAP = {0: 4, 1: 16}
HE_FB_MAP = {0: "SU", 1: "MU", 2: "CQI"}
VHT_BW_MAP = {0: 20, 1: 40, 2: 80, 3: 160}
VHT_NG_MAP = {0: 1, 1: 2, 2: 4}
VHT_FB_MAP = {0: "SU", 1: "MU"}

PARSED_COLUMNS = [
    ("nc", "INTEGER"),
    ("nr", "INTEGER"),
    ("bw_mhz", "INTEGER"),
    ("ng", "INTEGER"),
    ("codebook", "INTEGER"),
    ("feedback_type", "TEXT"),
    ("dialog_token", "INTEGER"),
    ("avg_snr_hex", "TEXT"),
    ("seq_num", "INTEGER"),
    ("frag_num", "INTEGER"),
]


def _bits(v: int, start: int, n: int) -> int:
    return (v >> start) & ((1 << n) - 1)


def parse_he_mimo_ctrl(b: bytes) -> dict:
    """Parse 5-byte HE MIMO Control field (IEEE 802.11ax 9.4.1.65).

    Bit layout, LSB first within little-endian byte stream:
        0..2   Nc Index (Nc - 1)
        3..5   Nr Index (Nr - 1)
        6..7   BW                 (0=20, 1=40, 2=80, 3=160 or 80+80)
        8      Grouping (Ng)      (0=4, 1=16)
        9      Codebook info
        10..11 Feedback type      (0=SU, 1=MU, 2=CQI)
        12..14 Remaining feedback segments
        15     First feedback segment
        16..22 RU start index
        23..29 RU end index
        30..35 Sounding dialog token
        36..39 Reserved / disallowed subchannel bitmap present

    Cross-checked against Wireshark's wlan.he.mimo.* bitmasks. Before
    2026-10-07 this parser used the VHT offsets (grouping 8..9, codebook 10,
    feedback type 11..12, token 31..36): stored codebook/ng/dialog_token
    values for HE rows written before then are wrong whenever codebook=1
    (and the token always); re-parse them from body_hex.
    """
    if len(b) < HE_MIMO_CTRL_LEN:
        return {}
    v = int.from_bytes(b[:HE_MIMO_CTRL_LEN], "little")
    return dict(
        nc=_bits(v, 0, 3) + 1,
        nr=_bits(v, 3, 3) + 1,
        bw_mhz=HE_BW_MAP.get(_bits(v, 6, 2)),
        ng=HE_NG_MAP.get(_bits(v, 8, 1)),
        codebook=_bits(v, 9, 1),
        feedback_type=HE_FB_MAP.get(_bits(v, 10, 2)),
        dialog_token=_bits(v, 30, 6),
    )


def parse_vht_mimo_ctrl(b: bytes) -> dict:
    """Parse 3-byte VHT MIMO Control field (IEEE 802.11ac 8.4.1.48).

    Bit layout:
        0..2   Nc Index (Nc - 1)
        3..5   Nr Index (Nr - 1)
        6..7   Channel width      (0=20, 1=40, 2=80, 3=160/80+80)
        8..9   Grouping (Ng)      (0=1, 1=2, 2=4)
        10     Codebook info
        11     Feedback type      (0=SU, 1=MU)
        12..14 Remaining feedback segments
        15     First feedback segment
        16..17 Reserved
        18..23 Sounding dialog token
    """
    if len(b) < VHT_MIMO_CTRL_LEN:
        return {}
    v = int.from_bytes(b[:VHT_MIMO_CTRL_LEN], "little")
    return dict(
        nc=_bits(v, 0, 3) + 1,
        nr=_bits(v, 3, 3) + 1,
        bw_mhz=VHT_BW_MAP.get(_bits(v, 6, 2)),
        ng=VHT_NG_MAP.get(_bits(v, 8, 2)),
        codebook=_bits(v, 10, 1),
        feedback_type=VHT_FB_MAP.get(_bits(v, 11, 1)),
        dialog_token=_bits(v, 18, 6),
    )


class Capture:
    def __init__(
        self,
        db_dir: Path,
        iface: str,
        log_every: int = DEFAULT_LOG_EVERY,
        heartbeat_s: int = DEFAULT_HEARTBEAT_S,
        stall_warn_s: int = DEFAULT_STALL_WARN_S,
        stall_exit_s: int = DEFAULT_STALL_EXIT_S,
        commit_every: int = DEFAULT_COMMIT_EVERY,
    ):
        self.db_dir = db_dir
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.iface = iface
        self.log_every = log_every
        self.heartbeat_s = heartbeat_s
        self.stall_warn_s = stall_warn_s
        self.stall_exit_s = stall_exit_s
        self.commit_every = commit_every
        self.current_date: str | None = None
        self.conn: sqlite3.Connection | None = None
        self.frames_seen = 0
        self.bfi_today = 0
        self.uncommitted = 0
        self.start_ts = time.time()
        self.last_frame_ts = self.start_ts
        self.last_bfi_ts = self.start_ts
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def open_db_for_today(self) -> sqlite3.Connection:
        today = datetime.date.today().isoformat()
        if today == self.current_date and self.conn is not None:
            return self.conn
        if self.conn is not None:
            try:
                self.conn.commit()
                self.conn.close()
            except Exception as e:
                print(f"[error] closing prev db: {e}", file=sys.stderr, flush=True)
        db_path = self.db_dir / f"bfi_{today}.sqlite"
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS bfi (
                ts REAL, src_mac TEXT, ap_mac TEXT,
                category INTEGER, subtype INTEGER,
                mimo_ctrl TEXT, body_hex TEXT,
                body_len INTEGER, rssi INTEGER,
                chan_freq INTEGER, mactime INTEGER
            )"""
        )
        for col, sqltype in PARSED_COLUMNS:
            try:
                conn.execute(f"ALTER TABLE bfi ADD COLUMN {col} {sqltype}")
            except sqlite3.OperationalError:
                pass
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON bfi(ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_src ON bfi(src_mac)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_token ON bfi(dialog_token)")
        conn.commit()
        self.conn = conn
        self.current_date = today
        self.bfi_today = 0
        self.uncommitted = 0
        print(f"[rotate] now writing to {db_path}", flush=True)
        return conn

    def _radiotap_field(self, pkt, name):
        try:
            rt = pkt.getlayer(RadioTap)
            if rt is None:
                return None
            return getattr(rt, name, None)
        except Exception:
            return None

    def _mimo_ctrl_len(self, category: int) -> int:
        return HE_MIMO_CTRL_LEN if category == CATEGORY_HE else VHT_MIMO_CTRL_LEN

    def callback(self, pkt) -> None:
        now = time.time()
        try:
            if Dot11 not in pkt:
                return
            with self._lock:
                self.frames_seen += 1
                self.last_frame_ts = now
                frames_seen = self.frames_seen
            if frames_seen % self.log_every == 0:
                up = now - self.start_ts
                print(
                    f"[stat] frames={frames_seen} bfi_today={self.bfi_today} "
                    f"uptime={up:.0f}s",
                    flush=True,
                )
            d = pkt[Dot11]
            if d.type != 0 or d.subtype not in (13, 14):
                return
            body = bytes(d.payload) if d.payload else b""
            if len(body) < 7:
                return
            cat = body[0]
            act = body[1]
            if cat not in (CATEGORY_HE, CATEGORY_VHT):
                return
            if act != ACTION_COMPRESSED_BEAMFORMING:
                return
            mimo_len = self._mimo_ctrl_len(cat)
            mimo_end = 2 + mimo_len
            if len(body) < mimo_end:
                return
            mimo_ctrl_bytes = body[2:mimo_end]
            if cat == CATEGORY_HE:
                parsed = parse_he_mimo_ctrl(mimo_ctrl_bytes)
            else:
                parsed = parse_vht_mimo_ctrl(mimo_ctrl_bytes)
            nc = parsed.get("nc")
            avg_snr_hex = None
            if nc is not None:
                snr_end = mimo_end + nc
                if len(body) >= snr_end:
                    avg_snr_hex = body[mimo_end:snr_end].hex()
            sc = getattr(d, "SC", None)
            if sc is not None:
                seq_num = (int(sc) >> 4) & 0x0FFF
                frag_num = int(sc) & 0x000F
            else:
                seq_num = getattr(d, "seq_num", None)
                frag_num = getattr(d, "frag_num", None)
            rssi = self._radiotap_field(pkt, "dBm_AntSignal")
            chan_freq = self._radiotap_field(pkt, "ChannelFrequency")
            mactime = self._radiotap_field(pkt, "mac_timestamp")
            ts = float(pkt.time)
            db = self.open_db_for_today()
            db.execute(
                "INSERT INTO bfi (ts, src_mac, ap_mac, category, subtype, "
                "mimo_ctrl, body_hex, body_len, rssi, chan_freq, mactime, "
                "nc, nr, bw_mhz, ng, codebook, feedback_type, dialog_token, "
                "avg_snr_hex, seq_num, frag_num) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ts,
                    d.addr2,
                    d.addr1,
                    int(cat),
                    int(d.subtype),
                    mimo_ctrl_bytes.hex(),
                    body.hex(),
                    len(body),
                    int(rssi) if rssi is not None else None,
                    int(chan_freq) if chan_freq is not None else None,
                    int(mactime) if mactime is not None else None,
                    parsed.get("nc"),
                    parsed.get("nr"),
                    parsed.get("bw_mhz"),
                    parsed.get("ng"),
                    parsed.get("codebook"),
                    parsed.get("feedback_type"),
                    parsed.get("dialog_token"),
                    avg_snr_hex,
                    int(seq_num) if seq_num is not None else None,
                    int(frag_num) if frag_num is not None else None,
                ),
            )
            with self._lock:
                self.bfi_today += 1
                self.last_bfi_ts = now
                self.uncommitted += 1
                if self.uncommitted >= self.commit_every:
                    db.commit()
                    self.uncommitted = 0
        except Exception as e:
            print(f"[error] {type(e).__name__}: {e}", file=sys.stderr, flush=True)

    def heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_s):
            now = time.time()
            with self._lock:
                quiet = now - self.last_bfi_ts
                any_quiet = now - self.last_frame_ts
                bfi = self.bfi_today
                frames = self.frames_seen
                if self.conn is not None and self.uncommitted > 0:
                    try:
                        self.conn.commit()
                        self.uncommitted = 0
                    except Exception as e:
                        print(f"[error] heartbeat commit: {e}",
                              file=sys.stderr, flush=True)
            level = "warn" if any_quiet > self.stall_warn_s else "info"
            print(
                f"[heartbeat:{level}] frames={frames} bfi_today={bfi} "
                f"any_quiet_s={any_quiet:.0f} bfi_quiet_s={quiet:.0f}",
                flush=True,
            )
            self.open_db_for_today()
            if any_quiet > self.stall_exit_s and frames > 0:
                # Radio detuned or driver dropped frames. Exit so systemd
                # Restart=always plus ExecStartPre=setup_monitor.sh rebuilds
                # mon0 on the right channel. Skip on cold start (frames==0)
                # so a slow first frame does not loop us.
                msg = (f"[stall-exit] no frames for {any_quiet:.0f}s > "
                       f"{self.stall_exit_s}s threshold; exiting for systemd restart")
                print(msg, file=sys.stderr, flush=True)
                _sd_notify(
                    f"STOPPING=1\nSTATUS=stall-exit any_quiet_s={any_quiet:.0f}"
                )
                os.kill(os.getpid(), signal.SIGTERM)
                return
            _sd_notify(
                f"WATCHDOG=1\nSTATUS=frames={frames} bfi_today={bfi} "
                f"any_quiet_s={any_quiet:.0f}"
            )

    def shutdown(self, *_a) -> None:
        print("[stop] received signal, flushing", flush=True)
        _sd_notify("STOPPING=1")
        self._stop.set()
        with self._lock:
            if self.conn is not None:
                try:
                    self.conn.commit()
                    self.conn.close()
                except Exception as e:
                    print(f"[error] shutdown: {e}", file=sys.stderr, flush=True)
        sys.exit(0)

    def run(self) -> None:
        print(f"[start] iface={self.iface} db_dir={self.db_dir}", flush=True)
        signal.signal(signal.SIGTERM, self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown)
        self.open_db_for_today()
        hb = threading.Thread(target=self.heartbeat_loop, daemon=True)
        hb.start()
        _sd_notify("READY=1\nSTATUS=sniffing")
        sniff(iface=self.iface, prn=self.callback, store=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iface", default="mon0")
    p.add_argument("--db-dir", default="/var/lib/bfi-capture")
    p.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    p.add_argument("--heartbeat-s", type=int, default=DEFAULT_HEARTBEAT_S)
    p.add_argument("--stall-warn-s", type=int, default=DEFAULT_STALL_WARN_S)
    p.add_argument("--stall-exit-s", type=int, default=DEFAULT_STALL_EXIT_S)
    p.add_argument("--commit-every", type=int, default=DEFAULT_COMMIT_EVERY)
    args = p.parse_args()
    Capture(
        Path(args.db_dir), args.iface,
        log_every=args.log_every,
        heartbeat_s=args.heartbeat_s,
        stall_warn_s=args.stall_warn_s,
        stall_exit_s=args.stall_exit_s,
        commit_every=args.commit_every,
    ).run()


if __name__ == "__main__":
    main()
