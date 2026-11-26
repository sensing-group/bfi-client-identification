#!/usr/bin/env python3
"""C1: coverage of the ESP32-C5 mesh as a BFI receiver (dataset B), beside
the format mix of the paper's AX210 capture (dataset A, r cache).

Reads the per-source aggregates written by c5_common.py (one streaming pass
over each (B) file) and the r cache of (A). Never touches the JSONL itself.

Outputs (figures/):
  c5_coverage.csv          one row per (B source, report format, body length)
                           and one per (A, device, format) for comparison
  c5_coverage_nodes.csv    per receiving node: receptions, active hours, rate
  c5_coverage_sources.csv  per source MAC (labelled six + every other MAC
                           with >= MIN_SRC_ROWS unique reports)
  c5_coverage_summary.json everything else (dedup check, labelled devices
                           present, per-device format A vs B, truncation)
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5_common as C  # noqa: E402
import r_common as R  # noqa: E402
import xres_common as X  # noqa: E402

MIN_SRC_ROWS = 64          # unique reports for an unlabelled MAC to get a row
OUT = dict(csv=C.FIG_DIR / f"{C.OUT_PREFIX}coverage.csv",
           nodes=C.FIG_DIR / f"{C.OUT_PREFIX}coverage_nodes.csv",
           sources=C.FIG_DIR / f"{C.OUT_PREFIX}coverage_sources.csv",
           summary=C.FIG_DIR / f"{C.OUT_PREFIX}coverage_summary.json")
CSV_COLS = ["dataset", "source", "span", "device", "fmt", "bw_mhz", "nr",
            "nc", "ng", "feedback_type", "codebook", "bphi", "bpsi",
            "bb_format", "subtype", "category", "action", "body_len",
            "full_body_len", "angle_field_complete", "at_firmware_cap",
            "tones_present", "tones_full", "bins_kept", "receptions",
            "unique_reports", "n_src_macs", "n_nodes", "length_ok",
            "labelled_srcs"]


def a_format_mix(devs: list[str]) -> list[dict]:
    """Frames and windows of (A) per (device, geometry) from the r cache."""
    frames: Counter = Counter()
    for p in X.day_paths():
        d = R.load_day_cache(p.stem)
        if "dev" not in d or not len(d["dev"]):
            continue
        for di, gi in zip(d["dev"].tolist(), d["gid"].tolist()):
            frames[(di, d["geoms"][gi])] += 1
    z = np.load(R.windows_path(C.SESSION_GAP_S), allow_pickle=False)
    gt = [tuple(json.loads(s)) for s in z["geom_table"]]
    wins: Counter = Counter()
    for di, gi in zip(z["dev"].tolist(), z["geom"].tolist()):
        wins[(di, gt[gi])] += 1
    rows = []
    for (di, key) in sorted(frames):
        g = dict(zip(C.GEOM_FIELDS, key))
        g["mimo_len"] = (X.AD.HE_MIMO_CTRL_LEN if g["fmt"] == "HE"
                         else X.AD.VHT_MIMO_CTRL_LEN)
        fl = C.full_body_len(g)
        rows.append(dict(dataset="A", source="ax210", span="2026-05-08..20",
                         device=R.short(devs[di]), **{k: g[k] for k in
                                                       C.GEOM_FIELDS},
                         full_body_len=fl, angle_field_complete=True,
                         at_firmware_cap=False,
                         tones_present=X.AD.expected_subcarriers(g),
                         tones_full=X.AD.expected_subcarriers(g),
                         bins_kept=C.N_SC_BINS, receptions=frames[(di, key)],
                         unique_reports=frames[(di, key)], n_src_macs=1,
                         n_nodes=1, windows=wins[(di, key)]))
    return rows


def main() -> int:
    X.set_vm_limit()
    devs = C.labelled_macs()
    short = {m: R.short(m) for m in devs}
    rows, node_rows, src_rows = [], [], []
    summary: dict = dict(
        b_schema=C.__doc__.split("Dataset (B) record schema")[1]
        .split("The same report")[0].strip(),
        dedup_window_s=C.DEDUP_WINDOW_S, firmware_body_cap=C.BFI_MAX_BODY,
        ndpa_note=("(B) holds beamforming REPORTS only (Action/Action No-Ack "
                   "frames); the NDPA/NDP requests are not captured, so a "
                   "request-hearing rate cannot be computed from (B)."),
        sources={}, per_device={}, a_format_mix=None)
    per_device: dict = {m: dict(label=short[m], mac=m, in_B={}) for m in devs}
    for name in C.SOURCES:
        if not C.coverage_path(name).exists():
            summary["sources"][name] = dict(missing=True)
            continue
        cov = C.load_coverage(name)
        span = f"{cov['ts_min_iso'][:10]}..{cov['ts_max_iso'][:10]}"
        gt = {g["geom_id"]: g for g in cov["geom_table"]}
        present = [s for s in cov["sources"] if s in devs]
        summary["sources"][name] = dict(
            path=cov["path"], lines=cov["lines"], json_errors=cov["json_errors"],
            span=span, span_h=cov["span_h"], hours_with_data=cov["n_hours_with_data"],
            receptions=sum(r["receptions"] for r in cov["formats"]),
            unique_reports=sum(r["unique_reports"] for r in cov["formats"]),
            n_nodes=len(cov["nodes"]), n_src_macs=len(cov["sources"]),
            n_src_macs_global=sum(1 for s in cov["sources"]
                                  if not X.is_random(s)),
            labelled_devices_present=sorted(short[s] for s in present),
            labelled_devices_absent=sorted(short[s] for s in devs
                                           if s not in present),
            dup_gap_hist_s=cov["dup_gap_hist"], nnodes_hist=cov["nnodes_hist"],
            bfi_reports=sum(r["receptions"] for r in cov["formats"]
                            if r["geom_id"] is not None),
            non_bfi_records=sum(r["receptions"] for r in cov["formats"]
                                if r["geom_id"] is None),
            angle_complete_receptions=sum(
                r["receptions"] for r in cov["formats"]
                if r.get("angle_field_complete") is True),
            at_cap_receptions=sum(r["receptions"] for r in cov["formats"]
                                  if r.get("at_firmware_cap")),
            undecodable_first_column=cov["undecodable_first_column"],
            pass_seconds=cov["seconds"], pass_peak_rss_mb=cov["peak_rss_mb"])
        for r in cov["formats"]:
            g = gt.get(r["geom_id"])
            row = dict(dataset="B", source=name, span=span, device="",
                       bb_format=r["bb_format"], subtype=r["subtype"],
                       category=r["category"], action=r["action"],
                       body_len=r["body_len"], receptions=r["receptions"],
                       unique_reports=r["unique_reports"],
                       n_src_macs=r["n_src_macs"], n_nodes=r["n_nodes"],
                       length_ok=r["length_ok"],
                       labelled_srcs=";".join(short[s] for s in
                                              r["labelled_srcs"]))
            if g is not None:
                row.update({k: g[k] for k in C.GEOM_FIELDS})
                row.update(full_body_len=r["full_body_len"],
                           angle_field_complete=r["angle_field_complete"],
                           at_firmware_cap=r["at_firmware_cap"],
                           tones_present=r["tones_present"],
                           tones_full=r["tones_full"])
                if r["tones_full"]:
                    row["bins_kept"] = int(C.covered_bins(
                        r["tones_full"], r["tones_present"]).sum())
            else:
                row["fmt"] = r.get("non_bfi_name") or "non-BFI"
            rows.append(row)
        for n, v in sorted(cov["nodes"].items()):
            node_rows.append(dict(source=name, node_mac=n, **v))
        for s, v in sorted(cov["sources"].items(),
                           key=lambda kv: -kv[1]["unique_reports"]):
            if s in devs or v["unique_reports"] >= MIN_SRC_ROWS:
                src_rows.append(dict(source=name, src_mac=s,
                                     label=short.get(s, ""), **v))
        # per labelled device: formats seen in B, truncation, nodes hearing
        for s in present:
            fm = [r for r in cov["formats"] if s in r["labelled_srcs"]]
            per_device[s]["in_B"][name] = dict(
                receptions=cov["sources"][s]["receptions"],
                unique_reports=cov["sources"][s]["unique_reports"],
                first=cov["sources"][s]["first"], last=cov["sources"][s]["last"],
                hours_active=cov["sources"][s]["hours_active"],
                unique_per_active_hour=cov["sources"][s]["unique_per_active_hour"],
                n_nodes_hearing=cov["sources"][s]["n_nodes_hearing"],
                formats=[dict({k: gt[r["geom_id"]][k] for k in C.GEOM_FIELDS}
                              if r["geom_id"] is not None else {},
                              body_len=r["body_len"], receptions=r["receptions"],
                              full_body_len=r.get("full_body_len"),
                              tones_present=r.get("tones_present"),
                              tones_full=r.get("tones_full"),
                              angle_field_complete=r.get("angle_field_complete"))
                         for r in fm],
                frame_chunks=cov["frame_chunks"].get(s, 0))
    a_rows = a_format_mix(devs)
    rows += a_rows
    summary["a_format_mix"] = a_rows
    for m in devs:
        per_device[m]["in_A"] = [dict(fmt=r["fmt"], nc=r["nc"], nr=r["nr"],
                                      bphi=r["bphi"], bpsi=r["bpsi"],
                                      bw_mhz=r["bw_mhz"], ng=r["ng"],
                                      feedback_type=r["feedback_type"],
                                      codebook=r["codebook"],
                                      full_body_len=r["full_body_len"],
                                      frames=r["receptions"],
                                      windows=r["windows"])
                                 for r in a_rows if r["device"] == short[m]]
    summary["per_device"] = {short[m]: v for m, v in per_device.items()}
    # 07-24 decode summary of the esp32 pipeline, for cross-reference
    if C.B_COVERAGE_SUMMARY.exists():
        s = json.loads(C.B_COVERAGE_SUMMARY.read_text())
        summary["esp32_decode_summary_20260724"] = dict(
            path=str(C.B_COVERAGE_SUMMARY), total_records=s["total_records"],
            total_decoded=s["total_decoded"],
            n_distinct_formats=s["n_distinct_formats"])
    # write
    cols = CSV_COLS + ["windows"]
    with open(OUT["csv"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    for key, rr in (("nodes", node_rows), ("sources", src_rows)):
        if rr:
            with open(OUT[key], "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rr[0].keys()))
                w.writeheader()
                w.writerows(rr)
    X.write_json(OUT["summary"], summary)
    print(f"[out] {OUT['csv']} ({len(rows)} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
