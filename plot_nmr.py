#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive 1H NMR spectrum viewer for Bruker ASCII exports
===========================================================

Draws a spectrum the conventional way (ppm increasing to the LEFT) and lets you
zoom/pan into any multiplet.

  * INPUT is a single file  -> opens an interactive window (zoom/pan/hotkeys)
  * INPUT is a folder        -> batch-saves a PNG of every spectrum (no window)

Expected file format (one comma-separated row per point)::

    Sample id: ethyl_acetate          <- optional header line(s), auto-skipped
    1, 9618, 978.5105, 12.205745       <- index, intensity, frequency[Hz], ppm

>>> Just press "Run" in your IDE. Everything is controlled by CONFIG below. <<<
"""

from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt

# ===========================================================================
#  CONFIG  -- edit these; everything below is driven by this dictionary
# ===========================================================================
CONFIG = {
    # ---- INPUT: a single ascii-spec file OR a folder (batch) -------------
    "input":     r"NMR-EGDA_data",
    "file_glob": "ascii-spec*.txt",      # used when "input" is a folder

    # ---- VIEW ------------------------------------------------------------
    # For the EGDA hydrolysis the two regions worth looking at are
    #   (4.60, 3.45)  the backbone O-CH2 signals  (EGDA / EGMA / EG)
    #   (2.45, 1.85)  the acetyl CH3 singlets     (ester vs acetic acid)
    # Set one of those as "xlim" to check where your peaks actually sit before
    # you trust the anchor shifts in analyze_real_nmr.py / main.py.
    "xlim": None,            # initial ppm window, e.g. (4.60, 3.45); None = full
    "interactive": True,     # open a zoom/pan window (single-file input only)

    # ---- INTEGRAL OVERLAY (optional) -------------------------------------
    "show_integrals": False,
    "integrals_file": r"NMR-EGDA_data\integrals.txt",

    # ---- SAVING ----------------------------------------------------------
    "save_png":  True,       # also write a PNG
    "output":    r"results\plots",  # folder for the PNGs
    "dpi":       150,
}

C_LINE = "#1f4e79"


# ===========================================================================
#  DATA LOADING
# ===========================================================================
def load_spectrum(path):
    """Parse a Bruker ASCII spectrum -> (ppm, intensity, hz, sample_id)."""
    ppm, intensity, hz = [], [], []
    sample_id = None
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.lower().startswith("sample id"):
                sample_id = s.split(":", 1)[1].strip() if ":" in s else None
                continue
            parts = [p.strip() for p in s.split(",")]
            if len(parts) < 4:
                continue
            try:
                intensity.append(float(parts[1]))
                hz.append(float(parts[2]))
                ppm.append(float(parts[3]))
            except ValueError:
                continue
    if not ppm:
        sys.exit(f"No numeric spectral data found in {path!r}.")
    ppm = np.asarray(ppm)
    intensity = np.asarray(intensity)
    hz = np.asarray(hz)
    order = np.argsort(ppm)
    return ppm[order], intensity[order], hz[order], sample_id


def load_integrals(path):
    """Parse integrals.txt -> list of (left_ppm, right_ppm, value)."""
    regions = []
    if not os.path.exists(path):
        return regions
    row = re.compile(r"^\s*\d+\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$")
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = row.match(line)
            if m:
                a, b, val = (float(v) for v in m.groups())
                regions.append((max(a, b), min(a, b), val))
    return regions


HERE = os.path.dirname(os.path.abspath(__file__))


def _abs(path):
    """Relative CONFIG paths are read next to this script, not next to the cwd."""
    return path if os.path.isabs(path) else os.path.join(HERE, path)


def resolve_inputs(cfg):
    inp = _abs(cfg["input"])
    if os.path.isdir(inp):
        files = sorted(glob.glob(os.path.join(inp, cfg["file_glob"])))
        if not files:
            raise SystemExit(f"No files matching {cfg['file_glob']!r} in {inp!r}.")
        return files, True
    if os.path.isfile(inp):
        return [inp], False
    raise SystemExit(f"Input path does not exist: {inp!r}")


# ===========================================================================
#  PLOTTING
# ===========================================================================
def make_plot(ppm, intensity, sample_id, cfg, integrals=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    try:
        fig.canvas.manager.set_window_title(f"NMR spectrum - {sample_id or ''}")
    except Exception:
        pass

    ax.plot(ppm, intensity, lw=0.8, color=C_LINE)
    ax.axhline(0, color="0.7", lw=0.6, zorder=0)

    if integrals:
        ymax = intensity.max()
        for left, right, val in integrals:
            ax.axvspan(right, left, color="#e76f51", alpha=0.12, zorder=0)
            ax.text(0.5 * (left + right), ymax * 0.92, f"{val:.2f}", ha="center",
                    va="top", fontsize=9, color="#9c3b25")

    title = "1H NMR spectrum"
    if sample_id:
        title += f"  -  {sample_id}"
    ax.set_title(title)
    ax.set_xlabel(r"Chemical shift  $\delta$  (ppm)")
    ax.set_ylabel("Intensity (a.u.)")

    if cfg["xlim"] is not None:
        ax.set_xlim(max(cfg["xlim"]), min(cfg["xlim"]))
    else:
        ax.set_xlim(ppm.max(), ppm.min())          # NMR convention

    ax.grid(True, which="both", axis="x", color="0.9", lw=0.5)
    span = intensity.max() - intensity.min()
    ax.set_ylim(intensity.min() - 0.05 * span, intensity.max() + 0.10 * span)
    fig.tight_layout()
    _add_key_bindings(fig, ax, ppm, intensity)
    return fig, ax


def _add_key_bindings(fig, ax, ppm, intensity):
    """Hotkeys on top of the toolbar: 'r' reset, 'a' autoscale y to the view."""
    full_x = (ppm.max(), ppm.min())
    span = intensity.max() - intensity.min()
    full_y = (intensity.min() - 0.05 * span, intensity.max() + 0.10 * span)

    def on_key(event):
        if event.key == "r":
            ax.set_xlim(*full_x)
            ax.set_ylim(*full_y)
        elif event.key == "a":
            lo, hi = sorted(ax.get_xlim())
            mask = (ppm >= lo) & (ppm <= hi)
            if mask.any():
                yv = intensity[mask]
                m = yv.max() - yv.min()
                ax.set_ylim(yv.min() - 0.05 * m, yv.max() + 0.10 * m)
        else:
            return
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)


# ===========================================================================
#  DRIVER
# ===========================================================================
def main(cfg=CONFIG):
    files, is_batch = resolve_inputs(cfg)
    out_dir = _abs(cfg["output"])
    integrals = (load_integrals(_abs(cfg["integrals_file"]))
                 if cfg["show_integrals"] else None)

    if is_batch:
        os.makedirs(out_dir, exist_ok=True)
        print(f"Batch plotting {len(files)} spectra -> {out_dir}")
        for path in files:
            ppm, inten, hz, sid = load_spectrum(path)
            fig, _ = make_plot(ppm, inten, sid, cfg, integrals)
            stem = os.path.splitext(os.path.basename(path))[0]
            out = os.path.join(out_dir, f"{stem}.png")
            fig.savefig(out, dpi=cfg["dpi"])
            plt.close(fig)
            print(f"  saved {out}")
        return

    # single file
    path = files[0]
    ppm, inten, hz, sid = load_spectrum(path)
    print(f"Loaded {len(ppm)} points  |  ppm {ppm.min():.3f}..{ppm.max():.3f}  "
          f"|  sample: {sid or 'n/a'}")
    fig, _ = make_plot(ppm, inten, sid, cfg, integrals)

    if cfg["save_png"]:
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(out_dir, f"{stem}.png")
        fig.savefig(out, dpi=cfg["dpi"])
        print(f"saved {out}")

    if cfg["interactive"]:
        print("Controls: toolbar magnifier = box-zoom, pan tool = drag.  "
              "Hotkeys: 'r' reset view, 'a' autoscale y to visible region.")
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
