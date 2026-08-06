#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a synthetic EGDA-hydrolysis time series with KNOWN rate constants
==========================================================================

Writes Bruker-style ``ascii-spec_<minute>.txt`` files into ``test_data/`` that
look like a real run: four overlapping backbone signals, three acetyl singlets,
a temperature-shifted water peak that swamps the backbone region, a curved
instrumental background, per-spectrum referencing jitter and Gaussian noise.

Why bother: the whole point of this pipeline is to pull three overlapping mole
fractions out of an 80 MHz spectrum, and on real data you have no way to check
whether it got them right. Here you DO -- the answer is set by CONFIG below, so
you can run the pipeline end to end and compare.

    python make_test_data.py                 # writes test_data/
    (point analyze_real_nmr.py / main.py "input" at test_data, then run them)
    python make_test_data.py --check         # ...and print measured vs true

The defaults use k1/k2 = 2.5, i.e. slightly MORE than the statistical 2.0, so
the second ester group is a little harder to hydrolyse than a naive count of
ester groups would suggest -- a case where a lumped single-k fit would be
misleading and the consecutive fit in kinetics_estimation.py earns its keep.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import sim_nmr as s                       # noqa: E402  (path set above first)


# ===========================================================================
#  CONFIG  -- the ground truth
# ===========================================================================
CONFIG = {
    "output_folder": r"test_data",
    "sample_id":     "EGDA_synthetic_49C_0.4M",

    # ---- ground-truth kinetics (min^-1) ----------------------------------
    "k1": 0.020,        # EGDA -> EGMA
    "k2": 0.008,        # EGMA -> EG     (k1/k2 = 2.5; statistical would be 2.0)
    "minutes": list(range(0, 125, 5)),

    # ---- conditions ------------------------------------------------------
    # 49 C puts water at ~4.51 ppm: on the shoulder of the backbone signals but
    # not on top of them. Try 70 C to see the collision the docstrings warn
    # about, and watch the backbone handle degrade while acetyl stays solid.
    "temperature_C": 49.0,
    "initial_water": 100.0,

    # ---- instrument ------------------------------------------------------
    "n_points":   32768,
    "ppm_window": (-1.0, 13.0),
    "intensity_scale": 8.0e7,    # Bruker-sized intensities
    "noise_fraction":  0.004,    # sigma, as a fraction of the intensity scale
    "referencing_jitter_ppm": 0.006,   # per-spectrum shimming/referencing drift
    "seed": 7,
}


def true_fractions(t, k1, k2):
    """Analytical A->B->C solution from pure EGDA."""
    t = np.asarray(t, float)
    xD = np.exp(-k1 * t)
    if abs(k2 - k1) < 1e-12:
        xM = k1 * t * np.exp(-k1 * t)
    else:
        xM = k1 / (k2 - k1) * (np.exp(-k1 * t) - np.exp(-k2 * t))
    return xD, xM, np.clip(1.0 - xD - xM, 0.0, 1.0)


def write_test_data(cfg=CONFIG):
    out = cfg["output_folder"]
    if not os.path.isabs(out):
        out = os.path.join(HERE, out)
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])

    sim = dict(s.CONFIG)
    sim["temperature_C"] = cfg["temperature_C"]
    sim["initial_water"] = cfg["initial_water"]

    lo, hi = cfg["ppm_window"]
    n = int(cfg["n_points"])
    ppm = np.linspace(lo, hi, n)
    hz = ppm * s.SPEC_MHZ
    scale = float(cfg["intensity_scale"])

    rows = []
    for minute in cfg["minutes"]:
        xD, xM, xG = (float(v) for v in true_fractions(minute, cfg["k1"], cfg["k2"]))
        n_acid = xM + 2.0 * xG
        conc = {"D": xD, "M": xM, "G": xG, "aa": n_acid,
                "w": cfg["initial_water"] - n_acid}
        y = s.build_spectrum(ppm, conc, cfg=sim)

        # the whole spectrum wanders a little between acquisitions
        drift = float(rng.normal(0.0, cfg["referencing_jitter_ppm"]))
        y = np.interp(ppm, ppm + drift, y)

        # broad instrumental background + slope, then detector noise
        y = y + 0.35 * np.exp(-((ppm - 6.0) / 7.0) ** 2) + 0.02 * ppm
        y = y * scale
        y = y + rng.normal(0.0, cfg["noise_fraction"] * scale, y.size)

        path = os.path.join(out, f"ascii-spec_{minute}.txt")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"Sample id: {cfg['sample_id']}\n")
            for i in range(n - 1, -1, -1):       # Bruker writes high ppm first
                fh.write(f"{n - i}, {y[i]:.1f}, {hz[i]:.4f}, {ppm[i]:.6f}\n")
        rows.append((minute, xD, xM, xG, 0.5 * (xM + 2.0 * xG)))

    print(f"wrote {len(rows)} spectra to {out}")
    print(f"ground truth: k1={cfg['k1']} k2={cfg['k2']} min^-1  "
          f"(k1/k2 = {cfg['k1']/cfg['k2']:g}), water at "
          f"{s.water_shift(cfg['temperature_C']):.2f} ppm")
    print(f"{'min':>5}{'x_EGDA':>9}{'x_EGMA':>9}{'x_EG':>9}{'X_ester':>9}")
    for minute, xD, xM, xG, X in rows[::4]:
        print(f"{minute:>5}{xD:>9.4f}{xM:>9.4f}{xG:>9.4f}{X:>9.4f}")
    return out, rows


def check(cfg=CONFIG):
    """Run the backbone analysis on the test data and score it against truth."""
    import analyze_real_nmr as arn

    folder = cfg["output_folder"]
    if not os.path.isabs(folder):
        folder = os.path.join(HERE, folder)
    if not os.path.isdir(folder):
        raise SystemExit(f"{folder} does not exist -- run without --check first.")

    acfg = dict(arn.CONFIG)
    acfg["input"] = folder
    files = arn.resolve_inputs(acfg, HERE)
    anchors = arn.resolve_backbone_anchors(files, acfg, verbose=False)

    print(f"\n{'min':>5}{'x_EGDA meas/true':>20}{'x_EGMA meas/true':>20}"
          f"{'x_EG meas/true':>20}{'max err':>10}")
    worst = 0.0
    for path in files:
        minute = arn.minute_of(path)
        r = arn.analyze_one(path, acfg, anchors=anchors)
        m = r["speciation"]
        tD, tM, tG = (float(v) for v in true_fractions(minute, cfg["k1"], cfg["k2"]))
        err = max(abs(m["x_egda"] - tD), abs(m["x_egma"] - tM), abs(m["x_eg"] - tG))
        worst = max(worst, err)
        print(f"{minute:>5}{m['x_egda']:>10.3f}/{tD:<9.3f}"
              f"{m['x_egma']:>10.3f}/{tM:<9.3f}{m['x_eg']:>10.3f}/{tG:<9.3f}"
              f"{err:>10.3f}")
    print(f"\nworst absolute error in any mole fraction: {worst*100:.1f} "
          f"percentage points")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="also analyse the data and compare with the truth")
    args = parser.parse_args(argv)
    write_test_data()
    if args.check:
        check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
