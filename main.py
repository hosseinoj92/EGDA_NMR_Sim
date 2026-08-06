#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversion of the EGDA hydrolysis from EITHER the backbone O-CH2 signals OR the
acetyl CH3 singlets  (a user-selectable extension of analyze_real_nmr.py)
=============================================================================

    EGDA + H2O  ->  EGMA + AcOH        (step 1, cat. Amberlyst-15 / Lewatit)
    EGMA + H2O  ->  EG   + AcOH        (step 2)

This is a CONSECUTIVE hydrolysis, so there are two different, both useful,
definitions of "conversion":

    X_EGDA   = fraction of the DIESTER that has reacted at all
             = x(EGMA) + x(EG)
    X_ester  = fraction of ALL ester groups hydrolysed  (EGDA has 2, EGMA has 1)
             = [x(EGMA) + 2 x(EG)] / 2
             = n(AcOH) / (2 n0)

X_ester is the one written to the `conversion_X` column, because it is the
quantity BOTH handles can measure and the one that maps onto acetic acid
released. X_EGDA and the full speciation come from the backbone handle only.

TWO HANDLES ON THE SAME REACTION
--------------------------------
  * "backbone" -> the O-CH2 region (~3.6-4.4 ppm). Every glycol-derived molecule
                  keeps FOUR backbone protons, so the four overlapping signals
                  (EGDA singlet, EGMA's two triplets, EG singlet) give the FULL
                  speciation x(EGDA) / x(EGMA) / x(EG) directly, with the total
                  area conserved as a free internal standard. Deconvolved by
                  analyze_real_nmr.measure_backbone -- see that file for the
                  physics that makes the overlap tractable.

  * "acetyl"   -> the CH3 singlets (~2.1 ppm). An acetyl group is either still
                  ESTERIFIED (EGDA 2.14 / EGMA 2.13 ppm -- only ~1 Hz apart at
                  80 MHz, so they are pooled into one line, which is fine
                  because they are chemically the same thing) or FREE acetic
                  acid (~2.08 ppm). Both lines are 3 H per acetyl group, so

                      X_ester = A(AcOH) / [ A(AcOH) + A(bound acetyl) ]

                  is exact and gain-independent. This handle gives no EGMA/EG
                  split, but it sits FAR FROM WATER and is therefore the robust
                  cross-check -- especially at high temperature, where the water
                  resonance (~5.051 - 0.0111*T ppm) drifts onto the backbone
                  signals. At 70 C water is at ~4.28 ppm, right on the EGDA
                  singlet; the acetyl region does not care.

WHAT YOU GET (all tagged by the mode you chose, in a "<mode>_conversion/" folder
so a backbone run and an acetyl run never overwrite each other):

  <stem>_<mode>_fit.png       the deconvolution picture for THAT spectrum
  <stem>_<mode>_groups.csv    the measured signals (delta, J, FWHM, area, R^2)
  <stem>_<mode>_species.csv   per-compound amounts at that spectrum
  conversion_vs_time_<mode>.csv   the master kinetics table across the batch
  conversion_vs_time_<mode>.png   speciation / conversion / amounts vs time

Feed conversion_vs_time_<mode>.csv to kinetics_estimation.py for k1 and k2.

INPUT can be a single ascii-spec file OR a whole folder (batch).

>>> Just press "Run" in your IDE. Everything is controlled by CONFIG below. <<<
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")                 # headless: we only save PNGs
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- reuse analyze_real_nmr.py (the backbone pipeline + shared helpers) -----
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import analyze_real_nmr as arn        # noqa: E402  (path set above first)


# ===========================================================================
#  CONFIG  -- edit these; everything below is driven by this dictionary
# ===========================================================================
CONFIG = {
    # ---- CHOOSE THE REACTION COORDINATE ----------------------------------
    #   "backbone" -> the O-CH2 region: FULL speciation (EGDA/EGMA/EG)
    #   "acetyl"   -> the CH3 singlets: overall ester conversion, water-immune
    "conversion_mode": "backbone",

    # ---- INPUT: a single ascii-spec file OR a folder (batch) -------------
    "input":     r"NMR-EGDA_data",
    "file_glob": "ascii-spec*.txt",

    # ---- OUTPUT ----------------------------------------------------------
    "output":         r"results",
    "save_fit_image": True,     # per-spectrum deconvolution picture (PNG)
    "save_group_csv": True,     # per-spectrum measured-signals table
    "save_species_csv": True,   # per-spectrum per-compound amounts table
    "save_summary":   True,     # combined kinetics table (+ vs-time plot)
    "dpi":            150,

    # ---- EXPERIMENT (turns the dimensionless conversion into amounts) -----
    "v_solution_L":   0.200,    # L total solution (water + EGDA)
    "c_diester_M":    0.400,    # mol/L initial EGDA
    "catalyst_g":     2.0,      # g catalyst (metadata; not in the balance)
    "temperature_C":  70.0,     # C (also predicts where water sits)

    # ---- BACKBONE DECONVOLUTION (only when conversion_mode == "backbone")
    # The fit itself -- region, ANCHOR SHIFTS, line widths, water handling,
    # baseline -- lives in analyze_real_nmr.BACKBONE_DEFAULTS, so there is ONE
    # place to recalibrate it and the two scripts can never disagree.
    # !! RECALIBRATE THE CENTRES THERE, NOT HERE !!  Anything you add below
    # overrides that copy for this script only.
    "backbone": {
        **arn.BACKBONE_DEFAULTS,
        # refine the four centres from the batch before the real pass
        "auto_anchor": True,
    },

    # ---- ACETYL SINGLET FIT (only when conversion_mode == "acetyl") -------
    "acetyl": {
        # region containing all the acetyl lines with signal-free edges.
        # Nothing else lives here for this reaction.
        "window": (1.85, 2.42),

        # anchors (ppm). Only the SEPARATIONS have to be right -- a common
        # offset is absorbed by the free centre.
        "egda_center": 2.140,       # EGDA acetyl  (6 H = 2 acetyl groups)
        "egma_center": 2.125,       # EGMA acetyl  (3 H = 1 acetyl group)
        "acid_center": 2.080,       # acetic acid  (3 H)
        "center_drift": 0.060,      # how far the whole set may move
        "sep_bounds":   (0.025, 0.100),   # allowed EGDA-acetyl / AcOH separation

        "init_fwhm_hz": 1.8,
        "min_fwhm_hz":  0.4,
        "max_fwhm_hz":  8.0,
        "maxfev":  40000,
        # a singlet below snr_min x noise is treated as ABSENT (area = 0)
        # -> handles the 0 % spectra (no acetic acid yet).
        "snr_min": 8.0,

        "baseline_method": "arpls",
        "als_lambda": 1e8,
        "als_niter":  50,
        "baseline_deg": 1, "baseline_iter": 12, "baseline_reject": 2.0,
    },

    # noise is estimated from this signal-free window (shared by both modes)
    "noise_window": (5.5, 9.0),
}

MW, SPECIES_INFO = arn.MW, arn.SPECIES_INFO
C_EGDA, C_EGMA, C_EG, C_ACID = arn.C_EGDA, arn.C_EGMA, arn.C_EG, arn.C_ACID
C_DATA, C_FIT, C_WATER = arn.C_DATA, arn.C_FIT, arn.C_WATER


def _sub(cfg, key):
    """The mode's own settings, merged with the batch-wide ones it needs."""
    out = dict(cfg[key])
    out["noise_window"] = cfg["noise_window"]
    out["temperature_C"] = cfg["temperature_C"]
    return out


# ===========================================================================
#  ACETYL  --  two singlets: esterified acetyl vs free acetic acid
# ===========================================================================
def _acetyl_model(d_egma):
    """Linear baseline + EGDA acetyl + EGMA acetyl + acetic acid.

    Constraints that make this heavily-overlapping set well-posed:

      * acetic acid is ALWAYS up-field of an esterified acetyl (cA = cD - s with
        s > 0), so the lines can neither swap nor collapse onto each other;
      * all three are CH3 singlets on chemically similar acetyl carbons, so they
        SHARE the width `w` and shape `eta` and differ only in POSITION and
        AMPLITUDE -- which is exactly what changes with conversion. Sharing the
        width is what stops the fit from splitting the single line of a 0 %
        spectrum into phantoms;
      * the EGDA and EGMA acetyls are pinned a fixed `d_egma` apart. At 80 MHz
        that gap is only ~1 Hz, so their two amplitudes are NOT individually
        meaningful -- but their SUM is, and the sum is all the conversion needs.
        Modelling them as two lines rather than one pooled line matters: a
        single shared-width line cannot be as wide as the real 1 Hz-split pair,
        so the fit narrows it and hands the leftover intensity to acetic acid,
        which biases the conversion high by a few percent."""
    def f(x, a, b, cD, s, w, eta, hD, hM, hA):
        return (a * x + b
                + arn.pseudo_voigt(x, cD, w, hD, eta)
                + arn.pseudo_voigt(x, cD - d_egma, w, hM, eta)
                + arn.pseudo_voigt(x, cD - s, w, hA, eta))
    return f


def measure_acetyl(ppm, inten, cfg, sf, noise):
    """Fit the two acetyl singlets and integrate each one."""
    ca = cfg["acetyl"]
    lo, hi = ca["window"]
    xr, yr = arn.slice_region(ppm, inten, lo, hi)
    if xr.size < 30:
        raise ValueError(f"Acetyl window {lo}-{hi} ppm holds too few points")

    # smooth background first, then a residual straight line inside the fit
    sub = _sub(cfg, "acetyl")
    bl = arn.region_baseline(xr, yr, sub)
    x, y = xr, yr - bl

    yscale = float(np.max(np.abs(y))) if y.size else 1.0
    yscale = yscale if yscale > 0 else 1.0
    yn = y / yscale

    a0, b0 = arn.linear_baseline(x, yn, n_edge=max(6, len(x) // 15))
    yc = yn - (a0 * x + b0)
    ymax = float(yc.max()) if yc.size else 1.0
    ymax = ymax if ymax > 0 else 1.0

    def _val_at(c):
        if c < x.min() or c > x.max():
            return 0.05 * ymax
        return float(max(yc[np.argmin(np.abs(x - c))], 0.05 * ymax))

    cD0 = float(ca["egda_center"])
    d_egma = float(ca["egda_center"]) - float(ca["egma_center"])
    cdr = float(ca["center_drift"])
    s_lo, s_hi = ca["sep_bounds"]
    s0 = float(np.clip(cD0 - float(ca["acid_center"]), s_lo, s_hi))
    wmin = ca["min_fwhm_hz"] / 2.0 / sf
    wmax = ca["max_fwhm_hz"] / 2.0 / sf
    w0 = float(np.clip(ca["init_fwhm_hz"] / 2.0 / sf, wmin, wmax))

    #     a     b     cD        s     w     eta  hD             hM                  hA
    p0 = [a0, b0, cD0, s0, w0, 0.5, _val_at(cD0), _val_at(cD0 - d_egma),
          _val_at(cD0 - s0)]
    lb = [-10., -10., cD0 - cdr, s_lo, wmin, 0.0, 0.0, 0.0, 0.0]
    ub = [10., 10., cD0 + cdr, s_hi, wmax, 1.0, 3.0, 3.0, 3.0]
    p0 = [min(max(v, l), u) for v, l, u in zip(p0, lb, ub)]

    model = _acetyl_model(d_egma)
    ok = True
    try:
        popt, _ = curve_fit(model, x, yn, p0=p0, bounds=(lb, ub),
                            x_scale="jac", maxfev=ca["maxfev"])
        r2 = arn.r_squared(yn, model(x, *popt))
    except Exception:
        popt, r2, ok = np.asarray(p0, float), 0.0, False

    popt = np.asarray(popt, float).copy()
    popt[[0, 1, 6, 7, 8]] *= yscale        # a, b, hD, hM, hA back to real units
    a, b, cD, s, w, eta, hD, hM, hA = (float(v) for v in popt)
    cM, cA = cD - d_egma, cD - s

    xf = np.linspace(lo, hi, 6000)
    area_D = float(np.trapezoid(arn.pseudo_voigt(xf, cD, w, hD, eta), xf))
    area_M = float(np.trapezoid(arn.pseudo_voigt(xf, cM, w, hM, eta), xf))
    area_A = float(np.trapezoid(arn.pseudo_voigt(xf, cA, w, hA, eta), xf))

    # The two ester acetyls are ~1 Hz apart, so only their SUM is meaningful:
    # they are gated together on their combined height.
    gate = float(ca["snr_min"]) * noise
    pres_B, pres_A = (hD + hM) >= gate, hA >= gate
    area_B = (area_D + area_M) if pres_B else 0.0
    if not pres_A:
        area_A = 0.0
    # intensity-weighted centre of the pooled ester-acetyl line, for reporting
    cB = ((cD * hD + cM * hM) / (hD + hM)) if (hD + hM) > 0 else cD

    fwhm = 2.0 * w * sf
    mk = lambda label, species, delta, area, height, present: {
        "label": label, "species": species, "protons": 3, "delta": delta,
        "J_hz": 0.0, "fwhm_hz": fwhm, "eta": eta, "area": max(area, 0.0),
        "height": height, "r2": r2, "present": bool(present),
        "snr": height / noise if noise > 0 else float("inf")}

    return {
        "window": (lo, hi), "x": x, "y": y, "baseline_x": xr, "baseline_y": bl,
        "popt": [float(v) for v in popt], "d_egma": d_egma,
        "r2": r2, "ok": ok, "stage": "three-singlet", "noise": noise, "sf": sf,
        "bound": mk("acetyl, esterified (s)", "EGDA + EGMA", cB, area_B,
                    hD + hM, pres_B),
        "free": mk("acetyl, acetic acid (s)", "acetic acid", cA, area_A, hA, pres_A),
    }


# ===========================================================================
#  PER-SPECTRUM ANALYSIS  (dispatch on the chosen mode -> a common result)
# ===========================================================================
def analyze_spectrum(path, cfg, n0_diester, n0_water, anchors=None):
    """Analyse one spectrum in the configured mode.

    Returns a normalised `result` dict shared by both modes so the CSV writers
    and the vs-time plot don't care which handle produced the numbers."""
    mode = cfg["conversion_mode"].lower()
    ppm, inten, hz, sample_id = arn.load_spectrum(path)
    sf = float(np.polyfit(ppm, hz, 1)[0])
    noise = arn.noise_level(ppm, inten, cfg["noise_window"])

    if mode == "backbone":
        fit = arn.measure_backbone(ppm, inten, _sub(cfg, "backbone"), sf, noise,
                                   anchors=anchors)
        spec = arn.speciation_from_areas(
            fit["egda"]["area"],
            fit["egma_ester"]["area"] + fit["egma_oh"]["area"],
            fit["eg"]["area"])
        components = [fit["egda"], fit["egma_ester"], fit["egma_oh"], fit["eg"]]
        known_speciation = True
    elif mode == "acetyl":
        fit = measure_acetyl(ppm, inten, cfg, sf, noise)
        aB, aA = fit["bound"]["area"], fit["free"]["area"]
        tot = aB + aA
        X = float(min(max(aA / tot, 0.0), 1.0)) if tot > 0 else 0.0
        # the acetyl handle sees only how many ester groups are gone, not how
        # they are distributed between EGMA and EG -> speciation left unknown
        spec = {"x_egda": float("nan"), "x_egma": float("nan"),
                "x_eg": float("nan"), "X_egda": float("nan"),
                "X_ester": X, "total_area": tot}
        components = [fit["bound"], fit["free"]]
        known_speciation = False
    else:
        raise SystemExit(f"conversion_mode must be 'backbone' or 'acetyl'; "
                         f"got {mode!r}")

    amt = (arn.amounts_at(spec, n0_diester, n0_water) if known_speciation
           else acetyl_amounts(spec["X_ester"], n0_diester, n0_water))

    return {
        "file": os.path.basename(path),
        "stem": os.path.splitext(os.path.basename(path))[0],
        "sample_id": sample_id, "minute": arn.minute_of(path),
        "mode": mode, "sf_mhz": sf, "noise": noise,
        "ppm": ppm, "inten": inten, "fit": fit, "speciation": spec,
        "components": components, "known_speciation": known_speciation,
        "amounts": amt, "r2": fit["r2"],
    }


def acetyl_amounts(X_ester, n0_diester, n0_water):
    """What the acetyl handle alone can say: acetic acid released and water left.

    The EGDA/EGMA/EG split is NOT determined by this handle, so it is reported
    as NaN rather than guessed from a kinetic model."""
    n_acid = 2.0 * n0_diester * X_ester
    n_water = (n0_water - n_acid) if np.isfinite(n0_water) else float("nan")
    return {"D": float("nan"), "M": float("nan"), "G": float("nan"),
            "aa": n_acid,
            "w": max(n_water, 0.0) if np.isfinite(n_water) else float("nan")}


# ===========================================================================
#  PLOTTING
# ===========================================================================
def save_fit_image(result, cfg, out_png):
    if result["mode"] == "backbone":
        arn.save_fit_image(result, cfg["backbone"] | {"dpi": cfg["dpi"]}, out_png)
    else:
        _plot_acetyl(result, cfg, out_png)


def _plot_acetyl(result, cfg, out_png):
    """One panel: data, baseline, the deconvolved acetyl singlets + total."""
    fit = result["fit"]
    x, y = fit["x"], fit["y"]
    a, b, cD, s, w, eta, hD, hM, hA = fit["popt"]
    lo, hi = fit["window"]
    cM, cA = cD - fit["d_egma"], cD - s

    xf = np.linspace(lo, hi, 4000)
    base = a * xf + b
    # the two ester acetyls are ~1 Hz apart: drawn as the single line they are
    # in practice, because only their sum is measurable
    l_bound = (arn.pseudo_voigt(xf, cD, w, hD, eta)
               + arn.pseudo_voigt(xf, cM, w, hM, eta))
    l_free = arn.pseudo_voigt(xf, cA, w, hA, eta)
    total = base + l_bound + l_free

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.plot(x, y, color=C_DATA, lw=1.2, label="background-removed data", zorder=5)
    ax.plot(xf, base, color=arn.C_BASE, lw=0.9, ls=":", zorder=2, label="baseline")

    bound, free = fit["bound"], fit["free"]
    if bound["present"]:
        ax.fill_between(xf, base, base + l_bound, color=C_EGDA, alpha=0.18,
                        zorder=1, label="esterified acetyl (EGDA + EGMA)")
        ax.plot(xf, base + l_bound, color=C_EGDA, lw=1.0, zorder=3)
    if free["present"]:
        ax.fill_between(xf, base, base + l_free, color=C_ACID, alpha=0.20,
                        zorder=1, label="acetic acid CH$_3$")
        ax.plot(xf, base + l_free, color=C_ACID, lw=1.0, zorder=3)
    ax.plot(xf, total, color=C_FIT, lw=1.5, ls="--", zorder=4,
            label="acetyl fit")
    ax.axhline(0.0, color="0.72", lw=0.8, ls=":", zorder=0)

    def _box(comp, name, colour, xpos):
        lines = [name, f"$\\delta$ = {comp['delta']:.3f} ppm"]
        if comp["present"]:
            lines += [f"FWHM = {comp['fwhm_hz']:.2f} Hz",
                      f"area = {comp['area']:.3e}",
                      f"SNR = {comp['snr']:.0f}"]
        else:
            lines.append("not detected → area = 0")
        ax.text(xpos, 0.97, "\n".join(lines), transform=ax.transAxes, va="top",
                ha="left", fontsize=8.6, color=colour,
                bbox=dict(boxstyle="round", fc="white", ec=colour, alpha=0.9))

    _box(bound, "esterified acetyl (s, 3H each)", C_EGDA, 0.02)
    _box(free, "acetic acid CH$_3$ (s, 3H)", C_ACID, 0.76)

    top = float(np.max(l_bound + l_free)) if xf.size else 1.0
    ax.set_ylim(-0.12 * (top or 1.0), 1.55 * (top or 1.0))
    ax.set_xlim(hi, lo)                                   # NMR convention
    ax.set_yticks([])
    ax.set_xlabel(r"$\delta$ / ppm")

    amt = result["amounts"]
    sid = result["sample_id"] or result["file"]
    fig.suptitle(
        f"EGDA hydrolysis — acetyl deconvolution — {sid}\n"
        f"ester-group conversion X = {result['speciation']['X_ester']*100:.2f}%   "
        f"(X = A$_{{AcOH}}$ / [A$_{{AcOH}}$ + A$_{{ester}}$])   |   "
        f"n(AcOH) = {amt['aa']*1e3:.2f} mmol   |   $R^2$ = {fit['r2']:.5f}",
        fontsize=12, fontweight="bold")
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.20, top=0.80)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.005))
    fig.savefig(out_png, dpi=cfg["dpi"])
    plt.close(fig)


def plot_vs_time(path, results, cfg):
    """Speciation / conversion / amounts vs time (needs minutes in the names)."""
    pts = [r for r in results if r["minute"] is not None]
    if len(pts) < 2:
        return False
    pts.sort(key=lambda r: r["minute"])
    t = np.array([r["minute"] for r in pts], float)
    Xe = np.array([r["speciation"]["X_ester"] for r in pts]) * 100
    known = pts[0]["known_speciation"]

    n_panels = 3 if known else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.9 * n_panels, 4.7))
    axes = np.atleast_1d(axes)
    i = 0

    if known:
        xD = np.array([r["speciation"]["x_egda"] for r in pts]) * 100
        xM = np.array([r["speciation"]["x_egma"] for r in pts]) * 100
        xG = np.array([r["speciation"]["x_eg"] for r in pts]) * 100
        ax = axes[i]; i += 1
        ax.plot(t, xD, "o-", color=C_EGDA, lw=1.7, ms=5, label="EGDA (diester)")
        ax.plot(t, xM, "s-", color=C_EGMA, lw=1.7, ms=5, label="EGMA (monoester)")
        ax.plot(t, xG, "^-", color=C_EG, lw=1.7, ms=5, label="EG (glycol)")
        ax.set_xlabel("time / min")
        ax.set_ylabel("mole fraction of the glycol pool / %")
        ax.set_title("Speciation vs time")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, color="0.9")
        ax.set_ylim(-3, 103)

    ax = axes[i]; i += 1
    ax.plot(t, Xe, "s-", color=C_ACID, lw=1.7, ms=5,
            label="ester groups hydrolysed")
    if known:
        X1 = np.array([r["speciation"]["X_egda"] for r in pts]) * 100
        ax.plot(t, X1, "o-", color=C_EGDA, lw=1.7, ms=5, label="EGDA converted")
    ax.set_xlabel("time / min")
    ax.set_ylabel("conversion / %")
    ax.set_title("Conversion vs time")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, color="0.9")
    ax.set_ylim(-3, 103)

    ax = axes[i]
    nA = np.array([r["amounts"]["aa"] for r in pts], float) * 1e3
    ax.plot(t, nA, "s-", color=C_ACID, lw=1.7, ms=5, label="acetic acid")
    if known:
        nD = np.array([r["amounts"]["D"] for r in pts], float) * 1e3
        nM = np.array([r["amounts"]["M"] for r in pts], float) * 1e3
        nG = np.array([r["amounts"]["G"] for r in pts], float) * 1e3
        ax.plot(t, nD, "o-", color=C_EGDA, lw=1.7, ms=4, label="EGDA")
        ax.plot(t, nM, "^-", color=C_EGMA, lw=1.7, ms=4, label="EGMA")
        ax.plot(t, nG, "v-", color=C_EG, lw=1.7, ms=4, label="EG")
    ax.set_xlabel("time / min")
    ax.set_ylabel("amount / mmol")
    ax.set_title("Amounts vs time")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, color="0.9")

    handle = ("backbone O–CH$_2$" if pts[0]["mode"] == "backbone"
              else "acetyl CH$_3$")
    fig.suptitle(f"Hydrolysis of ethylene glycol diacetate — conversion from the "
                 f"{handle} signals  "
                 f"({cfg['c_diester_M']:.3f} M, {cfg['v_solution_L']*1000:.0f} mL, "
                 f"{cfg['catalyst_g']:.1f} g cat., {cfg['temperature_C']:.0f} C)",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=cfg["dpi"])
    plt.close(fig)
    return True


# ===========================================================================
#  CSV WRITERS
# ===========================================================================
def write_group_csv(result, path):
    """Per-spectrum measured signals (delta, J, FWHM, area, R^2), both modes."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(arn.GROUP_COLS)
        for c in result["components"]:
            w.writerow([c["label"], c["species"], c["protons"],
                        round(c["delta"], 4), round(c["J_hz"], 3),
                        round(c["fwhm_hz"], 3), round(c["eta"], 4),
                        f"{c['area']:.6g}", f"{c['height']:.6g}",
                        round(c["snr"], 2), round(c["r2"], 6), c["present"]])
        if result["mode"] == "backbone":
            wat = result["fit"]["water"]
            w.writerow(["water (broad, excluded)", "water", "",
                        round(wat["delta"], 4), "", round(wat["fwhm_hz"], 2), "",
                        f"{wat['area']:.6g}", f"{wat['height']:.6g}", "", "", ""])


SPECIES_COLS = ["species", "formula", "role", "molar_mass_g_per_mol",
                "concentration_M", "moles", "mass_g", "mole_fraction_glycol_pool",
                "backbone_H", "acetyl_H", "exchangeable_H"]


def _num(v, nd):
    return "" if (v is None or not np.isfinite(v)) else round(float(v), nd)


def build_species_rows(result, cfg):
    """Per-compound amounts (moles, mass, molarity, glycol-pool fraction)."""
    amt = result["amounts"]
    spec = result["speciation"]
    V = cfg["v_solution_L"]
    pool = {"D": spec["x_egda"], "M": spec["x_egma"], "G": spec["x_eg"]}

    rows = []
    for key in ("D", "M", "G", "aa", "w"):
        name, formula, role, mw, bbH, acH, exH = SPECIES_INFO[key]
        n = amt.get(key, float("nan"))
        n = float(n) if n is not None else float("nan")
        rows.append({
            "species": name, "formula": formula, "role": role,
            "molar_mass_g_per_mol": mw,
            "concentration_M": _num(n / V if np.isfinite(n) else n, 5),
            "moles": _num(n, 6),
            "mass_g": _num(n * mw if np.isfinite(n) else n, 5),
            "mole_fraction_glycol_pool": _num(pool.get(key), 5),
            "backbone_H": bbH, "acetyl_H": acH, "exchangeable_H": exH,
        })
    return rows


def write_species_csv(result, cfg, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SPECIES_COLS)
        w.writeheader()
        w.writerows(build_species_rows(result, cfg))


MASTER_COLS = [
    "file", "sample_id", "minute", "mode",
    # `conversion_X` is the ester-group conversion -- the column
    # kinetics_estimation.py picks up automatically.
    "conversion_X", "conversion_percent", "X_EGDA",
    "x_EGDA", "x_EGMA", "x_EG",
    "area_EGDA", "area_EGMA_ester", "area_EGMA_OH", "area_EG",
    "total_backbone_area", "area_acetyl_bound", "area_acetyl_free",
    "total_acetyl_area",
    "delta_EGDA", "delta_EGMA_ester", "delta_EGMA_OH", "delta_EG",
    "delta_acetyl_bound", "delta_acetyl_free", "fwhm_Hz", "J_Hz", "R2",
    "n_EGDA_mol", "n_EGMA_mol", "n_EG_mol", "n_AcOH_mol", "n_H2O_mol",
    "c_EGDA_M", "c_EGMA_M", "c_EG_M", "c_AcOH_M",
]


def write_master_csv(path, results, cfg):
    """Combined kinetics table across the whole batch (same columns either mode).

    Columns a mode cannot measure are left EMPTY rather than filled with a
    model-based guess."""
    V = cfg["v_solution_L"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(MASTER_COLS)
        for r in results:
            f, s, amt = r["fit"], r["speciation"], r["amounts"]
            row = {c: "" for c in MASTER_COLS}
            row.update({
                "file": r["file"], "sample_id": r["sample_id"] or "",
                "minute": r["minute"] if r["minute"] is not None else "",
                "mode": r["mode"],
                "conversion_X": _num(s["X_ester"], 5),
                "conversion_percent": _num(s["X_ester"] * 100, 3),
                "X_EGDA": _num(s["X_egda"], 5),
                "x_EGDA": _num(s["x_egda"], 5),
                "x_EGMA": _num(s["x_egma"], 5),
                "x_EG": _num(s["x_eg"], 5),
                "R2": _num(f["r2"], 6),
                "n_AcOH_mol": _num(amt["aa"], 6),
                "n_H2O_mol": _num(amt["w"], 6),
                "c_AcOH_M": _num(amt["aa"] / V if np.isfinite(amt["aa"]) else amt["aa"], 5),
            })
            if r["mode"] == "backbone":
                row.update({
                    "area_EGDA": f"{f['egda']['area']:.6g}",
                    "area_EGMA_ester": f"{f['egma_ester']['area']:.6g}",
                    "area_EGMA_OH": f"{f['egma_oh']['area']:.6g}",
                    "area_EG": f"{f['eg']['area']:.6g}",
                    "total_backbone_area": f"{s['total_area']:.6g}",
                    "delta_EGDA": _num(f["egda"]["delta"], 4),
                    "delta_EGMA_ester": _num(f["egma_ester"]["delta"], 4),
                    "delta_EGMA_OH": _num(f["egma_oh"]["delta"], 4),
                    "delta_EG": _num(f["eg"]["delta"], 4),
                    "fwhm_Hz": _num(f["egda"]["fwhm_hz"], 3),
                    "J_Hz": _num(f["egma_ester"]["J_hz"], 3),
                    "n_EGDA_mol": _num(amt["D"], 6),
                    "n_EGMA_mol": _num(amt["M"], 6),
                    "n_EG_mol": _num(amt["G"], 6),
                    "c_EGDA_M": _num(amt["D"] / V, 5),
                    "c_EGMA_M": _num(amt["M"] / V, 5),
                    "c_EG_M": _num(amt["G"] / V, 5),
                })
            else:
                row.update({
                    "area_acetyl_bound": f"{f['bound']['area']:.6g}",
                    "area_acetyl_free": f"{f['free']['area']:.6g}",
                    "total_acetyl_area": f"{s['total_area']:.6g}",
                    "delta_acetyl_bound": _num(f["bound"]["delta"], 4),
                    "delta_acetyl_free": _num(f["free"]["delta"], 4),
                    "fwhm_Hz": _num(f["bound"]["fwhm_hz"], 3),
                })
            w.writerow([row[c] for c in MASTER_COLS])


# ===========================================================================
#  DRIVER
# ===========================================================================
def main(cfg=CONFIG):
    mode = cfg["conversion_mode"].lower()
    if mode not in ("backbone", "acetyl"):
        raise SystemExit(f"conversion_mode must be 'backbone' or 'acetyl'; "
                         f"got {mode!r}")

    files = arn.resolve_inputs(cfg, HERE)
    out_dir = os.path.join(arn.resolve_output(cfg, HERE), f"{mode}_conversion")
    os.makedirs(out_dir, exist_ok=True)

    n0_diester, n0_water = arn.initial_amounts(
        cfg["c_diester_M"] * cfg["v_solution_L"], cfg["v_solution_L"])

    handle = ("the backbone O-CH2 region (full EGDA/EGMA/EG speciation)"
              if mode == "backbone"
              else "the acetyl CH3 singlets (ester groups hydrolysed)")
    print(f"Hydrolysis of ethylene glycol diacetate - conversion from {handle}")
    print(f"  EGDA          : {cfg['c_diester_M']:.3f} M x "
          f"{cfg['v_solution_L']*1000:.0f} mL = {n0_diester:.4f} mol "
          f"({2*n0_diester:.4f} mol ester groups)")
    print(f"  water (est.)  : {n0_water:.3f} mol "
          f"(~{n0_water/max(2*n0_diester, 1e-12):.0f}x molar excess over ester groups)")
    print(f"  water sits near {arn.water_shift(_sub(cfg, 'backbone')):.2f} ppm at "
          f"{cfg['temperature_C']:.0f} C"
          + ("  <-- close to the backbone signals; cross-check with 'acetyl'"
             if mode == "backbone" else ""))
    print(f"Processing {len(files)} spectrum file(s) -> {out_dir}\n")

    anchors = None
    if mode == "backbone" and cfg["backbone"].get("auto_anchor", True) and len(files) > 1:
        anchors = arn.resolve_backbone_anchors(files, _sub(cfg, "backbone"))
        print()

    results = []
    for path in files:
        r = analyze_spectrum(path, cfg, n0_diester, n0_water, anchors=anchors)
        results.append(r)
        s, a = r["speciation"], r["amounts"]
        if mode == "backbone":
            print(f"  {r['stem']:<22} x(EGDA)={s['x_egda']*100:6.2f}%  "
                  f"x(EGMA)={s['x_egma']*100:6.2f}%  x(EG)={s['x_eg']*100:6.2f}%  "
                  f"X(ester)={s['X_ester']*100:6.2f}%  R2={r['r2']:.4f}")
        else:
            print(f"  {r['stem']:<22} X(ester)={s['X_ester']*100:6.2f}%  "
                  f"dEster={r['fit']['bound']['delta']:.3f} "
                  f"dAcOH={r['fit']['free']['delta']:.3f}  R2={r['r2']:.4f}  "
                  f"n(AcOH)={a['aa']*1e3:.2f} mmol")

        if cfg["save_fit_image"]:
            save_fit_image(r, cfg, os.path.join(out_dir, f"{r['stem']}_{mode}_fit.png"))
        if cfg["save_group_csv"]:
            write_group_csv(r, os.path.join(out_dir, f"{r['stem']}_{mode}_groups.csv"))
        if cfg["save_species_csv"]:
            write_species_csv(r, cfg,
                              os.path.join(out_dir, f"{r['stem']}_{mode}_species.csv"))

    if cfg["save_summary"] and results:
        master = os.path.join(out_dir, f"conversion_vs_time_{mode}.csv")
        write_master_csv(master, results, cfg)
        print(f"\nsaved master table -> {master}")
        plot_path = os.path.join(out_dir, f"conversion_vs_time_{mode}.png")
        if plot_vs_time(plot_path, results, cfg):
            print(f"saved vs-time plot -> {plot_path}")
        print("\nNext: point kinetics_estimation.py at the master table to fit "
              "k1 and k2.")


if __name__ == "__main__":
    main()
