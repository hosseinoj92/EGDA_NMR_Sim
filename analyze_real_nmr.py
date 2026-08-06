#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speciation of the ethylene-glycol-diacetate hydrolysis from the backbone O-CH2 region
=====================================================================================

    EGDA + H2O  ->  EGMA + AcOH          (step 1, cat. Amberlyst-15 / Lewatit)
    EGMA + H2O  ->  EG   + AcOH          (step 2)

    EGDA = ethylene glycol DIacetate   CH3COO-CH2CH2-OOCCH3
    EGMA = ethylene glycol MONOacetate CH3COO-CH2CH2-OH   (2-hydroxyethyl acetate)
    EG   = ethylene glycol             HO-CH2CH2-OH

Unlike the ethyl-acetate hydrolysis (one step, two signals), this is a
CONSECUTIVE reaction: the intermediate EGMA is both a product and a reactant, so
a single "conversion" number is not enough -- the full speciation is needed.

WHY THE BACKBONE O-CH2 REGION IS THE RIGHT HANDLE
-------------------------------------------------
The -CH2CH2- backbone is never broken, so EVERY glycol-derived molecule carries
exactly FOUR backbone protons, whatever stage of hydrolysis it is in:

    EGDA   4 H   one singlet          ~4.34 ppm   (both CH2 equivalent)
    EGMA   4 H   two 1:2:1 triplets   ~4.25 + ~3.78 ppm  (2 H each, 3J ~ 4.7 Hz)
    EG     4 H   one singlet          ~3.66 ppm   (both CH2 equivalent)

Because the proton count per molecule is IDENTICAL (4 H) for all three, the
spectrometer response and the "4 H" factor cancel and the measured areas are
directly proportional to the molar amounts:

    x_EGDA : x_EGMA : x_EG  =  A_EGDA : (A_EGMA_ester + A_EGMA_OH) : A_EG

and the total backbone area is CONSERVED, which is a free internal standard and
a built-in quality check on every single spectrum.

THE KEY CONSTRAINT THAT MAKES THE OVERLAP TRACTABLE
---------------------------------------------------
At 80 MHz (1 ppm = 80 Hz) the four components are only 5-10 Hz apart and the
EGMA triplets (2J span = ~9.4 Hz) run straight into their neighbours, so fixed
integration windows cannot separate them. Instead all four are fitted TOGETHER
with the physics wired into the model:

  * EGMA's two triplets belong to the SAME molecule with the SAME proton count
    (2 H each), so they are forced to have the SAME amplitude and therefore the
    SAME area. The well-resolved ester-side triplet at ~4.25 ppm (which only has
    the EGDA singlet nearby) therefore PREDICTS the badly-overlapped hydroxy-side
    triplet at ~3.78 ppm outright, and whatever intensity is left over at
    ~3.66 ppm must be ethylene glycol. This single tie is what makes the
    EGMA/EG split well-posed at low field.
  * all four components sit on chemically near-identical -O-CH2- carbons, so
    they SHARE one line width and one line shape. They differ only in POSITION
    and AMPLITUDE -- which is exactly what changes with conversion.
  * the whole anchor set is first fitted RIGIDLY (relative positions locked,
    one global referencing shift), then relaxed by at most `center_drift`.
    A rigid pre-pass cannot invent a phantom component on a spectrum where one
    is absent, which is what keeps the 0 % spectra honest.

WATER  --  READ THIS BEFORE TRUSTING THE NUMBERS
------------------------------------------------
The water resonance moves UPFIELD with temperature, roughly

        delta(H2O) / ppm  ~  5.051 - 0.0111 * T(degC)

    25 C -> 4.77 ppm      49 C -> 4.51 ppm      70 C -> 4.28 ppm

so at 70 C water lands within ~5 Hz of the EGDA singlet (4.34 ppm). Two things
are done about it: a smooth arPLS background removes the broad tail, and an
explicit BROAD water component is fitted inside the region (its area is reported
but excluded from the speciation). Even so, at high temperature the backbone
handle can be compromised -- the acetyl handle (main.py, ~2.1 ppm, far from
water) is immune and is the better cross-check on such data.

INPUT can be a single ascii-spec file OR a whole folder (batch).

>>> Just press "Run" in your IDE. Everything is controlled by CONFIG below. <<<
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")                 # headless: we only save PNGs
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve


# ===========================================================================
#  BACKBONE FIT SETTINGS  --  THE SINGLE SOURCE OF TRUTH
#
#  main.py imports this dictionary rather than keeping its own copy, so
#  recalibrating the anchor shifts HERE updates both scripts and the two can
#  never disagree about what they measured.
# ===========================================================================
BACKBONE_DEFAULTS = {
    # ---- BACKBONE REGION -------------------------------------------------
    # One region holding all four components plus signal-free shoulders for the
    # background. Keep the upper edge below the water maximum when you can.
    "backbone_window": (3.40, 4.58),

    # ---- ANCHOR SHIFTS (ppm) ---------------------------------------------
    # !! RECALIBRATE THESE ON YOUR FIRST SPECTRUM !!  They are aqueous
    # literature values, not values fitted from your instrument (unlike the
    # ethyl-acetate pipeline, where they came from the real data). Only the
    # RELATIVE spacing has to be right: a common offset is absorbed by the
    # global shift fitted in the rigid pre-pass.
    "egda_center":         4.335,   # EGDA  O-CH2CH2-O  singlet   (4 H)
    "egma_ester_center":   4.245,   # EGMA  -CH2-OC(O)- triplet   (2 H)
    "egma_hydroxy_center": 3.780,   # EGMA  -CH2-OH     triplet   (2 H)
    "eg_center":           3.660,   # EG    HO-CH2CH2-OH singlet  (4 H)

    # ---- LINESHAPE / FIT -------------------------------------------------
    "J_hz":         4.7,        # EGMA vicinal 3J(H,H) across the backbone
    "J_tol_hz":     1.5,        # allowed drift of J during the fit
    "init_fwhm_hz": 2.0,
    "min_fwhm_hz":  0.5,
    "max_fwhm_hz":  12.0,

    # Rigid pre-pass: all four anchors move together by one global shift.
    # KEEP THIS BELOW HALF THE SMALLEST ANCHOR SPACING (EGDA-EGMA is 0.09 ppm,
    # so below ~0.045). A wider window lets the whole set slide by one signal and
    # lock onto the wrong neighbour -- a wrong answer that still looks plausible.
    # If your real shifts are further off than this, fix the anchors above
    # instead of widening the window.
    "global_shift_max": 0.040,
    # The rigid pass is started from this many shifts spread across that window
    # and the best R^2 wins, so it cannot be trapped by whichever basin happens
    # to be nearest zero.
    "global_shift_starts": 5,
    # relaxed pass: each centre may then drift this far from its rigid position
    "refine_centers": True,
    "center_drift":   0.030,
    "sep_drift":      0.020,    # allowed change of the EGMA internal separation

    # generous for 15 parameters; caps the cost of a near-degenerate spectrum
    "maxfev": 10000,
    # The lineshape is massively oversampled (~60 points per FWHM on a 32k
    # spectrum), so the fit is run on a decimated grid keeping this many points
    # across one line. It costs nothing in accuracy -- areas are integrated
    # analytically on a fine grid afterwards -- and speeds a 15-parameter fit up
    # several-fold. Raise it if your lines are unusually narrow.
    "fit_points_per_fwhm": 12,

    # a component whose tallest line is below snr_min x noise counts as ABSENT
    # (area = 0) -> handles the 0 % spectra (no EGMA / no EG yet).
    "snr_min": 8.0,

    # ---- WATER -----------------------------------------------------------
    # An explicit BROAD component soaks up the water resonance / its base so it
    # cannot be mistaken for backbone signal. Its area is reported separately
    # and never enters the speciation. Forced much broader than the sharp lines
    # so it cannot mimic one.
    "fit_water_component": True,
    "water_shift_ppm":     None,    # None -> 5.051 - 0.0111 * temperature_C
    "water_drift_ppm":     0.60,    # how far the water centre may move
    "water_min_fwhm_hz":   15.0,
    "water_max_fwhm_hz":   400.0,

    # ---- BACKGROUND / BASELINE -------------------------------------------
    # arPLS (Baek 2015) fits a smooth CURVED background that hugs the broad tail
    # but passes under the sharp components. Keep lam stiff so it removes only
    # the smooth part and leaves the water line to the explicit component above.
    "baseline_method": "arpls",     # "arpls" (recommended) or "linear"
    "als_lambda":      1e8,
    "als_niter":       50,
    "baseline_deg":    1,           # linear fallback
    "baseline_iter":   12,
    "baseline_reject": 2.0,
}


# ===========================================================================
#  CONFIG  -- edit these; everything below is driven by this dictionary
# ===========================================================================
CONFIG = {
    # ---- INPUT: a single ascii-spec file OR a folder (batch) -------------
    "input":      r"NMR-EGDA_data",
    "file_glob":  "ascii-spec*.txt",

    # ---- OUTPUT ----------------------------------------------------------
    "output":         r"results\backbone",
    "save_fit_image": True,     # per-spectrum deconvolution picture (PNG)
    "save_group_csv": True,     # per-spectrum measured-components table
    "save_summary":   True,     # combined speciation table across the batch
    "save_json":      True,     # raw fitted parameters
    "dpi":            150,

    # ---- CHEMISTRY -------------------------------------------------------
    "n0_diester_mol": 1.0,      # initial moles of EGDA -> absolute moles
    "temperature_C":  70.0,     # C; only used to predict where water sits

    # ---- THE FIT ITSELF (region, anchors, lineshape, water, baseline) -----
    **BACKBONE_DEFAULTS,

    # noise is estimated from this signal-free window
    "noise_window": (5.5, 9.0),
}

# ---- per-compound metadata -------------------------------------------------
#   key: (name, formula, role, molar_mass, backbone_H, acetyl_H, exchangeable_H)
MW = {"D": 146.14, "M": 104.10, "G": 62.07, "aa": 60.05, "w": 18.02}
SPECIES_INFO = {
    "D":  ("ethylene glycol diacetate",  "C6H10O4", "reactant",     MW["D"],  4, 6, 0),
    "M":  ("ethylene glycol monoacetate", "C4H8O3", "intermediate", MW["M"],  4, 3, 1),
    "G":  ("ethylene glycol",             "C2H6O2", "product",      MW["G"],  4, 0, 2),
    "aa": ("acetic acid",                 "C2H4O2", "product",      MW["aa"], 0, 3, 1),
    "w":  ("water",                       "H2O",    "reactant",     MW["w"],  0, 0, 2),
}
RHO_EGDA, RHO_WATER = 1.106, 1.000        # g/mL at ~20 C

# colours: diester -> monoester -> glycol runs dark blue -> teal -> red
C_EGDA, C_EGMA, C_EG, C_ACID = "#1b3a5c", "#2a7f62", "#a23b2e", "#8c5a2b"
C_DATA, C_FIT, C_BASE, C_WATER = "#222222", "#d1495b", "0.55", "#d17c20"


# ===========================================================================
#  DATA LOADING  (shared by every script in this pipeline)
# ===========================================================================
def load_spectrum(path):
    """Parse a Bruker ASCII spectrum -> (ppm, intensity, hz, sample_id)."""
    ppm, inten, hz = [], [], []
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
                inten.append(float(parts[1]))
                hz.append(float(parts[2]))
                ppm.append(float(parts[3]))
            except ValueError:
                continue
    ppm = np.asarray(ppm, float)
    inten = np.asarray(inten, float)
    hz = np.asarray(hz, float)
    if ppm.size == 0:
        raise SystemExit(f"No numeric spectral data found in {path!r}.")
    order = np.argsort(ppm)
    return ppm[order], inten[order], hz[order], sample_id


def minute_of(path):
    """Reaction time in minutes parsed from a trailing number, else None."""
    m = re.search(r"(\d+)\s*$", os.path.splitext(os.path.basename(path))[0])
    return int(m.group(1)) if m else None


def resolve_inputs(cfg, here=None):
    """CONFIG['input'] -> a sorted list of spectrum files (single file or batch)."""
    here = here or os.path.dirname(os.path.abspath(__file__))
    inp = cfg["input"]
    if not os.path.isabs(inp):
        inp = os.path.join(here, inp)
    if os.path.isdir(inp):
        files = sorted(glob.glob(os.path.join(inp, cfg["file_glob"])),
                       key=lambda p: (minute_of(p) is None, minute_of(p) or 0, p))
        if not files:
            raise SystemExit(
                f"No files matching {cfg['file_glob']!r} in {inp!r}.\n"
                f"Put your extracted ascii-spec_<minute>.txt files there (see "
                f"Brucker_batch_extractor.py), or run make_test_data.py and "
                f"point CONFIG['input'] at 'test_data' to try the pipeline on "
                f"synthetic data with known rate constants.")
        return files
    if os.path.isfile(inp):
        return [inp]
    raise SystemExit(f"Input path does not exist: {inp!r}")


def resolve_output(cfg, here=None):
    here = here or os.path.dirname(os.path.abspath(__file__))
    out = cfg["output"]
    return out if os.path.isabs(out) else os.path.join(here, out)


# ===========================================================================
#  SMALL HELPERS  (shared)
# ===========================================================================
def slice_region(ppm, y, lo, hi):
    m = (ppm >= lo) & (ppm <= hi)
    return ppm[m], y[m]


def noise_level(ppm, y, window, deg=2):
    """Robust noise sigma from a signal-free window.

    A plain standard deviation over the window measures the BACKGROUND as much
    as the noise -- broad peak tails and instrumental drift run everywhere, and
    on a spectrum with a strong water peak that inflates sigma by an order of
    magnitude. Every presence gate downstream is `height >= snr_min * sigma`, so
    an inflated sigma silently throws away real components.

    So: detrend the window with a low-order polynomial, then take sigma from the
    median absolute deviation of the POINT-TO-POINT DIFFERENCES. Differencing
    removes any residual smooth trend and the MAD ignores the occasional spike,
    which leaves the genuine random noise. The 1/(0.6745*sqrt(2)) factor turns a
    MAD of differences back into a Gaussian standard deviation."""
    x, yy = slice_region(ppm, y, *window)
    if yy.size < 20:
        x, yy = np.asarray(ppm, float), np.asarray(y, float)
    if yy.size < 5:
        return float(np.std(yy)) or 1.0
    try:
        yy = yy - np.polyval(np.polyfit(x, yy, deg), x)
    except (np.linalg.LinAlgError, ValueError):
        pass
    d = np.diff(yy)
    mad = float(np.median(np.abs(d - np.median(d))))
    sigma = mad / (0.6745 * np.sqrt(2.0))
    if sigma <= 0:
        sigma = float(np.std(yy))
    return sigma if sigma > 0 else 1.0


def linear_baseline(x, y, n_edge=6):
    """Straight baseline through the (signal-free) edges of a window."""
    n = max(3, min(n_edge, len(x) // 3))
    xe = np.r_[x[:n], x[-n:]]
    ye = np.r_[y[:n], y[-n:]]
    a, b = np.polyfit(xe, ye, 1)
    return float(a), float(b)


def robust_baseline(x, y, deg=1, n_iter=12, reject=2.0):
    """Low-order baseline through the signal-free points of a region.

    Iteratively rejects points sitting ABOVE the current baseline (i.e. peaks),
    so the result follows the true baseline even when one component dominates."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < deg + 2:
        return np.array([0.0, float(np.median(y)) if len(y) else 0.0])
    mask = np.ones(len(x), bool)
    coef = np.polyfit(x, y, deg)
    for _ in range(n_iter):
        resid = y - np.polyval(coef, x)
        sigma = np.std(resid[mask]) if mask.sum() > deg + 1 else np.std(resid)
        if sigma <= 0:
            break
        new_mask = resid < reject * sigma
        if new_mask.sum() < deg + 2 or np.array_equal(new_mask, mask):
            break
        mask = new_mask
        coef = np.polyfit(x[mask], y[mask], deg)
    return coef


def arpls_baseline(y, lam=1e8, niter=50, tol=1e-3):
    """Smooth CURVED background via asymmetrically-reweighted penalised least
    squares (arPLS, Baek et al. Analyst 2015).

    Minimises ||w(y - z)||^2 + lam*||D2 z||^2, with the weights driven to ~0
    wherever the data sit ABOVE the baseline. The result follows the broad tail
    and passes UNDER the sharp components. `lam` sets the stiffness."""
    y = np.asarray(y, float)
    L = len(y)
    if L < 5:
        return np.full(L, float(np.min(y)) if L else 0.0)
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2))
    H = lam * (D @ D.T)
    w = np.ones(L)
    z = y.copy()
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        z = spsolve((W + H).tocsc(), w * y)
        d = y - z
        dn = d[d < 0]
        if dn.size < 2:
            break
        m, s = float(dn.mean()), float(dn.std())
        if s <= 0:
            break
        wt = 1.0 / (1.0 + np.exp(np.clip(2.0 * (d - (2.0 * s - m)) / s, -50, 50)))
        if np.linalg.norm(w - wt) / max(np.linalg.norm(w), 1e-12) < tol:
            w = wt
            break
        w = wt
    return np.asarray(z, float)


def region_baseline(x, y, cfg):
    """Smooth background over a region as an ARRAY aligned with x.

    lam is auto-scaled by the number of points so a change in spectral
    resolution keeps the same smoothness in ppm."""
    if cfg.get("baseline_method", "arpls") == "arpls":
        try:
            lam = cfg.get("als_lambda", 1e8) * (len(x) / 2400.0) ** 2
            return arpls_baseline(y, lam, cfg.get("als_niter", 50))
        except Exception:
            pass
    coef = robust_baseline(x, y, cfg.get("baseline_deg", 1),
                           cfg.get("baseline_iter", 12),
                           cfg.get("baseline_reject", 2.0))
    return np.polyval(coef, np.asarray(x, float))


def pseudo_voigt(x, c, w, h, eta):
    """Pseudo-Voigt line: `w` is the half-width at half maximum (ppm)."""
    lor = w * w / ((x - c) ** 2 + w * w)
    gau = np.exp(-np.log(2.0) * ((x - c) / w) ** 2)
    return h * (eta * lor + (1.0 - eta) * gau)


def r_squared(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def water_shift(cfg):
    """Where the water resonance is expected: delta ~ 5.051 - 0.0111*T(degC)."""
    fixed = cfg.get("water_shift_ppm")
    if fixed is not None:
        return float(fixed)
    return 5.051 - 0.0111 * float(cfg.get("temperature_C", 25.0))


# ===========================================================================
#  BACKBONE LINESHAPE  --  four components with the physics wired in
# ===========================================================================
def triplet_121(x, center, J_ppm, w, h, eta):
    """A 1:2:1 backbone triplet (2 H coupled to 2 equivalent vicinal H).

    `h` is the height of the central line; the outer lines are h/2."""
    return (pseudo_voigt(x, center + J_ppm, w, 0.5 * h, eta)
            + pseudo_voigt(x, center, w, 1.0 * h, eta)
            + pseudo_voigt(x, center - J_ppm, w, 0.5 * h, eta))


def backbone_model(x, a, b, cD, cM, sM, cG, w, eta, J_ppm, hD, hM, hG,
                   cW, wW, hW):
    """Linear baseline + broad water + the four backbone components.

    Note `hM` appears TWICE: EGMA's ester-side and hydroxy-side triplets are the
    same 2 H of the same molecule seen at two shifts, so with a shared width and
    shape they must have the same amplitude -- and therefore the same area. That
    tie is what lets the clean ~4.25 ppm triplet resolve the EGMA/EG overlap
    down at ~3.7 ppm."""
    return (a * x + b
            + pseudo_voigt(x, cW, wW, hW, 1.0)          # broad water (excluded)
            + pseudo_voigt(x, cD, w, hD, eta)           # EGDA   singlet  4 H
            + triplet_121(x, cM, J_ppm, w, hM, eta)     # EGMA   -CH2-O2CR 2 H
            + triplet_121(x, cM - sM, J_ppm, w, hM, eta)  # EGMA -CH2-OH   2 H
            + pseudo_voigt(x, cG, w, hG, eta))          # EG     singlet  4 H


def _rigid_backbone(anchors):
    """backbone_model with the four centres locked to `anchors` + one shift."""
    cD0, cM0, cM2_0, cG0 = anchors
    sM0 = cM0 - cM2_0

    def f(x, a, b, shift, w, eta, J_ppm, hD, hM, hG, cW, wW, hW):
        return backbone_model(x, a, b, cD0 + shift, cM0 + shift, sM0,
                              cG0 + shift, w, eta, J_ppm, hD, hM, hG,
                              cW, wW, hW)
    return f


def anchors_of(cfg):
    """(EGDA, EGMA-ester, EGMA-hydroxy, EG) centres from CONFIG."""
    return (float(cfg["egda_center"]), float(cfg["egma_ester_center"]),
            float(cfg["egma_hydroxy_center"]), float(cfg["eg_center"]))


# ===========================================================================
#  BACKBONE FIT
# ===========================================================================
def measure_backbone(ppm, inten, cfg, sf, noise, anchors=None):
    """Deconvolve the backbone region into EGDA / EGMA / EG and integrate each.

    Two passes. The RIGID pass locks the relative positions of all four
    components and fits a single global referencing shift, which cannot invent a
    phantom component where none exists. The RELAXED pass then lets each centre
    drift by at most `center_drift` from its rigid position and is kept only if
    it genuinely improves R^2.

    Returns a dict with the fit, the reconstructed component areas and
    everything the plotter / CSV writers need."""
    anchors = anchors or anchors_of(cfg)
    lo, hi = cfg["backbone_window"]

    # ---- remove the smooth background so everything sits on a flat zero ----
    xr, yr = slice_region(ppm, inten, lo, hi)
    if xr.size < 50:
        raise ValueError(f"Backbone window {lo}-{hi} ppm holds too few points")
    bl = region_baseline(xr, yr, cfg)
    x, y = xr, yr - bl

    # ---- normalise to O(1): positions are ~1e0 ppm but intensities ~1e8, and
    # that 8-order span makes curve_fit ill-conditioned. Fit normalised, rescale.
    yscale = float(np.max(np.abs(y))) if y.size else 1.0
    yscale = yscale if yscale > 0 else 1.0
    yn = y / yscale
    noise_n = noise / yscale

    a0, b0 = linear_baseline(x, yn, n_edge=max(6, len(x) // 20))
    yc = yn - (a0 * x + b0)
    ymax = float(yc.max()) if yc.size else 1.0
    ymax = ymax if ymax > 0 else 1.0

    def _val_at(c):
        if c < x.min() or c > x.max():
            return 0.05 * ymax
        return float(max(yc[np.argmin(np.abs(x - c))], 0.05 * ymax))

    cD0, cM0, cM2_0, cG0 = anchors
    J_ppm0 = cfg["J_hz"] / sf
    Jmin = max((cfg["J_hz"] - cfg["J_tol_hz"]) / sf, 1e-4)
    Jmax = (cfg["J_hz"] + cfg["J_tol_hz"]) / sf
    wmin = cfg["min_fwhm_hz"] / 2.0 / sf
    wmax = cfg["max_fwhm_hz"] / 2.0 / sf
    w0 = float(np.clip(cfg["init_fwhm_hz"] / 2.0 / sf, wmin, wmax))

    # broad water component (kept out of the speciation)
    if cfg.get("fit_water_component", True):
        cW0 = water_shift(cfg)
        wdr = float(cfg["water_drift_ppm"])
        wWmin = cfg["water_min_fwhm_hz"] / 2.0 / sf
        wWmax = cfg["water_max_fwhm_hz"] / 2.0 / sf
        hW_ub = 50.0
    else:
        cW0, wdr = 0.5 * (lo + hi), 1e-6
        wWmin = wWmax = cfg["water_max_fwhm_hz"] / 2.0 / sf
        hW_ub = 0.0
    wW0 = float(np.clip(2.0 * wWmin, wWmin, wWmax))

    # ---- decimate the fit grid: ~60 points per FWHM is far more than the fit
    # needs, and a 15-parameter least squares over every point is what makes the
    # degenerate (single-component) spectra slow.
    per_fwhm = max(int(cfg.get("fit_points_per_fwhm", 12)), 4)
    dppm = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    step = max(1, int((2.0 * w0) / (per_fwhm * dppm))) if dppm > 0 else 1
    xs, ys = x[::step], yn[::step]

    # ---------------- pass 1: rigid ----------------
    gsm = float(cfg["global_shift_max"])
    #     a      b     shift  w    eta  J       hD        hM         hG     cW   wW   hW
    p0 = [a0, b0, 0.0, w0, 0.5, J_ppm0,
          _val_at(cD0), _val_at(cM0), _val_at(cG0), cW0, wW0, 0.0]
    lb = [-10., -10., -gsm, wmin, 0.0, Jmin, 0.0, 0.0, 0.0, cW0 - wdr, wWmin, 0.0]
    ub = [10., 10., gsm, wmax, 1.0, Jmax, 3.0, 3.0, 3.0, cW0 + wdr, wWmax,
          max(hW_ub, 1e-9)]
    p0 = [min(max(v, l), u) for v, l, u in zip(p0, lb, ub)]

    # Multi-start over the global shift. The anchor set is only ~0.09 ppm apart
    # in places, so a single start can slide into the neighbouring alignment and
    # sit there; starting from several shifts and keeping the best R^2 makes the
    # referencing robust at a cost of a few hundredths of a second.
    rigid = _rigid_backbone(anchors)
    n_starts = max(int(cfg.get("global_shift_starts", 5)), 1)
    starts = [0.0] if n_starts == 1 else list(np.linspace(-gsm, gsm, n_starts))
    pr, r2_rigid, ok = np.asarray(p0, float), -np.inf, False
    for shift0 in starts:
        trial = list(p0)
        trial[2] = float(np.clip(shift0, lb[2], ub[2]))
        try:
            cand, _ = curve_fit(rigid, xs, ys, p0=trial, bounds=(lb, ub),
                                x_scale="jac", maxfev=cfg["maxfev"])
        except Exception:
            continue
        r2 = r_squared(ys, rigid(xs, *cand))
        if r2 > r2_rigid:
            pr, r2_rigid, ok = cand, r2, True
    if not ok:
        r2_rigid = 0.0

    a_, b_, shift, w_, eta_, J_, hD_, hM_, hG_, cW_, wW_, hW_ = (float(v) for v in pr)
    best = [a_, b_, cD0 + shift, cM0 + shift, cM0 - cM2_0, cG0 + shift,
            w_, eta_, J_, hD_, hM_, hG_, cW_, wW_, hW_]
    best_r2, stage = r2_rigid, "rigid"

    # ---------------- pass 2: relaxed ----------------
    # Only components the rigid pass actually SAW get to move. Letting the
    # centre of an absent component drift is a perfectly flat direction in the
    # objective: the optimiser wanders along it for thousands of evaluations and
    # can park a phantom peak somewhere useful-looking. Pinning them keeps the
    # 0 % spectra honest and makes the fit several times faster.
    gate_n = float(cfg["snr_min"]) * noise_n
    seen = (hD_ >= gate_n, hM_ >= gate_n, hG_ >= gate_n)
    if ok and cfg.get("refine_centers", True) and sum(seen) >= 2:
        cdr, sdr = float(cfg["center_drift"]), float(cfg["sep_drift"])
        sM0 = cM0 - cM2_0
        dD = cdr if seen[0] else 1e-6
        dM = cdr if seen[1] else 1e-6
        dG = cdr if seen[2] else 1e-6
        dS = sdr if seen[1] else 1e-6
        p2 = list(best)
        lb2 = [-10., -10., best[2] - dD, best[3] - dM, max(sM0 - dS, 1e-4),
               best[5] - dG, wmin, 0.0, Jmin, 0.0, 0.0, 0.0,
               cW0 - wdr, wWmin, 0.0]
        ub2 = [10., 10., best[2] + dD, best[3] + dM, sM0 + dS,
               best[5] + dG, wmax, 1.0, Jmax, 3.0, 3.0, 3.0,
               cW0 + wdr, wWmax, max(hW_ub, 1e-9)]
        p2 = [min(max(v, l), u) for v, l, u in zip(p2, lb2, ub2)]
        try:
            pf, _ = curve_fit(backbone_model, xs, ys, p0=p2, bounds=(lb2, ub2),
                              x_scale="jac", maxfev=cfg["maxfev"])
            r2_free = r_squared(ys, backbone_model(xs, *pf))
            if r2_free > best_r2:
                best, best_r2, stage = [float(v) for v in pf], r2_free, "relaxed"
        except Exception:
            pass

    # ---- rescale the amplitude-like parameters back to real units ----------
    popt = list(best)
    for i in (0, 1, 9, 10, 11, 14):        # a, b, hD, hM, hG, hW
        popt[i] *= yscale
    (a, b, cD, cM, sM, cG, w, eta, J_ppm, hD, hM, hG, cW, wW, hW) = popt
    cM2 = cM - sM

    # ---- areas: integrate each reconstructed component on a fine grid ------
    xf = np.linspace(lo, hi, 8000)
    area_D = float(np.trapezoid(pseudo_voigt(xf, cD, w, hD, eta), xf))
    area_M1 = float(np.trapezoid(triplet_121(xf, cM, J_ppm, w, hM, eta), xf))
    area_M2 = float(np.trapezoid(triplet_121(xf, cM2, J_ppm, w, hM, eta), xf))
    area_G = float(np.trapezoid(pseudo_voigt(xf, cG, w, hG, eta), xf))
    area_W = float(np.trapezoid(pseudo_voigt(xf, cW, wW, hW, 1.0), xf))

    # ---- presence gate: below snr_min x noise counts as physically absent ---
    gate = float(cfg["snr_min"]) * noise
    pres_D, pres_M, pres_G = hD >= gate, hM >= gate, hG >= gate
    if not pres_D:
        area_D = 0.0
    if not pres_M:
        area_M1 = area_M2 = 0.0
    if not pres_G:
        area_G = 0.0

    fwhm = 2.0 * w * sf
    comp = lambda label, species, nH, delta, area, height, present, J_hz: {
        "label": label, "species": species, "protons": nH, "delta": delta,
        "J_hz": J_hz, "fwhm_hz": fwhm, "eta": eta, "area": max(area, 0.0),
        "height": height, "r2": best_r2, "present": bool(present),
        "snr": height / noise if noise > 0 else float("inf"),
    }
    return {
        "window": (lo, hi), "x": x, "y": y,
        "baseline_x": xr, "baseline_y": bl,
        "popt": popt, "r2": best_r2, "ok": ok, "stage": stage, "noise": noise,
        "yscale": yscale, "sf": sf,
        "egda":       comp("EGDA O-CH2 (s)", "ethylene glycol diacetate", 4,
                           cD, area_D, hD, pres_D, 0.0),
        "egma_ester": comp("EGMA CH2-OC(O) (t)", "ethylene glycol monoacetate", 2,
                           cM, area_M1, hM, pres_M, J_ppm * sf),
        "egma_oh":    comp("EGMA CH2-OH (t)", "ethylene glycol monoacetate", 2,
                           cM2, area_M2, hM, pres_M, J_ppm * sf),
        "eg":         comp("EG O-CH2 (s)", "ethylene glycol", 4,
                           cG, area_G, hG, pres_G, 0.0),
        "water":      {"delta": cW, "fwhm_hz": 2.0 * wW * sf, "area": max(area_W, 0.0),
                       "height": hW},
    }


def resolve_backbone_anchors(files, cfg, verbose=True):
    """Refine the four anchor shifts from the whole batch.

    Runs the rigid pre-pass on every spectrum and takes, for each component, the
    MEDIAN fitted centre over the spectra where that component is clearly
    present. A component never seen keeps its CONFIG value. This turns the
    literature anchors into instrument/sample-specific ones without any manual
    peak picking, and gives every spectrum a common reference frame."""
    seen = {"egda": [], "egma_ester": [], "egma_oh": [], "eg": []}
    base = anchors_of(cfg)
    for path in files:
        ppm, inten, hz, _ = load_spectrum(path)
        sf = float(np.polyfit(ppm, hz, 1)[0])
        noise = noise_level(ppm, inten, cfg["noise_window"])
        try:
            fit = measure_backbone(ppm, inten, cfg, sf, noise, anchors=base)
        except Exception:
            continue
        if fit["r2"] < 0.90:
            continue
        for key in seen:
            c = fit[key]
            if c["present"] and c["snr"] > 3.0 * cfg["snr_min"]:
                seen[key].append(c["delta"])

    out = []
    for key, fallback in zip(("egda", "egma_ester", "egma_oh", "eg"), base):
        vals = seen[key]
        out.append(float(np.median(vals)) if len(vals) >= 2 else fallback)

    # the EGMA pair must keep a sane internal separation even if one side was
    # only ever seen on noisy spectra
    sM_cfg = base[1] - base[2]
    if not (0.5 * sM_cfg < out[1] - out[2] < 1.5 * sM_cfg):
        out[2] = out[1] - sM_cfg

    if verbose:
        print(f"  batch anchors: EGDA {out[0]:.3f}  EGMA {out[1]:.3f}/{out[2]:.3f}  "
              f"EG {out[3]:.3f} ppm   (CONFIG: "
              f"{base[0]:.3f} {base[1]:.3f}/{base[2]:.3f} {base[3]:.3f})")
    return tuple(out)


# ===========================================================================
#  SPECIATION  (areas -> mole fractions -> conversions)
# ===========================================================================
def speciation_from_areas(area_egda, area_egma_total, area_eg):
    """Glycol-pool mole fractions and the two conversions.

    All three species carry FOUR backbone protons per molecule, so the common
    4 H factor cancels and the area fractions ARE the mole fractions."""
    tot = area_egda + area_egma_total + area_eg
    if tot <= 0:
        return {"x_egda": 1.0, "x_egma": 0.0, "x_eg": 0.0,
                "X_egda": 0.0, "X_ester": 0.0, "total_area": 0.0}
    xD = max(area_egda, 0.0) / tot
    xM = max(area_egma_total, 0.0) / tot
    xG = max(area_eg, 0.0) / tot
    return {
        "x_egda": xD, "x_egma": xM, "x_eg": xG,
        # fraction of the DIESTER that has reacted at all (step 1)
        "X_egda": min(max(xM + xG, 0.0), 1.0),
        # fraction of ALL ester groups hydrolysed: EGDA has 2, EGMA has 1
        "X_ester": min(max(0.5 * (xM + 2.0 * xG), 0.0), 1.0),
        "total_area": tot,
    }


def initial_amounts(n0_diester, v_solution_L=None):
    """(n0 EGDA, n0 water). Water is estimated by volume when V is known."""
    if not v_solution_L:
        return n0_diester, float("nan")
    egda_vol_mL = n0_diester * MW["D"] / RHO_EGDA
    water_vol_mL = max(v_solution_L * 1000.0 - egda_vol_mL, 0.0)
    return n0_diester, water_vol_mL * RHO_WATER / MW["w"]


def amounts_at(spec, n0_diester, n0_water=float("nan")):
    """Moles of every species from the measured glycol-pool composition."""
    nD = n0_diester * spec["x_egda"]
    nM = n0_diester * spec["x_egma"]
    nG = n0_diester * spec["x_eg"]
    n_acid = nM + 2.0 * nG          # one AcOH per ester group hydrolysed
    n_water = (n0_water - n_acid) if np.isfinite(n0_water) else float("nan")
    return {"D": nD, "M": nM, "G": nG, "aa": n_acid,
            "w": max(n_water, 0.0) if np.isfinite(n_water) else float("nan")}


# ===========================================================================
#  PLOTTING
# ===========================================================================
def save_fit_image(result, cfg, out_png, title_extra=""):
    """Two panels: the region as measured, then the same with water removed.

    The water resonance can be an order of magnitude taller than every backbone
    signal put together, so a single panel scaled to the data shows nothing but
    water. The left panel therefore shows the fit against the real, water-
    dominated data (so you can see whether the water model is sane), and the
    right panel shows the same fit after the broad water component and the local
    baseline have been taken out -- which is what the speciation is actually
    measured from."""
    fit = result["fit"]
    x, y = fit["x"], fit["y"]
    (a, b, cD, cM, sM, cG, w, eta, J_ppm, hD, hM, hG, cW, wW, hW) = fit["popt"]
    lo, hi = fit["window"]
    cM2 = cM - sM

    xf = np.linspace(lo, hi, 4000)
    water_f = pseudo_voigt(xf, cW, wW, hW, 1.0)
    base_f = a * xf + b + water_f
    parts = [
        ("egda", pseudo_voigt(xf, cD, w, hD, eta), C_EGDA, "EGDA O–CH$_2$ (s, 4H)"),
        ("egma_ester", triplet_121(xf, cM, J_ppm, w, hM, eta), C_EGMA,
         "EGMA CH$_2$–OC(O) (t, 2H)"),
        ("egma_oh", triplet_121(xf, cM2, J_ppm, w, hM, eta), C_EGMA,
         "EGMA CH$_2$–OH (t, 2H)"),
        ("eg", pseudo_voigt(xf, cG, w, hG, eta), C_EG, "EG O–CH$_2$ (s, 4H)"),
    ]
    sharp_f = sum(p[1] for p in parts)
    total_f = base_f + sharp_f
    # the same removal applied to the DATA, so the right panel compares like
    # with like
    y_corr = y - (a * x + b) - pseudo_voigt(x, cW, wW, hW, 1.0)

    fig, (ax_raw, ax) = plt.subplots(1, 2, figsize=(15, 5.8))

    # ---- left: as measured, water and all -------------------------------
    ax_raw.plot(x, y, color=C_DATA, lw=1.1, label="background-removed data",
                zorder=4)
    ax_raw.plot(xf, base_f, color=C_WATER, lw=1.1, ls=":", zorder=3,
                label="local baseline + broad water")
    ax_raw.plot(xf, total_f, color=C_FIT, lw=1.3, ls="--", zorder=5,
                label="total fit")
    ax_raw.axhline(0.0, color="0.72", lw=0.8, ls=":", zorder=0)
    ax_raw.set_xlim(hi, lo)
    ax_raw.set_yticks([])
    ax_raw.set_xlabel(r"$\delta$ / ppm")
    ax_raw.set_title("as measured — the water resonance dominates",
                     fontsize=10.5, fontweight="bold")
    ax_raw.legend(loc="upper right", fontsize=8, frameon=False)
    ax_raw.text(0.03, 0.02,
                f"water fitted at {fit['water']['delta']:.2f} ppm, "
                f"FWHM {fit['water']['fwhm_hz']:.0f} Hz\n(excluded from the "
                f"speciation)",
                transform=ax_raw.transAxes, ha="left", va="bottom", fontsize=8.2,
                color=C_WATER)

    # ---- right: water removed, scaled to the signals we measure ----------
    ax.plot(x, y_corr, color=C_DATA, lw=1.15, zorder=6,
            label="data, water + baseline removed")
    stacked = np.zeros_like(xf)
    for key, comp, colour, label in parts:
        if not fit[key]["present"]:
            continue
        ax.fill_between(xf, stacked, stacked + comp, color=colour, alpha=0.20,
                        zorder=1, label=label)
        ax.plot(xf, stacked + comp, color=colour, lw=1.0, zorder=3)
        stacked = stacked + comp
    ax.plot(xf, sharp_f, color=C_FIT, lw=1.4, ls="--", zorder=5,
            label="four-component fit")
    ax.axhline(0.0, color="0.72", lw=0.8, ls=":", zorder=0)

    top = float(np.max(sharp_f)) if sharp_f.size else 1.0
    top = top if top > 0 else 1.0
    ax.set_ylim(-0.12 * top, 1.85 * top)      # headroom for the three boxes

    def _box(keys, name, colour, xpos):
        c = fit[keys[0]]
        lines = [name]
        if c["present"]:
            deltas = " / ".join(f"{fit[k]['delta']:.3f}" for k in keys)
            area = sum(fit[k]["area"] for k in keys)
            lines += [f"$\\delta$ = {deltas} ppm",
                      f"FWHM = {c['fwhm_hz']:.2f} Hz",
                      f"area = {area:.3e}",
                      f"SNR = {c['snr']:.0f}"]
        else:
            lines += [f"$\\delta$ = {c['delta']:.3f} ppm",
                      f"not detected (SNR {c['snr']:.1f})",
                      "→ area = 0"]
        ax.text(xpos, 0.985, "\n".join(lines), transform=ax.transAxes, va="top",
                ha="left", fontsize=7.8, color=colour, linespacing=1.35,
                bbox=dict(boxstyle="round", fc="white", ec=colour, alpha=0.92))

    _box(["egda"], "EGDA (diester)", C_EGDA, 0.015)
    _box(["egma_ester", "egma_oh"], "EGMA (monoester)", C_EGMA, 0.345)
    _box(["eg"], "EG (glycol)", C_EG, 0.755)

    ax.set_xlim(hi, lo)                                   # NMR convention
    ax.set_yticks([])
    ax.set_xlabel(r"$\delta$ / ppm")
    ax.set_title(f"water removed — what the speciation is measured from   "
                 f"($R^2$ = {fit['r2']:.5f}, {fit['stage']})",
                 fontsize=10.5, fontweight="bold")

    spec = result["speciation"]
    sid = result.get("sample_id") or result.get("stem", "")
    fig.suptitle(
        f"EGDA hydrolysis — backbone O–CH$_2$ deconvolution — {sid}{title_extra}\n"
        f"x(EGDA) = {spec['x_egda']*100:.1f}%   "
        f"x(EGMA) = {spec['x_egma']*100:.1f}%   "
        f"x(EG) = {spec['x_eg']*100:.1f}%   |   "
        f"ester-group conversion X = {spec['X_ester']*100:.2f}%",
        fontsize=12.5, fontweight="bold")
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.20, top=0.82,
                        wspace=0.09)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.005))
    fig.savefig(out_png, dpi=cfg["dpi"])
    plt.close(fig)


def plot_speciation_vs_time(path, results, cfg, subtitle=""):
    """Mole fractions and conversions vs time (needs minutes in the filenames)."""
    pts = [r for r in results if r["minute"] is not None]
    if len(pts) < 2:
        return False
    pts.sort(key=lambda r: r["minute"])
    t = np.array([r["minute"] for r in pts], float)
    xD = np.array([r["speciation"]["x_egda"] for r in pts]) * 100
    xM = np.array([r["speciation"]["x_egma"] for r in pts]) * 100
    xG = np.array([r["speciation"]["x_eg"] for r in pts]) * 100
    X1 = np.array([r["speciation"]["X_egda"] for r in pts]) * 100
    Xe = np.array([r["speciation"]["X_ester"] for r in pts]) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax1.plot(t, xD, "o-", color=C_EGDA, lw=1.7, ms=5, label="EGDA (diester)")
    ax1.plot(t, xM, "s-", color=C_EGMA, lw=1.7, ms=5, label="EGMA (monoester)")
    ax1.plot(t, xG, "^-", color=C_EG, lw=1.7, ms=5, label="EG (glycol)")
    ax1.set_xlabel("time / min")
    ax1.set_ylabel("mole fraction of the glycol pool / %")
    ax1.set_title("Speciation vs time")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(True, color="0.9")
    ax1.set_ylim(-3, 103)

    ax2.plot(t, X1, "o-", color=C_EGDA, lw=1.7, ms=5, label="EGDA converted")
    ax2.plot(t, Xe, "s-", color=C_ACID, lw=1.7, ms=5, label="ester groups hydrolysed")
    ax2.set_xlabel("time / min")
    ax2.set_ylabel("conversion / %")
    ax2.set_title("Conversion vs time")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(True, color="0.9")
    ax2.set_ylim(-3, 103)

    fig.suptitle("Hydrolysis of ethylene glycol diacetate — from the backbone "
                 "O–CH$_2$ signals" + subtitle, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=cfg["dpi"])
    plt.close(fig)
    return True


# ===========================================================================
#  CSV WRITERS
# ===========================================================================
GROUP_COLS = ["group", "species", "protons_per_molecule", "delta_ppm", "J_Hz",
              "fwhm_Hz", "eta", "area_under_curve", "peak_height", "SNR", "R2",
              "present"]


def write_group_csv(result, path):
    """Per-spectrum measured components (delta, J, FWHM, area, R^2)."""
    fit = result["fit"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(GROUP_COLS)
        for key in ("egda", "egma_ester", "egma_oh", "eg"):
            c = fit[key]
            w.writerow([c["label"], c["species"], c["protons"],
                        round(c["delta"], 4), round(c["J_hz"], 3),
                        round(c["fwhm_hz"], 3), round(c["eta"], 4),
                        f"{c['area']:.6g}", f"{c['height']:.6g}",
                        round(c["snr"], 2), round(c["r2"], 6), c["present"]])
        wat = fit["water"]
        w.writerow(["water (broad, excluded)", "water", "", round(wat["delta"], 4),
                    "", round(wat["fwhm_hz"], 2), "", f"{wat['area']:.6g}",
                    f"{wat['height']:.6g}", "", "", ""])


SUMMARY_COLS = ["file", "sample_id", "minute", "x_EGDA", "x_EGMA", "x_EG",
                "X_EGDA", "X_ester", "X_ester_percent", "area_EGDA",
                "area_EGMA_ester", "area_EGMA_OH", "area_EG",
                "total_backbone_area", "delta_EGDA", "delta_EGMA_ester",
                "delta_EGMA_OH", "delta_EG", "fwhm_Hz", "J_Hz", "R2", "stage",
                "water_ppm", "n_EGDA_mol", "n_EGMA_mol", "n_EG_mol", "n_AcOH_mol"]


def write_summary_csv(path, results, cfg):
    n0 = float(cfg["n0_diester_mol"])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(SUMMARY_COLS)
        for r in results:
            f, s = r["fit"], r["speciation"]
            amt = amounts_at(s, n0)
            w.writerow([
                r["file"], r["sample_id"] or "",
                r["minute"] if r["minute"] is not None else "",
                round(s["x_egda"], 5), round(s["x_egma"], 5), round(s["x_eg"], 5),
                round(s["X_egda"], 5), round(s["X_ester"], 5),
                round(s["X_ester"] * 100, 3),
                f"{f['egda']['area']:.6g}", f"{f['egma_ester']['area']:.6g}",
                f"{f['egma_oh']['area']:.6g}", f"{f['eg']['area']:.6g}",
                f"{s['total_area']:.6g}",
                round(f["egda"]["delta"], 4), round(f["egma_ester"]["delta"], 4),
                round(f["egma_oh"]["delta"], 4), round(f["eg"]["delta"], 4),
                round(f["egda"]["fwhm_hz"], 3),
                round(f["egma_ester"]["J_hz"], 3),
                round(f["r2"], 6), f["stage"], round(f["water"]["delta"], 3),
                round(amt["D"], 6), round(amt["M"], 6), round(amt["G"], 6),
                round(amt["aa"], 6)])


def _json_safe(result):
    f = result["fit"]
    out = {"file": result["file"], "sample_id": result["sample_id"],
           "minute": result["minute"], "sf_mhz": float(f["sf"]),
           "r2": float(f["r2"]), "stage": f["stage"],
           "popt": [float(v) for v in f["popt"]],
           "speciation": {k: float(v) for k, v in result["speciation"].items()},
           "components": {}}
    for key in ("egda", "egma_ester", "egma_oh", "eg"):
        c = f[key]
        out["components"][key] = {
            "delta_ppm": float(c["delta"]), "area": float(c["area"]),
            "height": float(c["height"]), "fwhm_hz": float(c["fwhm_hz"]),
            "snr": float(c["snr"]), "present": bool(c["present"])}
    out["water"] = {k: float(v) for k, v in f["water"].items()}
    return out


# ===========================================================================
#  DRIVER
# ===========================================================================
def analyze_one(path, cfg, anchors=None):
    """Analyse one spectrum -> a normalised result dict."""
    ppm, inten, hz, sample_id = load_spectrum(path)
    sf = float(np.polyfit(ppm, hz, 1)[0])
    noise = noise_level(ppm, inten, cfg["noise_window"])
    fit = measure_backbone(ppm, inten, cfg, sf, noise, anchors=anchors)
    spec = speciation_from_areas(fit["egda"]["area"],
                                 fit["egma_ester"]["area"] + fit["egma_oh"]["area"],
                                 fit["eg"]["area"])
    return {"file": os.path.basename(path),
            "stem": os.path.splitext(os.path.basename(path))[0],
            "sample_id": sample_id, "minute": minute_of(path),
            "ppm": ppm, "inten": inten, "fit": fit, "speciation": spec}


def main(cfg=CONFIG):
    files = resolve_inputs(cfg)
    out_dir = resolve_output(cfg)
    os.makedirs(out_dir, exist_ok=True)

    print("Hydrolysis of ethylene glycol diacetate - speciation from the "
          "backbone O-CH2 region")
    print("  EGDA -> EGMA -> EG   (all three carry 4 backbone H, so the area "
          "fractions ARE the mole fractions)")
    print(f"  water expected near {water_shift(cfg):.2f} ppm at "
          f"{cfg['temperature_C']:.0f} C")
    print(f"Processing {len(files)} spectrum file(s) -> {out_dir}\n")

    anchors = resolve_backbone_anchors(files, cfg) if len(files) > 1 else anchors_of(cfg)
    print()

    results = []
    for path in files:
        r = analyze_one(path, cfg, anchors=anchors)
        results.append(r)
        s = r["speciation"]
        print(f"  {r['stem']:<22} x(EGDA)={s['x_egda']*100:6.2f}%  "
              f"x(EGMA)={s['x_egma']*100:6.2f}%  x(EG)={s['x_eg']*100:6.2f}%  "
              f"X(ester)={s['X_ester']*100:6.2f}%  R2={r['fit']['r2']:.4f}")

        if cfg["save_fit_image"]:
            save_fit_image(r, cfg, os.path.join(out_dir, f"{r['stem']}_backbone_fit.png"))
        if cfg["save_group_csv"]:
            write_group_csv(r, os.path.join(out_dir, f"{r['stem']}_backbone_groups.csv"))

    if cfg["save_summary"] and results:
        spath = os.path.join(out_dir, "speciation_vs_time.csv")
        write_summary_csv(spath, results, cfg)
        print(f"\nsaved speciation table -> {spath}")
        ppath = os.path.join(out_dir, "speciation_vs_time.png")
        if plot_speciation_vs_time(ppath, results, cfg):
            print(f"saved speciation plot  -> {ppath}")
    if cfg["save_json"] and results:
        jpath = os.path.join(out_dir, "fitted_components.json")
        with open(jpath, "w", encoding="utf-8") as fh:
            json.dump([_json_safe(r) for r in results], fh, indent=2)
        print(f"saved fitted parameters -> {jpath}")


if __name__ == "__main__":
    main()
