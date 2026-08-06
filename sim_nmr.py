#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulated 1H NMR (80 MHz) of the acid-catalysed hydrolysis of ethylene glycol diacetate
=======================================================================================

    EGDA + H2O  <=>  EGMA + AcOH        (step 1, cat. Amberlyst 15, -SO3H)
    EGMA + H2O  <=>  EG   + AcOH        (step 2)

    EGDA = CH3COO-CH2CH2-OOCCH3     EGMA = CH3COO-CH2CH2-OH     EG = HO-CH2CH2-OH

This script builds a first-order 1H NMR spectrum from the underlying quantum-
mechanical rules, for a CONSECUTIVE reaction whose intermediate (EGMA) rises and
falls. Use it to see what your real spectra should look like, to check that your
integration windows are where you think they are, and -- most importantly -- to
check WHERE THE WATER PEAK WILL LAND at your reaction temperature.

Physics encoded here (all from the nuclear-spin Hamiltonian in the high-field
limit, H/h = sum_i nu_i Iz_i + sum_{i<j} J_ij Ii.Ij):

  1. Peak POSITION  -> chemical shift delta (ppm). On an 80 MHz instrument,
                       1 ppm <-> 80 Hz, so frequency offsets scale with field
                       but J (in Hz) does not.
  2. Peak SPLITTING -> a nucleus coupled to n EQUIVALENT spin-1/2 partners gives
                       n+1 lines, spacing J, with binomial intensities. EGDA and
                       EG are SYMMETRIC, so their four backbone protons are
                       equivalent and give a SINGLET; only the desymmetrised
                       intermediate EGMA shows the 1:2:1 triplet pair.
  3. Peak AREA      -> integrated intensity proportional to (#protons) x
                       (concentration) -- what turns a spectrum into kinetics.
  4. EXCHANGE       -> under acid catalysis H2O, the EGMA/EG hydroxyls and the
                       AcOH COO-H exchange fast and COALESCE into one broad line
                       at the population-weighted-average shift, which drifts as
                       acetic acid accumulates.
  5. TEMPERATURE    -> the water resonance moves upfield by ~0.011 ppm per degC,
                       delta(H2O) ~ 5.051 - 0.0111*T. At 70 C that is ~4.28 ppm
                       -- essentially on top of the EGDA singlet. The simulation
                       shows you this collision before it ruins your data.

SPECIATION: the composition is not a single "conversion". At an overall
ester-group conversion X the mixture is set by the consecutive scheme
EGDA -k1-> EGMA -k2-> EG. The default is the STATISTICAL case (the two EGDA
ester groups hydrolyse independently at the same intrinsic rate, so k1 = 2k and
k2 = k, i.e. k2/k1 = 0.5), for which the composition is simply binomial:

        x(EGDA) = p^2      x(EGMA) = 2p(1-p)      x(EG) = (1-p)^2
        with p = 1 - X = the fraction of ester groups still esterified.

Set CONFIG["k2_over_k1"] to something else to see how a faster or slower second
step changes how much intermediate piles up.

NOTE ON CALIBRATION: unlike the ethyl-acetate simulator, the shifts below are
AQUEOUS LITERATURE VALUES, not values fitted from your instrument. Refit them
against your first real spectrum (analyze_real_nmr.py reports the fitted
centres) and paste them back in here.

Requirements: numpy, scipy, matplotlib

>>> Just press "Run" in your IDE. Everything is controlled by CONFIG below. <<<

  * CONFIG["mode"] = "single" -> one spectrum at CONFIG["conversion"], with an
    annotated PNG and two CSV reports:
        sim_NMR_<X>pct.png
        sim_NMR_<X>pct_groups.csv   (per proton group: delta, J, AREA, moles)
        sim_NMR_<X>pct_species.csv  (per compound: moles, mass, mole fraction)
    plus one ZOOM per window in CONFIG["zoom_regions"]. Inside a 0.1 ppm
    cluster the total trace is one lump, so each zoom re-draws the SAME
    spectrum with every contributing multiplet as its own curve + stick
    spectrum -- which triplet from which compound sits under which shoulder:
        sim_NMR_<X>pct_zoom_<region>.png
        sim_NMR_<X>pct_zoom_<region>_profile.csv     (ppm grid: total, and one
                                                      column per component)
        sim_NMR_<X>pct_zoom_<region>_components.csv  (one row per multiplet
                                                      LINE: position, binomial
                                                      intensity, area, %)
  * CONFIG["mode"] = "stack"  -> the 3-panel figure, plus the same zoom set at
    every stacked conversion:
        sim_80MHz_NMR.png  and  sim_NMR_<X>pct_zoom_<region>.png (+ CSVs)

ONLY SPECIES THAT EXIST ARE ANNOTATED. At 0% conversion there is no EGMA, EG
or AcOH, and at 100% there is no EGDA or EGMA; those leader lines are not
drawn at all, because a label pointing at an absent peak is a lie about the
spectrum. The test is CONFIG["label_threshold_frac"] of the tallest line.
"""

import csv
import os
from math import comb

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.optimize import brentq

# ===========================================================================
#  CONFIG  -- edit these; everything below is driven by this dictionary
# ===========================================================================
CONFIG = {
    # ---- WHAT TO RUN -----------------------------------------------------
    # "single" -> one spectrum at "conversion" (+ CSV report)
    # "stack"  -> the 0/50/90% 3-panel figure + the backbone zoom
    "mode":           "single",
    "conversion":     100,       # percent (50) OR fraction (0.5) of ESTER GROUPS
    "k2_over_k1":     0.5,      # 0.5 = statistical (two independent esters);
                                # <0.5 -> EGMA piles up; >0.5 -> it barely does
    "n0_diester_mol": 1.0,      # initial moles of EGDA -> absolute moles in CSV
    "write_csv":      True,
    "write_zoom":     True,     # the per-region zoom PNG + CSVs

    # ---- OUTPUT ----------------------------------------------------------
    "output_dir":    r"results\simulation",
    "single_output": None,      # PNG path; None -> auto "sim_NMR_<X>pct.png"
    "dpi":           160,

    # ---- ZOOM PANELS -----------------------------------------------------
    # The windows where signals overlap. Each is replotted on its own with
    # every contributing multiplet as a separate curve, so the individual
    # contributions under a merged envelope stay readable.
    #   (key, title fragment, ppm_lo, ppm_hi)
    "zoom_regions": [
        ("backbone", "backbone O–CH$_2$", 3.40, 4.70),
        ("acetyl",   "acetyl CH$_3$",     1.90, 2.35),
    ],
    "zoom_min_area_frac": 0.005,  # skip a component worth <0.5% of the window's
                                  # area - it is not resolvable there anyway
    "zoom_csv_max_rows":  4000,   # the profile CSV is strided down to this
    # A signal below this fraction of the tallest line is NOT annotated. This is
    # what keeps EGDA/EGMA off the 100% figure and EGMA/EG/AcOH off the 0% one.
    "label_threshold_frac": 0.005,

    # ---- INSTRUMENT / LINESHAPE ------------------------------------------
    "spectrometer_mhz": 80.168,  # 1 ppm = this many Hz
    "J_hz":             4.7,     # EGMA vicinal 3J(H,H) across the backbone
    "fwhm_sharp_hz":    1.5,     # carbon-bound (sharp) line width
    "fwhm_broad_hz":    30.0,    # exchangeable (broad) line width
    "ppm_window":       (0.0, 12.0),
    "n_points":         60_000,

    # ---- CONDITIONS ------------------------------------------------------
    "temperature_C":  70.0,     # sets where the water resonance sits
    # arbitrary molar units, relative to 1.0 EGDA. ~100 matches a real dilute
    # aqueous run (0.4 M ester in water); lowering it exaggerates how far the
    # exchange peak drifts downfield as acetic acid builds up.
    "initial_water":  100.0,

    # ---- STACK-MODE conversions (ester-group conversion + panel label) ---
    "stack_conversions": [(0.0, "0% conversion (start)"),
                          (0.5, "~50% of ester groups hydrolysed"),
                          (0.9, "~90% of ester groups hydrolysed")],
}

# ---- constants derived from CONFIG ----------------------------------------
SPEC_MHZ = CONFIG["spectrometer_mhz"]
HZ_PER_PPM = SPEC_MHZ
J_HZ = CONFIG["J_hz"]
PPM_MIN, PPM_MAX = CONFIG["ppm_window"]
N_POINTS = CONFIG["n_points"]
FWHM_SHARP_HZ = CONFIG["fwhm_sharp_hz"]
FWHM_BROAD_HZ = CONFIG["fwhm_broad_hz"]
W0_WATER = CONFIG["initial_water"]
CONVERSIONS = CONFIG["stack_conversions"]

# Colours: diester -> monoester -> glycol, plus the acid.
PANEL_COLORS = ["#1b3a5c", "#2a7f62", "#a23b2e"]
C_EGDA, C_EGMA, C_EG, C_ACID = "#1b3a5c", "#2a7f62", "#a23b2e", "#8c5a2b"


def water_shift(temperature_C):
    """delta(H2O) / ppm ~ 5.051 - 0.0111 * T(degC)  (empirical, aqueous)."""
    return 5.051 - 0.0111 * float(temperature_C)


# ----------------------------------------------------------------------------
# 2. SPECTRAL DATABASE
# ----------------------------------------------------------------------------
# Each carbon-bound group:
#   (label, delta_ppm, n_coupling_partners, n_protons, species_key, color)
# n_coupling_partners is the number of EQUIVALENT vicinal H on the adjacent
# carbon -> sets the multiplicity (0 = singlet, 2 = triplet, ...).
#
# EGDA and EG are symmetric: all four backbone protons are equivalent AND have
# no inequivalent vicinal neighbour, so each gives ONE SINGLET. EGMA is
# desymmetrised, so its two CH2 groups are inequivalent and couple to each other
# -> a pair of 1:2:1 triplets. That asymmetry is the fingerprint of the
# intermediate and the reason the backbone region can resolve all three species.
CARBON_BOUND_GROUPS = [
    # ethylene glycol diacetate  (species key 'D')
    ("EGDA O–CH$_2$ (s)",      4.335, 0, 4, "D", C_EGDA),
    ("EGDA CH$_3$ (s)",        2.140, 0, 6, "D", C_EGDA),
    # ethylene glycol monoacetate  (species key 'M')
    ("EGMA CH$_2$–OC(O) (t)",  4.245, 2, 2, "M", C_EGMA),
    ("EGMA CH$_2$–OH (t)",     3.780, 2, 2, "M", C_EGMA),
    ("EGMA CH$_3$ (s)",        2.125, 0, 3, "M", C_EGMA),
    # ethylene glycol  (species key 'G')
    ("EG O–CH$_2$ (s)",        3.660, 0, 4, "G", C_EG),
    # acetic acid  (species key 'aa')
    ("AcOH CH$_3$ (s)",        2.080, 0, 3, "aa", C_ACID),
]

# Exchangeable protons pooled into one fast-exchange peak:
#   (species_key, n_protons_per_molecule, intrinsic_shift_ppm)
# Water's entry is replaced at run time by the temperature-dependent shift.
EXCHANGEABLE = [
    ("w",  2, None),    # water -> water_shift(temperature)
    ("M",  1, 5.00),    # EGMA  O-H
    ("G",  2, 5.00),    # EG    O-H (two of them)
    ("aa", 1, 11.40),   # acetic acid COO-H (deshielded -> drags the average down)
]

# Per-compound metadata used for the CSV/mole report.
#   key: (name, formula, role, molar_mass, carbon_bound_H, exchangeable_H)
SPECIES_INFO = {
    "D":  ("ethylene glycol diacetate",   "C6H10O4", "reactant",     146.14, 10, 0),
    "M":  ("ethylene glycol monoacetate", "C4H8O3",  "intermediate", 104.10, 7, 1),
    "G":  ("ethylene glycol",             "C2H6O2",  "product",       62.07, 4, 2),
    "aa": ("acetic acid",                 "C2H4O2",  "product",       60.05, 3, 1),
    "w":  ("water",                       "H2O",     "reactant",      18.02, 0, 2),
}


# ----------------------------------------------------------------------------
# 3. PHYSICS / LINESHAPE HELPERS
# ----------------------------------------------------------------------------
def lorentzian(x, x0, fwhm_ppm, amp):
    """Lorentzian centred at x0 with full-width-at-half-maximum fwhm_ppm.
    Peak height = amp; integrated area = amp * (fwhm_ppm/2) * pi."""
    g = fwhm_ppm / 2.0
    return amp * g ** 2 / ((x - x0) ** 2 + g ** 2)


def first_order_multiplet(center_ppm, n_partners, J_hz, spec_mhz):
    """Return [(ppm, relative_intensity), ...] for a first-order multiplet.

    A spin coupled to n equivalent spin-1/2 partners splits into n+1 lines,
    spaced by J, with intensities given by the binomial coefficients C(n,k).
    The line at index k corresponds to the partner group having total
    z-projection M = n/2 - k, hence offset J*M from the centre."""
    weights = [comb(n_partners, k) for k in range(n_partners + 1)]
    norm = float(sum(weights))
    J_ppm = J_hz / spec_mhz
    return [(center_ppm + (n_partners / 2.0 - k) * J_ppm, wk / norm)
            for k, wk in enumerate(weights)]


MULT_NAMES = {0: "singlet", 1: "doublet", 2: "triplet", 3: "quartet",
              4: "quintet", 5: "sextet", 6: "septet"}


def multiplicity(n_partners):
    """'triplet' etc. for a spin coupled to n equivalent spin-1/2 partners."""
    return MULT_NAMES.get(n_partners, f"{n_partners + 1}-let")


def intensity_ratio(n_partners):
    """'1:2:1' -- the binomial line intensities within the multiplet."""
    return ":".join(str(comb(n_partners, k)) for k in range(n_partners + 1))


# ----------------------------------------------------------------------------
# 3b. CONSECUTIVE-REACTION SPECIATION
# ----------------------------------------------------------------------------
def fractions_at_tau(tau, r):
    """Glycol-pool fractions at dimensionless time tau = k1*t, with r = k2/k1.

    Standard A->B->C solution starting from pure A, including the r = 1 limit."""
    tau = float(tau)
    xD = np.exp(-tau)
    if abs(r - 1.0) < 1e-9:
        xM = tau * np.exp(-tau)
    else:
        xM = (np.exp(-tau) - np.exp(-r * tau)) / (r - 1.0)
    return float(xD), float(xM), float(max(1.0 - xD - xM, 0.0))


def ester_conversion_at_tau(tau, r):
    """Fraction of ALL ester groups hydrolysed (EGDA has 2, EGMA has 1)."""
    xD, xM, xG = fractions_at_tau(tau, r)
    return 0.5 * (xM + 2.0 * xG)


def speciation_at_conversion(X, r=0.5):
    """Glycol-pool composition at overall ester-group conversion X.

    Inverts X(tau) numerically. For the statistical case r = 0.5 this reduces
    exactly to the binomial result x = (p^2, 2p(1-p), (1-p)^2) with p = 1 - X."""
    X = float(np.clip(X, 0.0, 1.0 - 1e-9))
    if X <= 0.0:
        return 1.0, 0.0, 0.0
    upper = 1.0
    while ester_conversion_at_tau(upper, r) < X and upper < 1e6:
        upper *= 2.0
    tau = brentq(lambda t: ester_conversion_at_tau(t, r) - X, 0.0, upper,
                 xtol=1e-12)
    return fractions_at_tau(tau, r)


def concentrations_at(X, cfg=CONFIG):
    """Species amounts (arbitrary molar units, EGDA = 1 at t = 0) at conversion X.

    Water is consumed one-for-one with every ester group hydrolysed, and one
    acetic acid is released per hydrolysed ester group."""
    xD, xM, xG = speciation_at_conversion(X, cfg["k2_over_k1"])
    n_acid = xM + 2.0 * xG
    return {"D": xD, "M": xM, "G": xG, "aa": n_acid,
            "w": max(cfg["initial_water"] - n_acid, 0.0)}


def averaged_exchange_peak(conc, cfg=CONFIG):
    """Fast-exchange limit: one line at the population-weighted-average shift.
    Returns (delta_ppm, total_protons) or None if no exchangeable H present."""
    num = tot = 0.0
    for key, nH, shift in EXCHANGEABLE:
        if shift is None:
            shift = water_shift(cfg["temperature_C"])
        p = nH * max(conc[key], 0.0)
        num += p * shift
        tot += p
    if tot <= 1e-9:
        return None
    return num / tot, tot


def build_spectrum(ppm, conc, include_exchange=True, cfg=CONFIG):
    """Sum all carbon-bound multiplets (plus, optionally, the exchange peak).

    Convention:
      * sharp-line HEIGHT proportional to (#protons x concentration)
      * broad exchange peak AREA matched to the sharp lines (height scaled by
        the linewidth ratio, so equal area -> lower height).

    Set ``include_exchange=False`` to omit the pooled water/OH/COOH peak, which
    is useful when overlaying with real data."""
    fwhm_sharp = FWHM_SHARP_HZ / HZ_PER_PPM
    fwhm_broad = FWHM_BROAD_HZ / HZ_PER_PPM

    y = np.zeros_like(ppm)
    for _, delta, n_part, nH, key, _ in CARBON_BOUND_GROUPS:
        amp_group = nH * max(conc[key], 0.0)
        for p, rel in first_order_multiplet(delta, n_part, J_HZ, SPEC_MHZ):
            y += lorentzian(ppm, p, fwhm_sharp, rel * amp_group)

    if include_exchange:
        ex = averaged_exchange_peak(conc, cfg)
        if ex is not None:
            delta_ex, tot_H = ex
            y += lorentzian(ppm, delta_ex, fwhm_broad,
                            tot_H * (FWHM_SHARP_HZ / FWHM_BROAD_HZ))
    return y


def group_lineshape(ppm, delta, n_part, conc, nH):
    """Lineshape of a single carbon-bound group on the ppm grid (same model as
    build_spectrum), so its integral is literally the area under that peak."""
    fwhm_sharp = FWHM_SHARP_HZ / HZ_PER_PPM
    y = np.zeros_like(ppm)
    for p, rel in first_order_multiplet(delta, n_part, J_HZ, SPEC_MHZ):
        y += lorentzian(ppm, p, fwhm_sharp, rel * nH * conc)
    return y


def exchange_lineshape(ppm, conc, cfg=CONFIG):
    """Lineshape of the pooled exchangeable peak, or None if there is none.

    Broken out of build_spectrum because the zoom panels have to draw it as its
    own curve: at 70 degC it lands at 4.28 ppm, right on the backbone cluster,
    and the whole point of the zoom is to show what is underneath it."""
    ex = averaged_exchange_peak(conc, cfg)
    if ex is None:
        return None
    delta_ex, tot_H = ex
    y = lorentzian(ppm, delta_ex, FWHM_BROAD_HZ / HZ_PER_PPM,
                   tot_H * (FWHM_SHARP_HZ / FWHM_BROAD_HZ))
    return delta_ex, tot_H, y


def area_under_curve(ppm, y):
    return float(np.trapezoid(y, ppm))


def visible_groups(conc, cfg=CONFIG):
    """The carbon-bound groups that actually HAVE a peak at this composition.

    At 0% conversion EGMA, EG and AcOH do not exist; at 100% EGDA and EGMA do
    not. Their entries stay in CARBON_BOUND_GROUPS (the CSV still reports them
    at zero) but nothing may be drawn or annotated for them. The threshold also
    drops a species present at a fraction of a percent, which is below the
    detection floor of a real 80 MHz spectrum anyway.

    The test is per SPECIES, not per group: it is the compound that is present
    or absent. Judging each group on its own amplitude would label EGMA's
    3-proton acetyl singlet while dropping its 2-proton backbone triplets at
    the same concentration, which is a statement about proton counts dressed up
    as a statement about what is in the flask."""
    amps = [(g, g[3] * max(conc[g[4]], 0.0)) for g in CARBON_BOUND_GROUPS]
    tallest = max((a for _, a in amps), default=0.0)
    if tallest <= 0.0:
        return []
    floor = cfg["label_threshold_frac"] * tallest
    seen = {key for (_, _, _, _, key, _), a in amps if a >= floor}
    return [g for g, _ in amps if g[4] in seen]


def region_components(ppm, conc, lo, hi, cfg=CONFIG):
    """Decompose the window [lo, hi] into the separate signals that build it.

    Returns one record per contributing signal, each carrying its OWN lineshape
    on the full ppm grid plus the individual multiplet lines. Summing the
    records reproduces the total trace exactly -- that is what makes the zoom an
    honest decomposition rather than a redrawing."""
    mask = (ppm >= lo) & (ppm <= hi)
    comps = []
    for label, delta, n_part, nH, key, col in visible_groups(conc, cfg):
        c = max(conc[key], 0.0)
        y = group_lineshape(ppm, delta, n_part, c, nH)
        amp = nH * c
        comps.append({
            "label": label, "plain": plain_label(label), "is_exchange": False,
            "species": SPECIES_INFO[key][0], "species_key": key,
            "color": col, "delta": delta, "n_part": n_part, "nH": nH,
            "conc": c, "multiplicity": multiplicity(n_part),
            "ratio": intensity_ratio(n_part),
            "J_Hz": J_HZ if n_part > 0 else 0.0,
            "fwhm_hz": FWHM_SHARP_HZ,
            "lines": [(p, rel, rel * amp)
                      for p, rel in first_order_multiplet(delta, n_part, J_HZ,
                                                          SPEC_MHZ)],
            "y": y, "area": area_under_curve(ppm[mask], y[mask]),
        })

    ex = exchange_lineshape(ppm, conc, cfg)
    if ex is not None:
        delta_ex, tot_H, y_ex = ex
        height = tot_H * (FWHM_SHARP_HZ / FWHM_BROAD_HZ)
        comps.append({
            "label": "exchangeable H$_2$O/OH/COOH",
            "plain": "exchangeable pool (H2O/OH/COOH)", "is_exchange": True,
            "species": ", ".join(SPECIES_INFO[k][0] for k, _, _ in EXCHANGEABLE
                                 if conc.get(k, 0.0) > 1e-9),
            "species_key": "ex", "color": "#8a8a8a", "delta": delta_ex,
            "n_part": 0, "nH": "", "conc": "",
            "multiplicity": "broad singlet (fast exchange)", "ratio": "1",
            "J_Hz": 0.0, "fwhm_hz": FWHM_BROAD_HZ,
            "lines": [(delta_ex, 1.0, height)],
            "y": y_ex, "area": area_under_curve(ppm[mask], y_ex[mask]),
        })

    # Membership rule: a signal belongs to this window if the compound is
    # present (visible_groups already decided that) and its CENTRE is in the
    # window. A signal centred outside only earns a place if its tail actually
    # puts area here. Judging in-window peaks on area instead would drop a
    # species the full-range figure labels -- and the area threshold is taken
    # against the CARBON-BOUND total, because measured against the whole window
    # the exchange pool is ~98% of it and every real signal looks like rounding
    # error. The pool itself is never dropped.
    cb = sum(c["area"] for c in comps if not c["is_exchange"])
    keep = [c for c in comps
            if c["is_exchange"] or cb <= 0 or lo <= c["delta"] <= hi
            or c["area"] >= cfg["zoom_min_area_frac"] * cb]
    # Two denominators, both reported. The share of the WHOLE window says how
    # badly the exchange peak swamps the region; the share of the CARBON-BOUND
    # area is the number that matters for integration, because that is the pool
    # analyze_real_nmr.py actually splits between species.
    all_total = sum(c["area"] for c in keep) or 1.0
    cb_total = sum(c["area"] for c in keep if not c["is_exchange"]) or 1.0
    for c in keep:
        c["area_percent"] = 100.0 * c["area"] / all_total
        c["area_pct_cb"] = 100.0 * c["area"] / cb_total
    return sorted(keep, key=lambda c: -c["delta"])


# ----------------------------------------------------------------------------
# 4. CONSOLE REPORT
# ----------------------------------------------------------------------------
def plain_label(label):
    """Matplotlib mathtext label -> plain ASCII, for the console and the CSVs.

    rsplit, not split: 'EGMA CH2-OC(O) (t)' has a bracket in the group name
    itself, so splitting on the FIRST ' (' truncates it to 'EGMA CH2-OC'."""
    return (label.replace("$_2$", "2").replace("$_3$", "3")
            .replace("–", "-").rsplit(" (", 1)[0])


def strip_pattern(label):
    """Drop the trailing ' (s)'/' (t)' but keep the mathtext, for plot labels."""
    return label.rsplit(" (", 1)[0]


def print_peak_table(cfg=CONFIG):
    print(f"\nPredicted carbon-bound 1H resonances "
          f"({SPEC_MHZ:.0f} MHz, 1 ppm = {HZ_PER_PPM:.1f} Hz):")
    print("-" * 78)
    print(f"{'group':<24}{'delta/ppm':>10}{'offset/Hz':>11}"
          f"{'pattern':>11}{'lines (rel int)':>20}")
    print("-" * 78)
    for label, delta, n_part, nH, key, _ in CARBON_BOUND_GROUPS:
        print(f"{plain_label(label):<24}{delta:>10.3f}{delta*HZ_PER_PPM:>11.1f}"
              f"{multiplicity(n_part):>11}{intensity_ratio(n_part):>20}")
    print("-" * 78)

    dw = water_shift(cfg["temperature_C"])
    print(f"  water resonance at {cfg['temperature_C']:.0f} C -> {dw:.2f} ppm "
          f"(5.051 - 0.0111*T)")
    nearest = min(CARBON_BOUND_GROUPS, key=lambda g: abs(g[1] - dw))
    gap_hz = abs(nearest[1] - dw) * HZ_PER_PPM
    flag = "  <-- COLLISION, cross-check with the acetyl handle" if gap_hz < 15 else ""
    print(f"  nearest carbon-bound signal: {plain_label(nearest[0])} at "
          f"{nearest[1]:.3f} ppm ({gap_hz:.0f} Hz away){flag}")
    print("-" * 78)
    for X, label in CONVERSIONS:
        xD, xM, xG = speciation_at_conversion(X, cfg["k2_over_k1"])
        ex = averaged_exchange_peak(concentrations_at(X, cfg), cfg)
        pos = f"{ex[0]:.2f} ppm" if ex else "n/a"
        print(f"  {label:<38} x(EGDA)={xD*100:5.1f}%  x(EGMA)={xM*100:5.1f}%  "
              f"x(EG)={xG*100:5.1f}%   exchange pool @ {pos}")
    print()


# ----------------------------------------------------------------------------
# 5. PLOTTING
# ----------------------------------------------------------------------------
def _label_items(conc, cfg=CONFIG):
    """(delta, text, colour) for every signal PRESENT at this composition.

    Built from the composition, not hard-coded: that is what stops a 100%
    figure from carrying leader lines to EGDA and EGMA peaks that are not
    there. The exchangeable pool is included in the same list so it takes part
    in the same collision-avoidance pass instead of being dropped on top of
    whatever happens to sit at its shift."""
    items = []
    for label, delta, _, _, _, col in visible_groups(conc, cfg):
        head, _sp, tail = label.partition(" ")     # "EGDA" / "O-CH2 (s)"
        items.append((delta, f"{head}\n{tail}" if tail else head, col))
    ex = averaged_exchange_peak(conc, cfg)
    if ex is not None:
        items.append((ex[0],
                      f"exchangeable\nH$_2$O/OH/COOH ({ex[0]:.2f} ppm)",
                      "#6b6b6b"))
    return sorted(items, key=lambda it: -it[0])


def _spread(values, min_sep, lo, hi):
    """Push descending values apart to at least min_sep, keeping them in [lo,hi].

    Order is preserved, and that is the whole trick: if the labels stay in the
    same order as the peaks they point at, the leader lines cannot cross."""
    v = [float(min(max(x, lo), hi)) for x in values]
    n = len(v)
    if n < 2:
        return v
    if (n - 1) * min_sep > (hi - lo):        # too many labels - pack them tight
        min_sep = (hi - lo) / (n - 1)
    for i in range(1, n):                    # walk down, opening every gap
        v[i] = min(v[i], v[i - 1] - min_sep)
    v[-1] = max(v[-1], lo)
    for i in range(n - 2, -1, -1):           # walk back up, back inside the axis
        v[i] = max(v[i], v[i + 1] + min_sep)
    return v


def _draw_labels(ax, ppm, y, items, scale, band=(1.05, 1.27), fontsize=8.2,
                 min_sep=0.95, x_range=None, anchor_ys=None):
    """Annotate signals in a clear band ABOVE the trace.

    Two things went wrong with fixed label positions: text blocks landed on top
    of each other around the water peak, and leader lines ran through the data.
    Here the labels live above every peak (band is in units of the data
    maximum), are pushed apart horizontally, and alternate between two rows, so
    neighbouring signals 0.09 ppm apart still get separately readable labels.
    Each leader starts at the top of its own peak, so a shoulder buried under
    the exchange line is still traceable. Pass anchor_ys to attach the leaders
    to per-signal curves instead of the shared trace -- the zoom panels do
    that, so each label points at the component it names."""
    if not items:
        return
    x_a, x_b = ax.get_xlim()                      # reversed axis: left = high ppm
    hi_x, lo_x = max(x_a, x_b), min(x_a, x_b)
    if x_range is None:
        x_range = (lo_x + 0.35, hi_x - 0.35)
    xs = _spread([d for d, _, _ in items], min_sep, x_range[0], x_range[1])
    for i, ((delta, text, col), tx) in enumerate(zip(items, xs)):
        ty = (band[1] if i % 2 == 0 else band[0]) * scale
        y_at = (float(anchor_ys[i]) if anchor_ys is not None
                else float(np.interp(delta, ppm, y)))
        ax.annotate(text, xy=(delta, min(y_at + 0.02 * scale, scale)),
                    xytext=(tx, ty), ha="center", va="bottom",
                    fontsize=fontsize, color=col,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.7,
                                    alpha=0.55, shrinkA=1.5, shrinkB=1.5))


def _footnote(cfg):
    return (f"Aqueous literature shifts (J = {J_HZ:.1f} Hz), "
            f"{SPEC_MHZ:.1f} MHz. EGDA and EG are SYMMETRIC → one singlet each "
            f"(4.34 / 3.66); only the intermediate EGMA is desymmetrised → two "
            f"1:2:1 triplets (4.25 / 3.78). The three acetyl singlets at "
            f"2.14/2.13/2.08 are only ~1-5 Hz apart, so EGDA and EGMA acetyl "
            f"are pooled and only the ESTER-vs-ACID split is measurable. Water "
            f"sits at {water_shift(cfg['temperature_C']):.2f} ppm at "
            f"{cfg['temperature_C']:.0f} °C and moves upfield ~0.011 ppm/°C.")


def plot_stacked(ppm, savepath, cfg=CONFIG):
    """3-panel stacked spectrum at the configured conversions.

    Every panel is annotated with ITS OWN composition, so the 0% panel carries
    no EGMA/EG/AcOH labels and a 100% panel would carry no EGDA/EGMA ones. The
    y-scale is shared across panels so peak heights stay comparable."""
    concs = [concentrations_at(X, cfg) for X, _ in CONVERSIONS]
    specs = [build_spectrum(ppm, c, cfg=cfg) for c in concs]
    ytop = max(s.max() for s in specs)
    ymax = ytop * 1.48                       # headroom for the label band
    xmin, xmax = 0.6, 12.2

    fig = plt.figure(figsize=(13, 10.0))
    gs = gridspec.GridSpec(3, 1, hspace=0.12)

    for i, ((X, label), conc, y, c) in enumerate(zip(CONVERSIONS, concs, specs,
                                                     PANEL_COLORS)):
        ax = fig.add_subplot(gs[i])
        ax.plot(ppm, y, color=c, lw=1.1)
        ax.fill_between(ppm, 0, y, color=c, alpha=0.10)
        ax.set_xlim(xmax, xmin)
        ax.set_ylim(-0.04 * ymax, ymax)
        ax.set_yticks([])
        xD, xM, xG = speciation_at_conversion(X, cfg["k2_over_k1"])
        ax.text(0.012, 0.60,
                f"{label}\nEGDA {xD*100:.0f}%  EGMA {xM*100:.0f}%  EG {xG*100:.0f}%",
                transform=ax.transAxes, fontsize=11, fontweight="bold", color=c)
        if i < len(CONVERSIONS) - 1:
            ax.set_xticklabels([])
        for spine in ("left", "right", "top"):
            ax.spines[spine].set_visible(False)

        _draw_labels(ax, ppm, y, _label_items(conc, cfg), ytop,
                     band=(1.02, 1.22), fontsize=7.4, min_sep=0.95,
                     x_range=(xmin + 0.35, xmax - 3.0))

    ax.set_xlabel(r"$\delta$  /  ppm   ($^{1}$H, 80 MHz)", fontsize=12)
    fig.suptitle("Simulated $^{1}$H NMR (80 MHz) — hydrolysis of ethylene glycol "
                 "diacetate\nEGDA $\\rightarrow$ EGMA $\\rightarrow$ ethylene "
                 "glycol  (+ 2 AcOH)", fontsize=13.5, fontweight="bold", y=0.98)
    fig.text(0.5, 0.005, _footnote(cfg), ha="center", fontsize=8.0, color="#444",
             wrap=True)
    fig.savefig(savepath, dpi=cfg["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {savepath}")


def plot_region_zoom(ppm, X, title, lo, hi, savepath, cfg=CONFIG):
    """Zoom on one window with every contributing multiplet drawn separately.

    The full-range figure can only show the SUM: at 80 MHz the EGDA singlet
    (4.335) and the EGMA ester triplet (4.245) are 7 Hz apart and merge into one
    lump, and the exchange peak sits on top of both. Here each signal is drawn
    as its own dashed curve and its own stick spectrum, so the binomial
    structure (which line of which 1:2:1 triplet belongs to which compound) and
    the amount of overlap are both directly readable off the figure."""
    conc = concentrations_at(X, cfg)
    comps = region_components(ppm, conc, lo, hi, cfg)
    y_all = build_spectrum(ppm, conc, cfg=cfg)
    y_dry = build_spectrum(ppm, conc, include_exchange=False, cfg=cfg)
    mask = (ppm >= lo) & (ppm <= hi)
    # Scale to the carbon-bound signals: the pooled exchange peak is several
    # times taller and would flatten everything else if it set the scale.
    top = float(y_dry[mask].max()) if np.any(mask) and y_dry[mask].max() > 0 else 1.0

    fig, ax = plt.subplots(figsize=(11.0, 6.4))
    ax.set_xlim(hi, lo)
    ax.plot(ppm, y_all, color="#c2c2c2", lw=1.0, zorder=2, alpha=0.85,
            label="total, incl. the exchangeable pool (runs off scale)")
    ax.plot(ppm, y_dry, color="#2b2b2b", lw=1.7, zorder=4,
            label="total, carbon-bound signals only")

    for comp in comps:
        col = comp["color"]
        if comp["is_exchange"]:
            # Line only, no fill and no stick: it is 3-10x taller than the
            # sharp lines here, and shading it would bury everything the zoom
            # exists to show. The curve still marks where it cuts through.
            ax.plot(ppm, comp["y"], color=col, lw=1.1, ls=(0, (6, 3)),
                    alpha=0.9, zorder=3)
            continue
        ax.fill_between(ppm, 0, comp["y"], color=col, alpha=0.16, zorder=1)
        ax.plot(ppm, comp["y"], color=col, lw=1.2, ls="--", alpha=0.95, zorder=3)
        # Stick spectrum: one stem per transition, height = that line's
        # Lorentzian amplitude, so a 1:2:1 triplet reads as 1:2:1 by eye even
        # where the lines have merged into a single bump.
        pos = [p for p, _, _ in comp["lines"]]
        hgt = [h for _, _, h in comp["lines"]]
        ax.vlines(pos, 0, hgt, color=col, lw=1.3, alpha=0.9, zorder=5)
        ax.plot(pos, hgt, "o", ms=3.2, color=col, zorder=6)

    # The pure-water shift, not the pooled average: labelled through the legend
    # rather than inline, because inline text here lands on the very peaks the
    # panel is meant to resolve.
    dw = water_shift(cfg["temperature_C"])
    if lo < dw < hi:
        ax.axvline(dw, color="#c98a3a", lw=1.3, ls=":", zorder=1,
                   label=f"pure H$_2$O shift @ {cfg['temperature_C']:.0f} °C "
                         f"({dw:.2f} ppm)")

    items = [(c["delta"],
              f"{strip_pattern(c['label'])}\n{c['multiplicity']} {c['ratio']}"
              + (f", J = {c['J_Hz']:.1f} Hz" if c["J_Hz"] else "")
              + ("\n" + (f"{c['area_pct_cb']/100:.1f}× the C–H area"
                         if c["is_exchange"] else
                         f"{c['area_pct_cb']:.1f}% of the C–H area")),
              c["color"]) for c in comps]
    _draw_labels(ax, ppm, y_dry, items, top, band=(1.03, 1.24), fontsize=8.0,
                 min_sep=(hi - lo) / max(len(items), 1) * 0.92,
                 x_range=(lo + 0.02 * (hi - lo), hi - 0.02 * (hi - lo)),
                 anchor_ys=[float(np.interp(c["delta"], ppm, c["y"]))
                            for c in comps])

    ax.set_ylim(-0.04 * top, top * 1.56)
    ax.set_yticks([])
    ax.set_xlabel(r"$\delta$ / ppm", fontsize=12)
    xD, xM, xG = speciation_at_conversion(X, cfg["k2_over_k1"])
    ax.set_title(f"Zoom: {title} region ({lo:.2f}–{hi:.2f} ppm) at "
                 f"{X*100:.0f}% ester-group conversion\n"
                 f"EGDA {xD*100:.0f}% · EGMA {xM*100:.0f}% · EG {xG*100:.0f}% "
                 f"— dashed curves and sticks are the individual contributions",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
              fontsize=8.5, frameon=False)
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)

    fig.savefig(savepath, dpi=cfg["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {savepath}")


def plot_single(ppm, X, savepath, cfg=CONFIG):
    """Single fully-annotated spectrum at one chosen ester-group conversion.

    Only the species actually present are labelled -- see _label_items."""
    conc = concentrations_at(X, cfg)
    y = build_spectrum(ppm, conc, cfg=cfg)
    ytop = y.max()
    ymax = ytop * 1.52                       # headroom for the label band
    xmin, xmax = 0.6, 12.2
    pct = X * 100.0
    xD, xM, xG = speciation_at_conversion(X, cfg["k2_over_k1"])
    c = PANEL_COLORS[1]

    fig, ax = plt.subplots(figsize=(13, 6.4))
    ax.plot(ppm, y, color=c, lw=1.1)
    ax.fill_between(ppm, 0, y, color=c, alpha=0.10)
    ax.set_xlim(xmax, xmin)
    ax.set_ylim(-0.04 * ymax, ymax)
    ax.set_yticks([])
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)

    ax.text(0.012, 0.86,
            f"{pct:.0f}% of ester groups hydrolysed\n"
            f"EGDA {xD*100:.0f}%   EGMA {xM*100:.0f}%   EG {xG*100:.0f}%",
            transform=ax.transAxes, fontsize=12, fontweight="bold", color=c)

    _draw_labels(ax, ppm, y, _label_items(conc, cfg), ytop,
                 band=(1.05, 1.27), fontsize=8.2, min_sep=0.95,
                 x_range=(xmin + 0.35, xmax - 3.0))

    ax.set_xlabel(r"$\delta$  /  ppm   ($^{1}$H, 80 MHz)", fontsize=12)
    fig.suptitle(f"Simulated $^{{1}}$H NMR (80 MHz) at {pct:.0f}% ester-group "
                 f"conversion — hydrolysis of ethylene glycol diacetate",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.subplots_adjust(bottom=0.20, top=0.90)
    fig.text(0.5, 0.04, _footnote(cfg), ha="center", fontsize=8.0, color="#444",
             wrap=True)
    fig.savefig(savepath, dpi=cfg["dpi"])
    plt.close(fig)
    print(f"saved: {savepath}")


# ----------------------------------------------------------------------------
# 5b. DATA EXPORT  (per-conversion CSV report)
# ----------------------------------------------------------------------------
def build_group_records(ppm, X, n0_diester, cfg=CONFIG):
    """One dict row per proton group at ester-group conversion X.

    'area_under_curve' is obtained by NUMERICALLY INTEGRATING each group's
    simulated lineshape -- exactly the quantity measured from the experimental
    spectra by analyze_real_nmr.py, so the two are directly comparable."""
    conc = concentrations_at(X, cfg)

    rows = []
    for label, delta, n_part, nH, key, _ in CARBON_BOUND_GROUPS:
        c = max(conc[key], 0.0)
        area = area_under_curve(ppm, group_lineshape(ppm, delta, n_part, c, nH))
        rows.append({
            "group": plain_label(label),
            "species": SPECIES_INFO[key][0],
            "region": "backbone" if delta > 3.0 else "acetyl",
            "delta_ppm": round(delta, 4),
            "multiplicity": multiplicity(n_part),
            "n_lines": n_part + 1,
            "J_Hz": round(J_HZ, 2) if n_part > 0 else 0.0,
            "protons_per_molecule": nH,
            "concentration": round(c, 5),
            "moles": round(c * n0_diester, 6),
            "area_under_curve": area,
        })

    ex = averaged_exchange_peak(conc, cfg)
    if ex is not None:
        delta_ex, tot_H = ex
        fwhm_broad = FWHM_BROAD_HZ / HZ_PER_PPM
        y_ex = lorentzian(ppm, delta_ex, fwhm_broad,
                          tot_H * (FWHM_SHARP_HZ / FWHM_BROAD_HZ))
        contributors = ", ".join(SPECIES_INFO[k][0] for k, _, _ in EXCHANGEABLE
                                 if conc.get(k, 0) > 1e-9)
        rows.append({
            "group": "exchangeable pool (H2O/OH/COOH)",
            "species": contributors, "region": "exchangeable",
            "delta_ppm": round(delta_ex, 4), "multiplicity": "broad singlet",
            "n_lines": 1, "J_Hz": 0.0, "protons_per_molecule": "",
            "concentration": "", "moles": "",
            "area_under_curve": area_under_curve(ppm, y_ex),
        })

    areas = [r["area_under_curve"] for r in rows if r["area_under_curve"] > 1e-9]
    ref = min(areas) if areas else 1.0
    total = sum(areas) if areas else 1.0
    for r in rows:
        a = r["area_under_curve"]
        r["area_norm"] = round(a / ref, 4) if ref > 0 else 0.0
        r["area_percent"] = round(100.0 * a / total, 3) if total > 0 else 0.0
        r["area_under_curve"] = round(a, 6)
    return rows


def build_species_records(X, n0_diester, cfg=CONFIG):
    """Per-compound rows: amounts, masses, mole fractions, proton counts."""
    conc = concentrations_at(X, cfg)
    pool = ("D", "M", "G")
    tot_all = sum(max(conc[k], 0.0) for k in SPECIES_INFO)
    tot_pool = sum(max(conc[k], 0.0) for k in pool)

    rows = []
    for key in ("D", "M", "G", "aa", "w"):
        name, formula, role, mw, cbH, exH = SPECIES_INFO[key]
        c = max(conc[key], 0.0)
        moles = c * n0_diester
        rows.append({
            "species": name, "formula": formula, "role": role,
            "molar_mass_g_per_mol": mw,
            "concentration_rel": round(c, 5),
            "moles": round(moles, 6),
            "mass_g": round(moles * mw, 5),
            "mole_fraction_all": round(c / tot_all, 5) if tot_all > 0 else 0.0,
            "mole_fraction_glycol_pool": (round(c / tot_pool, 5)
                                          if (tot_pool > 0 and key in pool) else ""),
            "carbon_bound_H": cbH, "exchangeable_H": exH,
        })
    return rows


def export_conversion_data(ppm, X, n0_diester, prefix, cfg=CONFIG):
    """Write the per-group and per-species CSVs and print a console summary."""
    groups = build_group_records(ppm, X, n0_diester, cfg)
    species = build_species_records(X, n0_diester, cfg)

    g_path, s_path = f"{prefix}_groups.csv", f"{prefix}_species.csv"
    g_cols = ["group", "species", "region", "delta_ppm", "multiplicity",
              "n_lines", "J_Hz", "protons_per_molecule", "concentration",
              "moles", "area_under_curve", "area_norm", "area_percent"]
    with open(g_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=g_cols)
        w.writeheader()
        w.writerows(groups)

    s_cols = ["species", "formula", "role", "molar_mass_g_per_mol",
              "concentration_rel", "moles", "mass_g", "mole_fraction_all",
              "mole_fraction_glycol_pool", "carbon_bound_H", "exchangeable_H"]
    with open(s_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=s_cols)
        w.writeheader()
        w.writerows(species)

    print(f"\n=== Conversion report @ {X*100:.1f}% of ester groups  "
          f"(initial EGDA = {n0_diester:g} mol, k2/k1 = {cfg['k2_over_k1']:g}) ===")
    print(f"{'species':<32}{'moles':>10}{'mass/g':>10}{'x(all)':>9}{'x(pool)':>10}")
    for r in species:
        xp = r["mole_fraction_glycol_pool"]
        xp = f"{xp:.3f}" if xp != "" else "  -  "
        print(f"{r['species']:<32}{r['moles']:>10.4f}{r['mass_g']:>10.3f}"
              f"{r['mole_fraction_all']:>9.3f}{xp:>10}")
    print(f"saved: {g_path}")
    print(f"saved: {s_path}")


def export_region_data(ppm, X, comps, lo, hi, n0_diester, prefix, cfg=CONFIG):
    """Write the two CSVs behind one zoom panel.

      <prefix>_profile.csv     the plotted curves: ppm, both totals, and one
                               column per component -- the columns sum to the
                               total, so the decomposition is checkable.
      <prefix>_components.csv  one row per multiplet LINE: where it sits, its
                               binomial weight within the multiplet, its height,
                               and how much of the window its group owns.
    """
    conc = concentrations_at(X, cfg)
    mask = (ppm >= lo) & (ppm <= hi)
    x = ppm[mask]
    step = max(1, int(np.ceil(len(x) / max(cfg["zoom_csv_max_rows"], 1))))
    y_all = build_spectrum(ppm, conc, cfg=cfg)[mask]
    y_dry = build_spectrum(ppm, conc, include_exchange=False, cfg=cfg)[mask]

    p_path = f"{prefix}_profile.csv"
    cols = ["ppm", "hz", "total_with_exchange", "total_carbon_bound"]
    cols += [c["plain"] for c in comps]
    with open(p_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        ys = [c["y"][mask] for c in comps]
        for i in range(0, len(x), step):
            w.writerow([f"{x[i]:.5f}", f"{x[i]*HZ_PER_PPM:.2f}",
                        f"{y_all[i]:.6f}", f"{y_dry[i]:.6f}"]
                       + [f"{yc[i]:.6f}" for yc in ys])

    c_path = f"{prefix}_components.csv"
    c_cols = ["group", "species", "multiplicity", "intensity_ratio", "n_lines",
              "J_Hz", "fwhm_Hz", "group_delta_ppm", "line_index", "line_ppm",
              "line_Hz", "line_rel_intensity", "line_height",
              "protons_per_molecule", "concentration", "moles",
              "group_area_in_window", "group_area_pct_of_window",
              "group_area_pct_of_carbon_bound", "window_lo_ppm",
              "window_hi_ppm"]
    with open(c_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=c_cols)
        w.writeheader()
        for c in comps:
            moles = (round(c["conc"] * n0_diester, 6)
                     if isinstance(c["conc"], float) else "")
            for k, (p, rel, h) in enumerate(c["lines"]):
                w.writerow({
                    "group": c["plain"], "species": c["species"],
                    "multiplicity": c["multiplicity"],
                    "intensity_ratio": c["ratio"], "n_lines": len(c["lines"]),
                    "J_Hz": round(c["J_Hz"], 2),
                    "fwhm_Hz": round(c["fwhm_hz"], 2),
                    "group_delta_ppm": round(c["delta"], 4),
                    "line_index": k, "line_ppm": round(p, 4),
                    "line_Hz": round(p * HZ_PER_PPM, 2),
                    "line_rel_intensity": round(rel, 5),
                    "line_height": round(h, 6),
                    "protons_per_molecule": c["nH"],
                    "concentration": (round(c["conc"], 5)
                                      if isinstance(c["conc"], float) else ""),
                    "moles": moles,
                    "group_area_in_window": round(c["area"], 6),
                    "group_area_pct_of_window": round(c["area_percent"], 3),
                    "group_area_pct_of_carbon_bound": round(c["area_pct_cb"], 3),
                    "window_lo_ppm": lo, "window_hi_ppm": hi,
                })

    print(f"saved: {p_path}")
    print(f"saved: {c_path}")


def emit_zooms(ppm, X, prefix, n0_diester, cfg=CONFIG):
    """One zoom PNG (+ CSVs) per window in CONFIG['zoom_regions']."""
    conc = concentrations_at(X, cfg)
    for key, title, lo, hi in cfg["zoom_regions"]:
        comps = region_components(ppm, conc, lo, hi, cfg)
        if not comps:
            print(f"  {key} zoom skipped - no signal in {lo:.2f}-{hi:.2f} ppm "
                  f"at {X*100:.0f}% conversion")
            continue
        z_prefix = f"{prefix}_zoom_{key}"
        plot_region_zoom(ppm, X, title, lo, hi, f"{z_prefix}.png", cfg)
        if cfg["write_csv"]:
            export_region_data(ppm, X, comps, lo, hi, n0_diester, z_prefix, cfg)


# ----------------------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------------------
def main(cfg=CONFIG):
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = cfg["output_dir"]
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(here, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    ppm = np.linspace(PPM_MIN, PPM_MAX, N_POINTS)
    print_peak_table(cfg)

    if cfg["mode"] == "stack":
        plot_stacked(ppm, os.path.join(out_dir, "sim_80MHz_NMR.png"), cfg)
        if cfg["write_zoom"]:
            for X, _ in CONVERSIONS:
                emit_zooms(ppm, X,
                           os.path.join(out_dir, f"sim_NMR_{X*100:.0f}pct"),
                           cfg["n0_diester_mol"], cfg)
        return

    raw = cfg["conversion"]
    X = raw / 100.0 if raw > 1.0 else raw
    if not 0.0 <= X <= 1.0:
        raise SystemExit(f"CONFIG['conversion'] must be in [0,1] or [0,100]%; "
                         f"got {raw}.")
    out = cfg["single_output"] or os.path.join(out_dir, f"sim_NMR_{X*100:.0f}pct.png")
    plot_single(ppm, X, out, cfg)
    prefix = out[:-4] if out.lower().endswith(".png") else out

    if cfg["write_csv"]:
        export_conversion_data(ppm, X, cfg["n0_diester_mol"], prefix, cfg)
    if cfg["write_zoom"]:
        emit_zooms(ppm, X, prefix, cfg["n0_diester_mol"], cfg)


if __name__ == "__main__":
    main()
