#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estimate the EGDA hydrolysis kinetics from an NMR conversion-vs-time CSV.

    EGDA --k1--> EGMA --k2--> EG        (+ one AcOH per step)

Two independent analyses are run on the same file:

1. LUMPED RATE LAW on the ester-group conversion X (the `conversion_X` column,
   which both handles in main.py produce). The usual candidate rate laws are
   fitted and ranked by AICc, exactly as for a single-step hydrolysis. This is
   what you quote as an "apparent k".

2. CONSECUTIVE SCHEME on the full speciation (the `x_EGDA` / `x_EGMA` / `x_EG`
   columns, which only the "backbone" handle produces). All three mole fractions
   are fitted SIMULTANEOUSLY with the analytical A->B->C solution, giving the two
   individual rate constants k1 and k2. Run this whenever you have the backbone
   data -- a lumped k cannot describe a consecutive reaction, and the k1/k2
   ratio is the physically interesting number:

       if the two ester groups of EGDA hydrolyse independently at the same
       intrinsic rate k, then k1 = 2k (two groups to attack) and k2 = k, so

               k1 / k2 = 2      <- the purely statistical expectation

       k1/k2 > 2 means the first hydrolysis is ACTIVATED relative to statistics
       (the intermediate is harder to hydrolyse than a naive count suggests);
       k1/k2 < 2 means EGMA reacts faster than statistics -- the monoester's
       free OH is activating the remaining ester, so EGMA never accumulates.

The initial zero-conversion points produced while a flow cell fills and
stabilises are detected as transport delay. They are shown in the output plot
but are not used to fit the chemical kinetics. Edit CONFIG and press Run.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, curve_fit


# ===========================================================================
# CONFIG -- edit these values, then press Run in your IDE
# ===========================================================================
CONFIG = {
    "input_file": r"results\backbone_conversion\conversion_vs_time_backbone.csv",
    "output_folder": r"results\kinetics_backbone",

    # Leave as None to detect the columns automatically. Supported automatic
    # names include minute/time_min and conversion_X/conversion_percent.
    "time_column": None,
    "conversion_column": None,

    # Speciation columns for the consecutive A->B->C fit. Set
    # fit_consecutive to False to skip it, or leave the names as None to
    # auto-detect x_EGDA / x_EGMA / x_EG. The fit is skipped automatically when
    # the columns are absent (e.g. an "acetyl"-mode master table).
    "fit_consecutive": True,
    "x_egda_column": None,
    "x_egma_column": None,
    "x_eg_column": None,

    # Flow-cell delay / reaction-onset detection. The first run of this many
    # points at or above the threshold marks the start of usable kinetic data.
    # Set manual_onset_minute to a number to override automatic detection.
    "manual_onset_minute": None,
    "onset_threshold_percent": 1.0,
    "onset_consecutive_points": 2,

    # Apparent rate-law candidates for the LUMPED ester-group conversion.
    # "nth_order" estimates the reaction order. Selection uses AICc.
    "models": (
        "zero_order",
        "first_order",
        "second_order",
        "nth_order",
        "first_order_plateau",
    ),
    "preferred_model": "auto",  # or one of the names above

    # Optional initial EGDA concentration for a concentration-based rate
    # constant. None reads the largest c_EGDA_M value from the CSV, if present.
    "initial_concentration_M": None,

    # The saved prediction curve extends this many active-data durations past
    # the detected onset. Set prediction_end_minute to override it directly.
    "prediction_extension_factor": 1.5,
    "prediction_end_minute": None,
    "prediction_target_percent": 90.0,
    "prediction_points": 600,

    "dpi": 180,
}


C_DATA = "#287f67"
C_FIT = "#c43c39"
C_DELAY = "#8a8f98"
C_BAND = "#e8aaa7"
# diester -> monoester -> glycol, matching analyze_real_nmr.py
C_EGDA, C_EGMA, C_EG = "#1b3a5c", "#2a7f62", "#a23b2e"

# Two equivalent ester groups on EGDA -> k1 = 2k, k2 = k if they are independent.
STATISTICAL_K1_OVER_K2 = 2.0


@dataclass
class FitResult:
    name: str
    label: str
    parameter_names: tuple[str, ...]
    parameters: np.ndarray
    covariance: np.ndarray
    sse: float
    rmse: float
    mae: float
    r2: float
    aic: float
    aicc: float
    bic: float
    akaike_weight: float = 0.0


@dataclass
class ConsecutiveFit:
    """Global A->B->C fit to the three measured mole fractions."""
    k1: float
    k2: float
    xD0: float
    xM0: float
    std_errors: dict[str, float]
    r2_overall: float
    r2_per_species: dict[str, float]
    rmse: float
    n_obs: int
    covariance: np.ndarray = field(default_factory=lambda: np.zeros((4, 4)))


# ===========================================================================
# DATA LOADING AND FLOW-DELAY DETECTION
# ===========================================================================
def resolve_config_path(value, *, must_exist=False):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if must_exist and not path.is_file() and path.suffix.lower() != ".csv":
        csv_path = path.with_suffix(".csv")
        if csv_path.is_file():
            path = csv_path
    return path.resolve()


def _choose_column(fieldnames, requested, candidates, kind, required=True):
    if requested is not None:
        if requested not in fieldnames:
            raise ValueError(f"Configured {kind} column {requested!r} was not found")
        return requested
    lower = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    if not required:
        return None
    raise ValueError(f"Could not detect the {kind} column. Available: {fieldnames}")


def _float_or_none(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def load_conversion_csv(path, cfg):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("The input CSV has no header")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    time_col = _choose_column(
        fieldnames, cfg["time_column"],
        ("minute", "time_min", "time_minutes", "time", "t_min"), "time",
    )
    conversion_col = _choose_column(
        fieldnames, cfg["conversion_column"],
        ("conversion_X", "conversion", "conversion_percent", "conversion_%"),
        "conversion",
    )
    is_percent = "percent" in conversion_col.lower() or "%" in conversion_col

    species_cols = None
    if cfg.get("fit_consecutive", True):
        cols = [
            _choose_column(fieldnames, cfg.get("x_egda_column"),
                           ("x_EGDA", "x_egda"), "EGDA fraction", required=False),
            _choose_column(fieldnames, cfg.get("x_egma_column"),
                           ("x_EGMA", "x_egma"), "EGMA fraction", required=False),
            _choose_column(fieldnames, cfg.get("x_eg_column"),
                           ("x_EG", "x_eg"), "EG fraction", required=False),
        ]
        species_cols = tuple(cols) if all(cols) else None

    points = []
    speciation = {}
    concentrations = []
    sample_ids = []
    modes = []
    for row in rows:
        time_value = _float_or_none(row.get(time_col))
        raw_conversion = _float_or_none(row.get(conversion_col))
        if time_value is None or raw_conversion is None:
            continue
        points.append((time_value, raw_conversion / (100.0 if is_percent else 1.0)))

        if species_cols is not None:
            trio = [_float_or_none(row.get(c)) for c in species_cols]
            if all(v is not None for v in trio):
                speciation.setdefault(time_value, []).append(trio)

        concentration = _float_or_none(row.get("c_EGDA_M"))
        if concentration is not None and concentration > 0:
            concentrations.append(concentration)
        if row.get("sample_id"):
            sample_ids.append(row["sample_id"])
        if row.get("mode"):
            modes.append(row["mode"])

    if len(points) < 6:
        raise ValueError("At least six valid time/conversion rows are required")

    # Sort and average duplicate time stamps.
    points.sort(key=lambda item: item[0])
    grouped = {}
    for time_value, conversion in points:
        grouped.setdefault(time_value, []).append(conversion)
    time = np.array(sorted(grouped), dtype=float)
    conversion = np.array([np.mean(grouped[t]) for t in time], dtype=float)

    if np.any(conversion < -0.02) or np.any(conversion > 1.02):
        raise ValueError("Conversion must be a fraction in [0, 1] or a percent column")
    conversion = np.clip(conversion, 0.0, 1.0)

    fractions = None
    if speciation and len(speciation) == len(time):
        fractions = np.array([np.mean(speciation[t], axis=0) for t in time], float)
        # Renormalise: the three measured areas should already sum to 1, but
        # rounding in the CSV can leave a few 1e-5 residuals.
        totals = fractions.sum(axis=1, keepdims=True)
        if np.all(totals > 0.5):
            fractions = np.clip(fractions / totals, 0.0, 1.0)
        else:
            fractions = None

    metadata = {
        "time_column": time_col,
        "conversion_column": conversion_col,
        "species_columns": species_cols,
        "sample_id": sample_ids[0] if sample_ids else path.stem,
        "mode": modes[0] if modes else "",
        "initial_concentration_M": max(concentrations) if concentrations else None,
    }
    return time, conversion, fractions, metadata


def detect_onset(time, conversion, cfg):
    manual = cfg.get("manual_onset_minute")
    if manual is not None:
        candidates = np.flatnonzero(time >= float(manual))
        if candidates.size == 0:
            raise ValueError("manual_onset_minute is later than the final data point")
        return int(candidates[0])

    threshold = float(cfg["onset_threshold_percent"]) / 100.0
    consecutive = max(int(cfg["onset_consecutive_points"]), 1)
    for index in range(0, len(time) - consecutive + 1):
        if np.all(conversion[index:index + consecutive] >= threshold):
            return index
    raise ValueError(
        "No sustained reaction onset was found. Lower onset_threshold_percent "
        "or set manual_onset_minute."
    )


# ===========================================================================
# LUMPED RATE-LAW MODELS  (on the ester-group conversion X)
# ===========================================================================
def model_zero_order(t, x0, k):
    return x0 + k * np.asarray(t, float)


def model_first_order(t, x0, k):
    return 1.0 - (1.0 - x0) * np.exp(-k * np.asarray(t, float))


def model_second_order(t, x0, k):
    t = np.asarray(t, float)
    remaining0 = max(1.0 - x0, 1e-12)
    return 1.0 - remaining0 / (1.0 + k * remaining0 * t)


def model_nth_order(t, x0, k, order):
    """Integrated dX/dt = k(1-X)^order, including the order=1 limit."""
    t = np.asarray(t, float)
    remaining0 = max(1.0 - x0, 1e-12)
    epsilon = order - 1.0
    if abs(epsilon) < 1e-7:
        return 1.0 - remaining0 * np.exp(-k * t)
    base = 1.0 + epsilon * k * remaining0 ** epsilon * t
    remaining = remaining0 * np.exp(-np.log(np.maximum(base, 1e-14)) / epsilon)
    return 1.0 - remaining


def model_first_order_plateau(t, x0, x_infinity, k):
    return x_infinity - (x_infinity - x0) * np.exp(-k * np.asarray(t, float))


MODEL_INFO = {
    "zero_order": ("Zero order", ("X0", "k_app"), model_zero_order),
    "first_order": ("Pseudo-first order", ("X0", "k_app"), model_first_order),
    "second_order": ("Pseudo-second order", ("X0", "k_app"), model_second_order),
    "nth_order": ("General apparent order", ("X0", "k_app", "order"), model_nth_order),
    "first_order_plateau": (
        "First order with equilibrium plateau",
        ("X0", "X_infinity", "k_app"), model_first_order_plateau,
    ),
}


def model_function(name):
    return MODEL_INFO[name][2]


def _fit_once(function, reaction_time, conversion, p0, lower, upper):
    parameters, covariance = curve_fit(
        function,
        reaction_time,
        conversion,
        p0=p0,
        bounds=(lower, upper),
        maxfev=100000,
    )
    prediction = function(reaction_time, *parameters)
    sse = float(np.sum((conversion - prediction) ** 2))
    return parameters, covariance, sse


def fit_model(name, reaction_time, conversion):
    if name not in MODEL_INFO:
        raise ValueError(f"Unknown kinetic model: {name!r}")
    function = model_function(name)
    x0_guess = float(np.clip(conversion[0], 1e-6, 0.3))
    count_for_slope = min(8, conversion.size)
    slope = float(np.polyfit(reaction_time[:count_for_slope],
                             conversion[:count_for_slope], 1)[0])
    k_guess = float(np.clip(slope / max(1.0 - x0_guess, 0.05), 1e-5, 0.1))

    if name == "zero_order":
        attempts = [([x0_guess, max(slope, 1e-5)], [0.0, 0.0], [0.4, 1.0])]
    elif name in ("first_order", "second_order"):
        attempts = [([x0_guess, k_guess], [0.0, 1e-9], [0.4, 1.0])]
    elif name == "first_order_plateau":
        lower_plateau = min(max(float(np.max(conversion)) + 1e-5, 0.5), 0.9999)
        if lower_plateau >= 1.0:
            raise ValueError("The data already reach complete conversion")
        plateau_guess = min(max(lower_plateau + 0.05, 0.9), 0.999)
        attempts = [
            ([x0_guess, plateau_guess, k_guess],
             [0.0, lower_plateau, 1e-9], [0.4, 1.0, 1.0])
        ]
    else:
        # Multi-start avoids the shallow local stationary point exactly at n=1.
        attempts = [
            ([x0_guess, k_guess, order0], [0.0, 1e-9, 0.1], [0.4, 1.0, 3.0])
            for order0 in (0.5, 0.8, 0.95, 1.05, 1.3, 2.0)
        ]

    best = None
    for p0, lower, upper in attempts:
        try:
            candidate = _fit_once(function, reaction_time, conversion,
                                  p0, lower, upper)
        except (RuntimeError, ValueError, FloatingPointError):
            continue
        if best is None or candidate[2] < best[2]:
            best = candidate
    if best is None:
        raise RuntimeError(f"{name} fit did not converge")

    parameters, covariance, sse = best
    prediction = function(reaction_time, *parameters)
    residual = conversion - prediction
    n_obs = conversion.size
    n_parameters = parameters.size
    mse = max(sse / n_obs, np.finfo(float).tiny)
    aic = n_obs * math.log(mse) + 2 * n_parameters
    denominator = n_obs - n_parameters - 1
    aicc = (aic + 2 * n_parameters * (n_parameters + 1) / denominator
            if denominator > 0 else float("inf"))
    bic = n_obs * math.log(mse) + n_parameters * math.log(n_obs)
    ss_total = float(np.sum((conversion - np.mean(conversion)) ** 2))
    r2 = 1.0 - sse / ss_total if ss_total > 0 else float("nan")
    label, parameter_names, _ = MODEL_INFO[name]
    return FitResult(
        name=name,
        label=label,
        parameter_names=parameter_names,
        parameters=np.asarray(parameters, float),
        covariance=np.asarray(covariance, float),
        sse=sse,
        rmse=float(np.sqrt(np.mean(residual ** 2))),
        mae=float(np.mean(np.abs(residual))),
        r2=r2,
        aic=aic,
        aicc=aicc,
        bic=bic,
    )


def fit_all_models(reaction_time, conversion, cfg):
    fits = []
    failures = []
    for name in cfg["models"]:
        try:
            fits.append(fit_model(name, reaction_time, conversion))
        except (RuntimeError, ValueError) as exc:
            failures.append(f"{name}: {exc}")
    if not fits:
        raise RuntimeError("No kinetic model converged (" + "; ".join(failures) + ")")

    minimum_aicc = min(fit.aicc for fit in fits)
    relative = np.array([math.exp(-0.5 * (fit.aicc - minimum_aicc)) for fit in fits])
    relative /= float(np.sum(relative))
    for fit, weight in zip(fits, relative):
        fit.akaike_weight = float(weight)

    preferred = cfg.get("preferred_model", "auto")
    if preferred == "auto":
        best = min(fits, key=lambda fit: fit.aicc)
    else:
        matches = [fit for fit in fits if fit.name == preferred]
        if not matches:
            raise ValueError(f"preferred_model {preferred!r} was not fitted")
        best = matches[0]
    return fits, best, failures


# ===========================================================================
# CONSECUTIVE SCHEME  EGDA --k1--> EGMA --k2--> EG
# ===========================================================================
def consecutive_fractions(t, k1, k2, xD0, xM0):
    """Analytical A->B->C solution, including the degenerate k1 == k2 limit.

    Pseudo-first order in the ester (water is in large excess and its
    concentration is effectively constant), so both steps are exponential."""
    t = np.asarray(t, float)
    xD = xD0 * np.exp(-k1 * t)
    if abs(k2 - k1) < 1e-9 * max(k1, k2, 1e-12):
        xM = xM0 * np.exp(-k1 * t) + xD0 * k1 * t * np.exp(-k1 * t)
    else:
        xM = (xM0 * np.exp(-k2 * t)
              + xD0 * k1 / (k2 - k1) * (np.exp(-k1 * t) - np.exp(-k2 * t)))
    xG = 1.0 - xD - xM
    return xD, xM, np.clip(xG, 0.0, 1.0)


def consecutive_conversion(t, k1, k2, xD0, xM0):
    """Ester-group conversion X = [x_EGMA + 2 x_EG] / 2 from the scheme."""
    xD, xM, xG = consecutive_fractions(t, k1, k2, xD0, xM0)
    return 0.5 * (xM + 2.0 * xG)


def _stacked_consecutive(t_stacked, k1, k2, xD0, u):
    """curve_fit target: the three fractions concatenated into one vector.

    `u` parameterises xM0 = (1 - xD0) * u so that xD0 + xM0 <= 1 is enforced by
    plain box bounds -- the three fractions always stay a valid composition."""
    n = t_stacked.size // 3
    t = t_stacked[:n]
    xD, xM, xG = consecutive_fractions(t, k1, k2, xD0, (1.0 - xD0) * u)
    return np.concatenate([xD, xM, xG])


def fit_consecutive(reaction_time, fractions):
    """Fit k1, k2 and the starting composition to all three fractions at once."""
    if fractions is None or fractions.shape[0] < 5:
        raise ValueError("At least five speciation points are required")
    observed = np.concatenate([fractions[:, 0], fractions[:, 1], fractions[:, 2]])
    t_stacked = np.tile(reaction_time, 3)

    xD0_guess = float(np.clip(fractions[0, 0], 1e-3, 1.0))
    u_guess = float(np.clip(fractions[0, 1] / max(1.0 - xD0_guess, 1e-6), 0.0, 1.0))
    span = max(float(reaction_time[-1] - reaction_time[0]), 1.0)

    best = None
    # Multi-start over the k1/k2 ratio: k1 == k2 is a shallow stationary point,
    # and starting on the wrong side of it can trap the optimiser.
    for scale in (0.5, 1.0, 3.0):
        for ratio in (0.5, 1.0, 2.0, 5.0):
            k2_0 = float(np.clip(scale / span, 1e-6, 1.0))
            p0 = [min(k2_0 * ratio, 0.99), k2_0, xD0_guess, u_guess]
            try:
                params, covariance = curve_fit(
                    _stacked_consecutive, t_stacked, observed, p0=p0,
                    bounds=([1e-9, 1e-9, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]),
                    maxfev=200000)
            except (RuntimeError, ValueError, FloatingPointError):
                continue
            sse = float(np.sum((observed - _stacked_consecutive(t_stacked, *params)) ** 2))
            if best is None or sse < best[2]:
                best = (params, covariance, sse)
    if best is None:
        raise RuntimeError("The consecutive A->B->C fit did not converge")

    params, covariance, sse = best
    k1, k2, xD0, u = (float(v) for v in params)
    xM0 = (1.0 - xD0) * u

    prediction = _stacked_consecutive(t_stacked, *params)
    n_obs = observed.size
    ss_total = float(np.sum((observed - np.mean(observed)) ** 2))
    r2_overall = 1.0 - sse / ss_total if ss_total > 0 else float("nan")

    r2_per = {}
    n = fractions.shape[0]
    for index, key in enumerate(("x_EGDA", "x_EGMA", "x_EG")):
        obs = fractions[:, index]
        pred = prediction[index * n:(index + 1) * n]
        sst = float(np.sum((obs - np.mean(obs)) ** 2))
        r2_per[key] = (1.0 - float(np.sum((obs - pred) ** 2)) / sst
                       if sst > 0 else float("nan"))

    errors = np.sqrt(np.maximum(np.diag(np.asarray(covariance, float)), 0.0))
    # d(xM0)/d(u) = 1 - xD0 ; d(xM0)/d(xD0) = -u  (first-order propagation)
    xM0_error = float(np.hypot((1.0 - xD0) * errors[3], u * errors[2]))
    std_errors = {"k1": float(errors[0]), "k2": float(errors[1]),
                  "xD0": float(errors[2]), "xM0": xM0_error}

    return ConsecutiveFit(
        k1=k1, k2=k2, xD0=xD0, xM0=xM0, std_errors=std_errors,
        r2_overall=r2_overall, r2_per_species=r2_per,
        rmse=float(np.sqrt(sse / n_obs)), n_obs=n_obs,
        covariance=np.asarray(covariance, float),
    )


def egma_maximum(fit):
    """(time, fraction) of the peak EGMA concentration, or (None, None).

    For a clean start (xM0 = 0) the intermediate peaks at
    t_max = ln(k2/k1)/(k2 - k1); with a non-zero start the maximum is found
    numerically on a fine grid."""
    if fit.k1 <= 0 or fit.k2 <= 0:
        return None, None
    if fit.xM0 <= 1e-9 and abs(fit.k2 - fit.k1) > 1e-9:
        t_max = math.log(fit.k2 / fit.k1) / (fit.k2 - fit.k1)
        if t_max <= 0:
            return None, None
    elif fit.xM0 <= 1e-9:
        t_max = 1.0 / fit.k1
    else:
        grid = np.linspace(0.0, 20.0 / min(fit.k1, fit.k2), 20000)
        _, xM, _ = consecutive_fractions(grid, fit.k1, fit.k2, fit.xD0, fit.xM0)
        t_max = float(grid[int(np.argmax(xM))])
    _, xM_max, _ = consecutive_fractions(np.array([t_max]), fit.k1, fit.k2,
                                         fit.xD0, fit.xM0)
    return float(t_max), float(xM_max[0])


# ===========================================================================
# DERIVED KINETIC PARAMETERS AND UNCERTAINTY
# ===========================================================================
def parameter_map(fit):
    values = dict(zip(fit.parameter_names, fit.parameters))
    if "X_infinity" not in values:
        values["X_infinity"] = 1.0
    if "order" not in values:
        values["order"] = {
            "zero_order": 0.0,
            "first_order": 1.0,
            "second_order": 2.0,
            "first_order_plateau": 1.0,
        }[fit.name]
    return values


def time_to_conversion(fit, target):
    values = parameter_map(fit)
    if target <= values["X0"]:
        return 0.0
    if target >= values["X_infinity"] and fit.name != "zero_order":
        return None
    function = model_function(fit.name)

    def objective(time_value):
        return float(function(np.array([time_value]), *fit.parameters)[0] - target)

    upper = 1.0
    while objective(upper) < 0 and upper < 1e7:
        upper *= 2.0
    if objective(upper) < 0:
        return None
    return float(brentq(objective, 0.0, upper))


def model_initial_rate(fit):
    function = model_function(fit.name)
    dt = 1e-4
    return float((function(np.array([dt]), *fit.parameters)[0]
                  - function(np.array([0.0]), *fit.parameters)[0]) / dt)


def prediction_with_ci(fit, reaction_time):
    function = model_function(fit.name)
    reaction_time = np.asarray(reaction_time, float)
    center = np.asarray(function(reaction_time, *fit.parameters), float)
    if (fit.covariance.shape != (fit.parameters.size, fit.parameters.size)
            or not np.all(np.isfinite(fit.covariance))):
        return center, np.full(center.size, np.nan), np.full(center.size, np.nan)

    jacobian = np.empty((reaction_time.size, fit.parameters.size), float)
    for index, value in enumerate(fit.parameters):
        step = max(abs(float(value)) * 1e-5, 1e-7)
        high = fit.parameters.copy()
        low = fit.parameters.copy()
        high[index] += step
        low[index] -= step
        jacobian[:, index] = (
            function(reaction_time, *high) - function(reaction_time, *low)
        ) / (2.0 * step)
    variance = np.einsum("ij,jk,ik->i", jacobian, fit.covariance, jacobian)
    standard_error = np.sqrt(np.maximum(variance, 0.0))
    return center, np.clip(center - 1.96 * standard_error, 0.0, 1.0), \
        np.clip(center + 1.96 * standard_error, 0.0, 1.0)


def build_parameter_rows(best, time, onset_index, initial_M, consecutive):
    values = parameter_map(best)
    standard_errors = np.sqrt(np.maximum(np.diag(best.covariance), 0.0))
    fitted_errors = dict(zip(best.parameter_names, standard_errors))
    rows = []

    def add(section, name, value, unit="", description="", error=None):
        finite_error = (error is not None and np.isfinite(error)
                        and isinstance(value, (int, float)))
        rows.append({
            "section": section,
            "parameter": name,
            "value": value,
            "std_error": error if finite_error else "",
            "ci95_low": value - 1.96 * error if finite_error else "",
            "ci95_high": value + 1.96 * error if finite_error else "",
            "unit": unit,
            "description": description,
        })

    add("selection", "best_lumped_model", best.label,
        description="Lowest AICc unless overridden")
    add("selection", "best_lumped_model_key", best.name)
    add("flow", "flow_delay", time[onset_index] - time[0], "min",
        "Initial transport/stabilisation period excluded from kinetic fitting")
    add("flow", "fit_start_time", time[onset_index], "min")
    add("flow", "excluded_delay_points", onset_index, "points")
    add("fit", "observations_used", len(time) - onset_index, "points")
    add("fit", "X0_at_fit_start", values["X0"], "fraction",
        "Fitted ester-group conversion at reaction-time zero", fitted_errors.get("X0"))
    add("fit", "X_infinity", values["X_infinity"], "fraction",
        "Asymptotic conversion imposed or fitted", fitted_errors.get("X_infinity"))
    add("fit", "apparent_reaction_order", values["order"], "dimensionless",
        "Order in dX/dt = k_app(1-X)^n", fitted_errors.get("order"))
    add("fit", "k_app", values["k_app"], "min^-1",
        "Apparent ester-group conversion rate coefficient", fitted_errors.get("k_app"))

    if initial_M is not None and initial_M > 0 and best.name != "first_order_plateau":
        k_concentration = values["k_app"] / initial_M ** (values["order"] - 1.0)
        add("fit", "k_concentration_based", k_concentration,
            f"M^{1.0 - values['order']:.4g} min^-1",
            "Rate constant for -dC/dt = k C^n using the initial EGDA concentration")
        add("experiment", "initial_EGDA_concentration", initial_M, "M")

    initial_rate = model_initial_rate(best)
    add("derived", "initial_rate", initial_rate, "conversion fraction/min")
    add("derived", "initial_rate_percent", 100.0 * initial_rate, "%/min")
    half_target = values["X0"] + 0.5 * (values["X_infinity"] - values["X0"])
    half_time = time_to_conversion(best, half_target)
    add("derived", "half_time_from_fit_start",
        half_time if half_time is not None else "", "min")
    for percent in (10, 25, 50, 75, 90):
        relative_time = time_to_conversion(best, percent / 100.0)
        add("prediction", f"time_to_{percent}_percent_from_fit_start",
            relative_time if relative_time is not None else "", "min")
        add("prediction", f"experimental_time_at_{percent}_percent",
            time[onset_index] + relative_time if relative_time is not None else "",
            "min")

    add("quality", "R_squared", best.r2)
    add("quality", "RMSE", best.rmse, "conversion fraction")
    add("quality", "MAE", best.mae, "conversion fraction")
    add("quality", "SSE", best.sse)
    add("quality", "AIC", best.aic)
    add("quality", "AICc", best.aicc)
    add("quality", "BIC", best.bic)
    add("quality", "Akaike_weight", best.akaike_weight)

    if consecutive is not None:
        c = consecutive
        add("consecutive", "k1", c.k1, "min^-1",
            "EGDA -> EGMA (first ester group)", c.std_errors["k1"])
        add("consecutive", "k2", c.k2, "min^-1",
            "EGMA -> EG (second ester group)", c.std_errors["k2"])
        ratio = c.k1 / c.k2 if c.k2 > 0 else float("nan")
        add("consecutive", "k1_over_k2", ratio, "dimensionless",
            f"Statistical expectation for two independent equal groups is "
            f"{STATISTICAL_K1_OVER_K2:g}")
        add("consecutive", "k_per_ester_group_from_k1", 0.5 * c.k1, "min^-1",
            "k1/2: intrinsic per-group rate if the two EGDA esters are independent")
        add("consecutive", "x_EGDA_at_fit_start", c.xD0, "fraction", "",
            c.std_errors["xD0"])
        add("consecutive", "x_EGMA_at_fit_start", c.xM0, "fraction", "",
            c.std_errors["xM0"])
        add("consecutive", "half_life_EGDA", math.log(2.0) / c.k1 if c.k1 > 0 else "",
            "min", "ln2/k1")
        add("consecutive", "half_life_EGMA", math.log(2.0) / c.k2 if c.k2 > 0 else "",
            "min", "ln2/k2")
        t_max, x_max = egma_maximum(c)
        add("consecutive", "time_of_max_EGMA", t_max if t_max is not None else "",
            "min", "Measured from the fit start")
        add("consecutive", "max_EGMA_fraction", x_max if x_max is not None else "",
            "fraction", "Peak intermediate accumulation")
        add("consecutive", "R_squared_overall", c.r2_overall)
        for key, value in c.r2_per_species.items():
            add("consecutive", f"R_squared_{key}", value)
        add("consecutive", "RMSE", c.rmse, "mole fraction")
        add("consecutive", "observations_used", c.n_obs, "points",
            "Three mole fractions per time point")
    return rows


# ===========================================================================
# OUTPUT WRITERS AND PLOT
# ===========================================================================
def write_dict_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_model_comparison(path, fits):
    rows = []
    minimum = min(fit.aicc for fit in fits)
    for fit in sorted(fits, key=lambda item: item.aicc):
        values = parameter_map(fit)
        errors = np.sqrt(np.maximum(np.diag(fit.covariance), 0.0))
        error_map = dict(zip(fit.parameter_names, errors))
        rows.append({
            "model": fit.name,
            "description": fit.label,
            "n_parameters": len(fit.parameters),
            "X0": values["X0"],
            "X0_std_error": error_map.get("X0", ""),
            "X_infinity": values["X_infinity"],
            "X_infinity_std_error": error_map.get("X_infinity", ""),
            "k_app_per_min": values["k_app"],
            "k_app_std_error": error_map.get("k_app", ""),
            "reaction_order": values["order"],
            "reaction_order_std_error": error_map.get("order", ""),
            "R_squared": fit.r2,
            "RMSE": fit.rmse,
            "MAE": fit.mae,
            "AIC": fit.aic,
            "AICc": fit.aicc,
            "delta_AICc": fit.aicc - minimum,
            "Akaike_weight": fit.akaike_weight,
            "BIC": fit.bic,
        })
    write_dict_csv(path, rows, list(rows[0]))


def write_observed_fit(path, time, conversion, onset_index, fit):
    function = model_function(fit.name)
    rows = []
    for index, (time_value, measured) in enumerate(zip(time, conversion)):
        used = index >= onset_index
        reaction_time = time_value - time[onset_index] if used else ""
        fitted = (float(function(np.array([reaction_time]), *fit.parameters)[0])
                  if used else "")
        rows.append({
            "experimental_time_min": time_value,
            "reaction_time_min": reaction_time,
            "measured_conversion_X": measured,
            "measured_conversion_percent": 100.0 * measured,
            "fitted_conversion_X": fitted,
            "fitted_conversion_percent": 100.0 * fitted if used else "",
            "residual_X": measured - fitted if used else "",
            "used_for_kinetic_fit": used,
            "phase": "active_reaction" if used else "flow_delay_excluded",
        })
    write_dict_csv(path, rows, list(rows[0]))


def write_consecutive_fit(path, time, fractions, onset_index, fit):
    rows = []
    onset_time = time[onset_index]
    for index, time_value in enumerate(time):
        used = index >= onset_index
        reaction_time = time_value - onset_time if used else ""
        if used:
            xD, xM, xG = consecutive_fractions(np.array([reaction_time]),
                                               fit.k1, fit.k2, fit.xD0, fit.xM0)
            fitted = (float(xD[0]), float(xM[0]), float(xG[0]))
        else:
            fitted = ("", "", "")
        measured = fractions[index]
        rows.append({
            "experimental_time_min": time_value,
            "reaction_time_min": reaction_time,
            "measured_x_EGDA": measured[0],
            "measured_x_EGMA": measured[1],
            "measured_x_EG": measured[2],
            "fitted_x_EGDA": fitted[0],
            "fitted_x_EGMA": fitted[1],
            "fitted_x_EG": fitted[2],
            "residual_x_EGDA": measured[0] - fitted[0] if used else "",
            "residual_x_EGMA": measured[1] - fitted[1] if used else "",
            "residual_x_EG": measured[2] - fitted[2] if used else "",
            "used_for_kinetic_fit": used,
            "phase": "active_reaction" if used else "flow_delay_excluded",
        })
    write_dict_csv(path, rows, list(rows[0]))


def make_prediction_curve(time, onset_index, fit, cfg):
    onset_time = float(time[onset_index])
    active_duration = max(float(time[-1] - onset_time), 1.0)
    configured_end = cfg.get("prediction_end_minute")
    if configured_end is not None:
        end_time = float(configured_end)
    else:
        end_time = onset_time + float(cfg["prediction_extension_factor"]) * active_duration
        target = float(cfg.get("prediction_target_percent", 90.0)) / 100.0
        target_time = time_to_conversion(fit, target)
        if target_time is not None:
            end_time = max(end_time, onset_time + target_time)
    end_time = max(end_time, float(time[-1]))
    grid = np.linspace(float(time[0]), end_time, int(cfg["prediction_points"]))
    active = grid >= onset_time
    prediction = np.zeros(grid.size)
    lower = np.zeros(grid.size)
    upper = np.zeros(grid.size)
    center_active, lower_active, upper_active = prediction_with_ci(
        fit, grid[active] - onset_time
    )
    prediction[active] = center_active
    lower[active] = lower_active
    upper[active] = upper_active
    return grid, prediction, lower, upper


def write_prediction_curve(path, grid, prediction, lower, upper, onset_time, fit,
                           consecutive=None):
    rows = []
    for index, (time_value, center, lo, hi) in enumerate(
            zip(grid, prediction, lower, upper)):
        active = time_value >= onset_time
        row = {
            "experimental_time_min": time_value,
            "reaction_time_min": time_value - onset_time if active else "",
            "predicted_conversion_X": center,
            "predicted_conversion_percent": 100.0 * center,
            "ci95_low_X": lo,
            "ci95_high_X": hi,
            "model": fit.name,
            "phase": "active_reaction" if active else "flow_delay",
        }
        if consecutive is not None:
            if active:
                xD, xM, xG = consecutive_fractions(
                    np.array([time_value - onset_time]), consecutive.k1,
                    consecutive.k2, consecutive.xD0, consecutive.xM0)
                row.update({"consecutive_x_EGDA": float(xD[0]),
                            "consecutive_x_EGMA": float(xM[0]),
                            "consecutive_x_EG": float(xG[0]),
                            "consecutive_conversion_X": float(0.5 * (xM[0] + 2 * xG[0]))})
            else:
                row.update({"consecutive_x_EGDA": "", "consecutive_x_EGMA": "",
                            "consecutive_x_EG": "", "consecutive_conversion_X": ""})
        rows.append(row)
    write_dict_csv(path, rows, list(rows[0]))


def save_kinetics_plot(path, time, conversion, fractions, onset_index, fit,
                       grid, prediction, lower, upper, consecutive, metadata, cfg):
    onset_time = time[onset_index]
    used = np.arange(time.size) >= onset_index
    fitted_at_data = model_function(fit.name)(time[used] - onset_time, *fit.parameters)
    residual = conversion[used] - fitted_at_data
    measured_end = float(time[-1])
    show_speciation = consecutive is not None and fractions is not None

    if show_speciation:
        fig, (ax_sp, ax, ax_res) = plt.subplots(
            3, 1, figsize=(11.8, 12.0), sharex=True,
            gridspec_kw={"height_ratios": (2.6, 3.0, 1.0), "hspace": 0.09},
        )
    else:
        ax_sp = None
        fig, (ax, ax_res) = plt.subplots(
            2, 1, figsize=(11.8, 8.2), sharex=True,
            gridspec_kw={"height_ratios": (3.3, 1.0), "hspace": 0.08},
        )

    # ---- panel 1: speciation + the consecutive A->B->C fit ----------------
    if show_speciation:
        fine = np.linspace(onset_time, float(grid[-1]), 400)
        xD, xM, xG = consecutive_fractions(fine - onset_time, consecutive.k1,
                                           consecutive.k2, consecutive.xD0,
                                           consecutive.xM0)
        for series, colour, label, marker in (
                (fractions[:, 0], C_EGDA, "EGDA (diester)", "o"),
                (fractions[:, 1], C_EGMA, "EGMA (monoester)", "s"),
                (fractions[:, 2], C_EG, "EG (glycol)", "^")):
            ax_sp.scatter(time[used], 100.0 * series[used], s=28, color=colour,
                          marker=marker, edgecolor="white", linewidth=0.35, zorder=5,
                          label=label)
            if onset_index:
                ax_sp.scatter(time[:onset_index], 100.0 * series[:onset_index], s=22,
                              color=colour, marker=marker, alpha=0.30, zorder=4)
        for curve, colour in ((xD, C_EGDA), (xM, C_EGMA), (xG, C_EG)):
            ax_sp.plot(fine, 100.0 * curve, color=colour, lw=1.9, zorder=3)
        t_max, x_max = egma_maximum(consecutive)
        if t_max is not None and onset_time + t_max <= float(grid[-1]):
            ax_sp.plot([onset_time + t_max], [100.0 * x_max], marker="*", ms=15,
                       color=C_EGMA, zorder=6)
            ax_sp.annotate(f"max EGMA {100.0*x_max:.1f}%\nat {t_max:.0f} min",
                           xy=(onset_time + t_max, 100.0 * x_max),
                           xytext=(8, 12), textcoords="offset points",
                           fontsize=8.6, color=C_EGMA)
        ax_sp.axvline(onset_time, color="#6b4c9a", lw=1.2, ls="--")
        ax_sp.set_ylabel("mole fraction of the glycol pool / %")
        ax_sp.set_ylim(-4, 108)
        ax_sp.grid(alpha=0.22)
        ax_sp.legend(loc="center right", ncol=1, fontsize=9)
        ratio = consecutive.k1 / consecutive.k2 if consecutive.k2 > 0 else float("nan")
        ax_sp.text(
            0.985, 0.96,
            f"consecutive A→B→C fit\n"
            f"k$_1$ = {consecutive.k1:.5g} min$^{{-1}}$ "
            f"(± {consecutive.std_errors['k1']:.2g})\n"
            f"k$_2$ = {consecutive.k2:.5g} min$^{{-1}}$ "
            f"(± {consecutive.std_errors['k2']:.2g})\n"
            f"k$_1$/k$_2$ = {ratio:.3g}  "
            f"(statistical: {STATISTICAL_K1_OVER_K2:g})\n"
            f"R$^2$ = {consecutive.r2_overall:.5f}",
            transform=ax_sp.transAxes, ha="right", va="top", fontsize=9.2,
            bbox=dict(boxstyle="round", fc="white", ec="0.72", alpha=0.94))

    # ---- panel 2: lumped ester-group conversion ---------------------------
    if onset_index:
        ax.scatter(time[:onset_index], 100.0 * conversion[:onset_index], s=28,
                   color=C_DELAY, alpha=0.75, label="flow delay (excluded)", zorder=4)
    ax.scatter(time[used], 100.0 * conversion[used], s=30, color=C_DATA,
               edgecolor="white", linewidth=0.35,
               label="NMR ester-group conversion (fitted)", zorder=5)
    ax.fill_between(grid, 100.0 * lower, 100.0 * upper, color=C_BAND, alpha=0.35,
                    label="95% parameter CI", zorder=1)
    observed_curve = grid <= measured_end
    ax.plot(grid[observed_curve], 100.0 * prediction[observed_curve], color=C_FIT,
            lw=2.0, label=f"{fit.label} fit", zorder=3)
    if np.any(~observed_curve):
        ax.plot(grid[~observed_curve], 100.0 * prediction[~observed_curve], color=C_FIT,
                lw=1.8, ls="--", label="kinetic prediction", zorder=3)
    if show_speciation:
        active_grid = grid >= onset_time
        ax.plot(grid[active_grid],
                100.0 * consecutive_conversion(grid[active_grid] - onset_time,
                                               consecutive.k1, consecutive.k2,
                                               consecutive.xD0, consecutive.xM0),
                color=C_EGMA, lw=1.6, ls="-.", zorder=3,
                label="from the consecutive fit")
    ax.axvline(onset_time, color="#6b4c9a", lw=1.2, ls="--",
               label=f"fit starts at {onset_time:g} min")
    ax.set_ylabel("ester groups hydrolysed / %")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper left", ncol=2, fontsize=9)

    values = parameter_map(fit)
    half_target = values["X0"] + 0.5 * (values["X_infinity"] - values["X0"])
    half_time = time_to_conversion(fit, half_target)
    text = (
        f"lumped model: {fit.label}\n"
        f"apparent order n = {values['order']:.4g}\n"
        f"k$_{{app}}$ = {values['k_app']:.5g} min$^{{-1}}$\n"
        f"X$_0$ at fit start = {100.0 * values['X0']:.2f}%\n"
        f"X$_{{\\infty}}$ = {100.0 * values['X_infinity']:.2f}%\n"
        f"half-time = {half_time:.1f} min\n"
        f"R$^2$ = {fit.r2:.6f}; RMSE = {100.0 * fit.rmse:.3f}%"
    )
    ax.text(0.985, 0.04, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, bbox=dict(boxstyle="round", fc="white", ec="0.72", alpha=0.94))

    # ---- panel 3: residuals ----------------------------------------------
    ax_res.axhline(0.0, color="0.45", lw=0.9)
    ax_res.scatter(time[used], 100.0 * residual, s=22, color=C_DATA)
    ax_res.axvline(onset_time, color="#6b4c9a", lw=1.0, ls="--")
    ax_res.set_ylabel("residual / %")
    ax_res.set_xlabel("experimental time / min")
    ax_res.grid(alpha=0.20)

    title_bits = ["EGDA hydrolysis kinetics"]
    if metadata.get("sample_id"):
        title_bits.append(str(metadata["sample_id"]))
    if metadata.get("mode"):
        title_bits.append(str(metadata["mode"]))
    fig.suptitle(" — ".join(title_bits), fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.07, top=0.93)
    fig.savefig(path, dpi=cfg["dpi"], bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# DRIVER
# ===========================================================================
def main(cfg=CONFIG):
    input_path = resolve_config_path(cfg["input_file"], must_exist=True)
    if not input_path.is_file():
        raise SystemExit(f"Input CSV does not exist: {input_path}")
    output_folder = resolve_config_path(cfg["output_folder"])
    output_folder.mkdir(parents=True, exist_ok=True)

    try:
        time, conversion, fractions, metadata = load_conversion_csv(input_path, cfg)
        onset_index = detect_onset(time, conversion, cfg)
        reaction_time = time[onset_index:] - time[onset_index]
        active_conversion = conversion[onset_index:]
        if active_conversion.size < 6:
            raise ValueError("Fewer than six points remain after flow-delay removal")
        fits, best, failures = fit_all_models(reaction_time, active_conversion, cfg)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Kinetics analysis failed: {exc}") from exc

    consecutive = None
    consecutive_note = ""
    if cfg.get("fit_consecutive", True):
        if fractions is None:
            consecutive_note = ("no x_EGDA/x_EGMA/x_EG columns -- run main.py in "
                                "'backbone' mode to get the speciation")
        else:
            try:
                consecutive = fit_consecutive(reaction_time, fractions[onset_index:])
            except (RuntimeError, ValueError) as exc:
                consecutive_note = str(exc)

    initial_M = cfg.get("initial_concentration_M")
    if initial_M is None:
        initial_M = metadata.get("initial_concentration_M")
    if initial_M is not None:
        initial_M = float(initial_M)

    parameter_rows = build_parameter_rows(best, time, onset_index, initial_M,
                                          consecutive)
    parameter_path = output_folder / "kinetic_parameters.csv"
    write_dict_csv(parameter_path, parameter_rows, list(parameter_rows[0]))

    comparison_path = output_folder / "kinetic_model_comparison.csv"
    write_model_comparison(comparison_path, fits)

    observed_path = output_folder / "kinetic_observed_vs_fitted.csv"
    write_observed_fit(observed_path, time, conversion, onset_index, best)

    written = [parameter_path, comparison_path, observed_path]

    if consecutive is not None:
        consecutive_path = output_folder / "consecutive_observed_vs_fitted.csv"
        write_consecutive_fit(consecutive_path, time, fractions, onset_index,
                              consecutive)
        written.append(consecutive_path)

    grid, prediction, lower, upper = make_prediction_curve(time, onset_index, best, cfg)
    prediction_path = output_folder / "kinetic_prediction_curve.csv"
    write_prediction_curve(prediction_path, grid, prediction, lower, upper,
                           time[onset_index], best, consecutive)
    written.append(prediction_path)

    plot_path = output_folder / "kinetics_fit.png"
    save_kinetics_plot(plot_path, time, conversion, fractions, onset_index, best,
                       grid, prediction, lower, upper, consecutive, metadata, cfg)
    written.append(plot_path)

    values = parameter_map(best)
    half_target = values["X0"] + 0.5 * (values["X_infinity"] - values["X0"])
    half_time = time_to_conversion(best, half_target)
    print(f"Loaded {len(time)} observations from {input_path}")
    print(f"Flow delay: {time[onset_index] - time[0]:.1f} min "
          f"({onset_index} point(s) excluded)")
    print(f"Lumped model: {best.label}  |  order={values['order']:.4g}  "
          f"k_app={values['k_app']:.6g} min^-1")
    print(f"  R2={best.r2:.6f}  RMSE={100.0 * best.rmse:.3f}%  "
          f"half-time={half_time:.1f} min")
    if consecutive is not None:
        ratio = consecutive.k1 / consecutive.k2 if consecutive.k2 > 0 else float("nan")
        t_max, x_max = egma_maximum(consecutive)
        print(f"Consecutive EGDA->EGMA->EG:  "
              f"k1={consecutive.k1:.6g}  k2={consecutive.k2:.6g} min^-1  "
              f"k1/k2={ratio:.3g} (statistical {STATISTICAL_K1_OVER_K2:g})")
        print(f"  R2(overall)={consecutive.r2_overall:.6f}  "
              + "  ".join(f"R2({k})={v:.4f}"
                          for k, v in consecutive.r2_per_species.items()))
        if t_max is not None:
            print(f"  EGMA peaks at {100.0*x_max:.1f}% after {t_max:.1f} min "
                  f"of reaction")
    elif consecutive_note:
        print(f"Consecutive fit skipped: {consecutive_note}")
    if failures:
        print("Lumped models skipped: " + "; ".join(failures))
    print(f"Saved kinetics outputs to {output_folder}")
    for path in written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
