# EGDA hydrolysis — 1H NMR pipeline

The same pipeline as `NMR/EtAC`, rebuilt for the acid-catalysed hydrolysis of
**ethylene glycol diacetate**.

```
EGDA + H2O  --k1-->  EGMA + AcOH          EGDA = CH3COO-CH2CH2-OOCCH3
EGMA + H2O  --k2-->  EG   + AcOH          EGMA = CH3COO-CH2CH2-OH   (2-hydroxyethyl acetate)
                                          EG   = HO-CH2CH2-OH
```

## What is different from the ethyl-acetate pipeline

Ethyl acetate hydrolyses in **one** step, so two signals and one ratio gave you
everything. EGDA is a **consecutive** reaction with an intermediate that rises
and falls, and that changes three things:

| | ethyl acetate | EGDA |
|---|---|---|
| what you measure | one conversion `X` | the full speciation `x(EGDA) / x(EGMA) / x(EG)` |
| signals to separate | 2 (well apart) | 4 (overlapping, 5–10 Hz apart at 80 MHz) |
| kinetics | one `k` | `k1` **and** `k2`, plus the ratio `k1/k2` |

Two conversions are reported, because both are useful:

- **`X_EGDA`** — the fraction of the diester that has reacted at all
  `= x(EGMA) + x(EG)`
- **`X_ester`** — the fraction of **all ester groups** hydrolysed (EGDA has two,
  EGMA has one) `= [x(EGMA) + 2·x(EG)] / 2 = n(AcOH) / (2·n0)`

`X_ester` is what goes in the `conversion_X` column, because it is the quantity
both handles can measure and the one that maps onto acetic acid released.

## The pipeline

Every script is IDE-run-and-go: edit the `CONFIG` dict at the top, press Run.

| # | script | what it does |
|---|---|---|
| 1 | `Brucker_batch_extractor.py` | raw Bruker folders → `ascii-spec_<minute>.txt`, ordered by the `auditp` start time |
| 2 | `plot_nmr.py` | look at a spectrum (interactive) or batch-save PNGs — **use this first to check where your peaks actually are** |
| 3 | `analyze_real_nmr.py` | the backbone O-CH2 deconvolution → per-spectrum speciation, standalone |
| 4 | `main.py` | the same, as a batch driver with two selectable handles → `conversion_vs_time_<mode>.csv` |
| 5 | `kinetics_estimation.py` | that CSV → `k1`, `k2`, the lumped rate law, prediction curves |
| — | `sim_nmr.py` | forward-simulate the 80 MHz spectrum at any conversion |
| — | `make_test_data.py` | synthetic data with known `k1`/`k2`, to verify the whole chain |

## The two handles

Set `CONFIG["conversion_mode"]` in `main.py`.

### `"backbone"` — the O-CH2 region (~3.6–4.4 ppm)

The `-CH2CH2-` backbone is never broken, so **every** glycol-derived molecule
carries exactly **four** backbone protons:

| species | signal | δ (ppm) |
|---|---|---|
| EGDA | singlet, 4 H (both CH2 equivalent) | ~4.34 |
| EGMA | 1:2:1 triplet, 2 H (ester side) | ~4.25 |
| EGMA | 1:2:1 triplet, 2 H (hydroxy side) | ~3.78 |
| EG | singlet, 4 H (both CH2 equivalent) | ~3.66 |

Identical proton counts means the common `4 H` factor cancels, so **the area
fractions *are* the mole fractions** — no internal standard, no calibration.
The total backbone area is conserved, which is a free quality check on every
spectrum.

The four signals overlap badly at 80 MHz, so they are fitted together with the
physics wired into the model. The constraint that makes it work: **EGMA's two
triplets are the same 2 H of the same molecule, so they are forced to equal
amplitude and therefore equal area.** The clean ~4.25 ppm triplet therefore
*predicts* the badly-overlapped ~3.78 ppm one outright, and whatever is left at
~3.66 ppm must be ethylene glycol. All four also share one line width and shape,
and the anchor set is fitted rigidly first (one global referencing shift,
multi-started so it cannot lock onto the wrong neighbour) before any centre is
allowed to drift.

### `"acetyl"` — the CH3 singlets (~2.1 ppm)

An acetyl group is either still esterified (EGDA 2.140 / EGMA 2.125 ppm) or free
acetic acid (~2.080 ppm). All are 3 H per acetyl group, so

```
X_ester = A(AcOH) / [ A(AcOH) + A(esterified acetyl) ]
```

is exact and gain-independent. The two ester acetyls are only ~1 Hz apart at
80 MHz so their individual amplitudes are meaningless — but they are modelled as
two fixed-offset lines anyway, because collapsing them into one shared-width
line makes it too narrow, leaks intensity into acetic acid and biases `X` high
by ~3 percentage points.

This handle gives **no EGMA/EG split**, but it sits far from water. Use it as
the cross-check.

## ⚠ Water — read this before trusting the backbone numbers

The water resonance moves upfield with temperature:

```
delta(H2O) / ppm  ~  5.051 - 0.0111 * T(degC)
```

| T | δ(H2O) | nearest backbone signal |
|---|---|---|
| 25 °C | 4.77 ppm | clear |
| 49 °C | 4.51 ppm | on the shoulder of EGDA (4.34) |
| **70 °C** | **4.28 ppm** | **on top of EGDA (4.34) and EGMA-ester (4.25)** |

At your usual reaction temperatures water lands *in* the backbone region. Two
things are done about it — a stiff arPLS background removes the smooth tail, and
an explicit **broad water component** is fitted inside the region and excluded
from the speciation — and the per-spectrum PNG shows you both the raw
water-dominated data and the water-removed trace so you can judge it yourself.
Even so: **at high temperature, cross-check with the acetyl handle**, which does
not care where water is. `sim_nmr.py` prints a collision warning for your
configured temperature.

## Calibrate the anchor shifts before your first real run

Unlike the ethyl-acetate pipeline — whose shifts were fitted from real spectra —
the shifts here are **aqueous literature values**. They live in one place:

```python
# analyze_real_nmr.py
BACKBONE_DEFAULTS = {
    "egda_center":         4.335,
    "egma_ester_center":   4.245,
    "egma_hydroxy_center": 3.780,
    "eg_center":           3.660,
    ...
}
```

`main.py` imports that dictionary rather than keeping a copy, so editing it here
updates both scripts. Only the **relative** spacing has to be right — a common
offset is absorbed by the global shift fitted in the rigid pre-pass, and
`resolve_backbone_anchors` refines all four from your batch automatically. The
acetyl anchors are in `main.py` under `CONFIG["acetyl"]`.

Do check with `plot_nmr.py` first (`"xlim": (4.60, 3.45)`).

## Outputs

`main.py` writes into `results/<mode>_conversion/`:

- `<stem>_<mode>_fit.png` — the deconvolution for that spectrum
- `<stem>_<mode>_groups.csv` — measured signals (δ, J, FWHM, area, SNR, R²)
- `<stem>_<mode>_species.csv` — per-compound moles / mass / molarity
- `conversion_vs_time_<mode>.csv` — the master table (feed this to step 5)
- `conversion_vs_time_<mode>.png` — speciation, conversion and amounts vs time

`kinetics_estimation.py` writes into `results/kinetics_<mode>/`:

- `kinetic_parameters.csv` — `k1`, `k2`, `k1/k2`, half-lives, the EGMA maximum,
  the lumped rate law, quality metrics
- `kinetic_model_comparison.csv` — the lumped rate laws ranked by AICc
- `consecutive_observed_vs_fitted.csv` — measured vs fitted mole fractions
- `kinetic_observed_vs_fitted.csv`, `kinetic_prediction_curve.csv`
- `kinetics_fit.png` — speciation + consecutive fit, lumped fit, residuals

Columns a mode cannot measure are left **empty**, never filled with a guess: an
acetyl-mode table has no `x_EGDA`/`x_EGMA`/`x_EG`, and `kinetics_estimation.py`
then skips the consecutive fit and says so.

## Reading `k1/k2`

If the two ester groups of EGDA hydrolysed independently at the same intrinsic
rate `k`, then `k1 = 2k` (two groups to attack) and `k2 = k`, so

```
k1 / k2 = 2      <- the purely statistical expectation
```

- `k1/k2 > 2` — the second hydrolysis is *harder* than statistics; EGMA piles up
- `k1/k2 < 2` — EGMA reacts faster than statistics (its free OH is activating the
  remaining ester) and the intermediate barely accumulates

For the statistical case the speciation is just binomial,
`x = (p², 2p(1−p), (1−p)²)` with `p = 1 − X_ester` — which is what `sim_nmr.py`
uses by default (`k2_over_k1 = 0.5`).

## Verifying it works

```
python make_test_data.py --check
```

writes 25 synthetic spectra into `test_data/` from a known `k1 = 0.020`,
`k2 = 0.008 min⁻¹` (with water, a curved background, referencing jitter and
noise), then analyses them and prints measured vs true side by side. Point the
`"input"` of `analyze_real_nmr.py` / `main.py` at `test_data` to run the rest of
the chain on it.

On that data the pipeline recovers `k1 = 0.0197`, `k2 = 0.0075 min⁻¹`
(`k1/k2 = 2.62` vs the true 2.5), and the two independent handles agree on
`X_ester` to within 0.2–3 percentage points.

Note that a `test_data` run writes into the same `results/` tree as a real run
and the per-spectrum filenames are the same, so set `CONFIG["output"]` to
something like `results_test` if you want to keep both.

## Known limits

- **Detection floor ~1 %.** On a 0 % spectrum the fit typically parks ~1 % of the
  area on a phantom EG peak. Anything below a couple of percent is not real.
- **Early-time minor components are the worst case.** Errors reach ~5–7
  percentage points in the first few minutes, when EGMA and EG are just emerging,
  and fall below 1 pp once all three are well established.
- **The acetyl handle gives no EGMA/EG split** — only `X_ester` and `n(AcOH)`.
- Shifts are literature values until you recalibrate them (see above).

## Not ported

`analyze_etac_kinetics.py`, `simulate_etac_kinetics.py` and
`compare_etac_kinetics.py` from `EtAC/NMR-EtOH/` are superseded by
`main.py` + `kinetics_estimation.py` and were not carried over.
