# Graph Report - .  (2026-08-06)

## Corpus Check
- Corpus is ~43,250 words - fits in a single context window. You may not need a graph.

## Summary
- 226 nodes · 432 edges · 15 communities
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 15 edges (avg confidence: 0.68)
- Token cost: 40,194 input · 0 output

## Community Hubs (Navigation)
- Kinetic Model Fitting
- Synthetic NMR Spectrum Simulation
- Bruker Batch Data Extraction
- EGDA Hydrolysis Chemistry Concepts
- Acetyl-Channel Analysis Pipeline
- Real Spectrum Analysis Orchestration
- Backbone Triplet Peak Fitting
- Anchor Calibration and Spectrum Loading
- Interactive Spectrum Plotting
- Region Slicing and Speciation from Areas
- Synthetic Test Data Generation
- arPLS Robust Baseline Correction
- Graphify Agent Instructions
- Linear Baseline and Fit Quality
- Time Axis and Input Resolution

## God Nodes (most connected - your core abstractions)
1. `main()` - 17 edges
2. `main()` - 17 edges
3. `measure_backbone()` - 15 edges
4. `main()` - 13 edges
5. `analyze_spectrum()` - 12 edges
6. `measure_acetyl()` - 10 edges
7. `Backbone O-CH2 handle (3.6-4.4 ppm)` - 10 edges
8. `resolve_backbone_anchors()` - 9 edges
9. `speciation_at_conversion()` - 9 edges
10. `concentrations_at()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `main()` --implements--> `Backbone O-CH2 handle (3.6-4.4 ppm)`  [EXTRACTED]
  analyze_real_nmr.py → README.md
- `main()` --implements--> `Acetyl CH3 handle (~2.1 ppm)`  [EXTRACTED]
  main.py → README.md
- `main()` --references--> `Anchor shift calibration (literature values)`  [INFERRED]
  main.py → README.md
- `main()` --implements--> `Backbone O-CH2 handle (3.6-4.4 ppm)`  [EXTRACTED]
  main.py → README.md
- `main()` --shares_data_with--> `conversion_vs_time_<mode>.csv master table`  [EXTRACTED]
  main.py → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Constraints that make the four-signal backbone fit identifiable** — readme_equal_proton_count_cancellation, readme_egma_equal_amplitude_constraint, readme_rigid_anchor_prepass, readme_broad_water_component [EXTRACTED 1.00]
- **Five-stage EGDA analysis pipeline (extract -> inspect -> deconvolve -> batch -> kinetics)** — readme_egda_pipeline, readme_backbone_handle, readme_acetyl_handle, readme_conversion_vs_time_csv, readme_k1_k2_ratio [EXTRACTED 1.00]
- **Repo-wide graphify agent instruction set** — _claude_claude_graphify, agents_graphify, claude_graphify, agents_graph_first_query_policy [EXTRACTED 1.00]

## Communities (15 total, 0 thin omitted)

### Community 0 - "Kinetic Model Fitting"
Cohesion: 0.11
Nodes (37): build_parameter_rows(), _choose_column(), consecutive_conversion(), consecutive_fractions(), ConsecutiveFit, detect_onset(), egma_maximum(), fit_all_models() (+29 more)

### Community 1 - "Synthetic NMR Spectrum Simulation"
Cohesion: 0.10
Nodes (40): _annotations(), area_under_curve(), averaged_exchange_peak(), build_group_records(), build_species_records(), build_spectrum(), concentrations_at(), ester_conversion_at_tau() (+32 more)

### Community 2 - "Bruker Batch Data Extraction"
Cohesion: 0.17
Nodes (19): ArgumentParser, add_sample_header(), build_argument_parser(), discover_measurements(), extract_batch(), main(), Measurement, parse_audit_timestamp() (+11 more)

### Community 3 - "EGDA Hydrolysis Chemistry Concepts"
Cohesion: 0.15
Nodes (20): Acetyl CH3 handle (~2.1 ppm), Stiff arPLS background removal, Backbone O-CH2 handle (3.6-4.4 ppm), Binomial statistical speciation (k1/k2 = 2), Explicit broad water component excluded from speciation, Consecutive hydrolysis EGDA -> EGMA -> EG, conversion_vs_time_<mode>.csv master table, ~1 % detection floor / phantom EG peak (+12 more)

### Community 4 - "Acetyl-Channel Analysis Pipeline"
Cohesion: 0.14
Nodes (17): acetyl_amounts(), _acetyl_model(), build_species_rows(), _num(), _plot_acetyl(), plot_vs_time(), Linear baseline + EGDA acetyl + EGMA acetyl + acetic acid. Constraints that…, What the acetyl handle alone can say: acetic acid released and water left. The… (+9 more)

### Community 5 - "Real Spectrum Analysis Orchestration"
Cohesion: 0.21
Nodes (14): amounts_at(), anchors_of(), _json_safe(), main(), plot_speciation_vs_time(), Where the water resonance is expected: delta ~ 5.051 - 0.0111*T(degC)., (EGDA, EGMA-ester, EGMA-hydroxy, EG) centres from CONFIG., Moles of every species from the measured glycol-pool composition. (+6 more)

### Community 6 - "Backbone Triplet Peak Fitting"
Cohesion: 0.23
Nodes (12): backbone_model(), measure_backbone(), pseudo_voigt(), Pseudo-Voigt line: `w` is the half-width at half maximum (ppm)., A 1:2:1 backbone triplet (2 H coupled to 2 equivalent vicinal H). `h` is the…, Linear baseline + broad water + the four backbone components. Note `hM` appears…, backbone_model with the four centres locked to `anchors` + one shift., Deconvolve the backbone region into EGDA / EGMA / EG and integrate each. Two… (+4 more)

### Community 7 - "Anchor Calibration and Spectrum Loading"
Cohesion: 0.21
Nodes (12): initial_amounts(), load_spectrum(), Parse a Bruker ASCII spectrum -> (ppm, intensity, hz, sample_id)., Refine the four anchor shifts from the whole batch. Runs the rigid pre-pass on…, (n0 EGDA, n0 water). Water is estimated by volume when V is known., resolve_backbone_anchors(), analyze_spectrum(), main() (+4 more)

### Community 8 - "Interactive Spectrum Plotting"
Cohesion: 0.27
Nodes (11): _abs(), _add_key_bindings(), load_integrals(), load_spectrum(), main(), make_plot(), Relative CONFIG paths are read next to this script, not next to the cwd., Hotkeys on top of the toolbar: 'r' reset, 'a' autoscale y to the view. (+3 more)

### Community 9 - "Region Slicing and Speciation from Areas"
Cohesion: 0.29
Nodes (7): analyze_one(), noise_level(), Analyse one spectrum -> a normalised result dict., Robust noise sigma from a signal-free window. A plain standard deviation over…, Glycol-pool mole fractions and the two conversions. All three species carry…, slice_region(), speciation_from_areas()

### Community 10 - "Synthetic Test Data Generation"
Cohesion: 0.48
Nodes (6): check(), main(), Run the backbone analysis on the test data and score it against truth., Analytical A->B->C solution from pure EGDA., true_fractions(), write_test_data()

### Community 11 - "arPLS Robust Baseline Correction"
Cohesion: 0.33
Nodes (6): arpls_baseline(), Low-order baseline through the signal-free points of a region. Iteratively…, Smooth CURVED background via asymmetrically-reweighted penalised least squares…, Smooth background over a region as an ARRAY aligned with x. lam is auto-scaled…, region_baseline(), robust_baseline()

### Community 12 - "Graphify Agent Instructions"
Cohesion: 0.50
Nodes (5): graphify skill trigger (.claude/CLAUDE.md), Query-the-graph-before-grep policy, graphify usage rules for agents, graphify update . after code changes, graphify project rules (CLAUDE.md)

### Community 13 - "Linear Baseline and Fit Quality"
Cohesion: 0.40
Nodes (5): linear_baseline(), r_squared(), Straight baseline through the (signal-free) edges of a window., measure_acetyl(), Fit the two acetyl singlets and integrate each one.

### Community 14 - "Time Axis and Input Resolution"
Cohesion: 0.50
Nodes (4): minute_of(), Reaction time in minutes parsed from a trailing number, else None., CONFIG['input'] -> a sorted list of spectrum files (single file or batch)., resolve_inputs()

## Knowledge Gaps
- **3 isolated node(s):** `graphify skill trigger (.claude/CLAUDE.md)`, `graphify update . after code changes`, `Binomial statistical speciation (k1/k2 = 2)`
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Anchor Calibration and Spectrum Loading` to `EGDA Hydrolysis Chemistry Concepts`, `Acetyl-Channel Analysis Pipeline`, `Real Spectrum Analysis Orchestration`, `Time Axis and Input Resolution`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Why does `Backbone O-CH2 handle (3.6-4.4 ppm)` connect `EGDA Hydrolysis Chemistry Concepts` to `Real Spectrum Analysis Orchestration`, `Anchor Calibration and Spectrum Loading`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Why does `resolve_config_path()` connect `Kinetic Model Fitting` to `Bruker Batch Data Extraction`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **What connects `graphify skill trigger (.claude/CLAUDE.md)`, `graphify update . after code changes`, `Binomial statistical speciation (k1/k2 = 2)` to the rest of the system?**
  _3 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Kinetic Model Fitting` be split into smaller, more focused modules?**
  _Cohesion score 0.10685249709639953 - nodes in this community are weakly interconnected._
- **Should `Synthetic NMR Spectrum Simulation` be split into smaller, more focused modules?**
  _Cohesion score 0.1048780487804878 - nodes in this community are weakly interconnected._
- **Should `EGDA Hydrolysis Chemistry Concepts` be split into smaller, more focused modules?**
  _Cohesion score 0.14736842105263157 - nodes in this community are weakly interconnected._