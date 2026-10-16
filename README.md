# Stable OU identification under unknown observation noise

Research materials accompanying *Identification of Stable Ornstein-Uhlenbeck Dynamics under Unknown Observation Noise*.

ZHANG WENHAN, Institute of Mathematical Sciences, Faculty of Science, Universiti Malaya, Malaysia. Contact: 24084017@siswa.um.edu.my.

## Research materials

Version 0.22 contains the code and simulation records for the study. Download the [complete research archive](https://github.com/vincent-zhang0116/stable-ou-identification/releases/download/v0.22/Stable_OU_Research_Materials_v0.22.zip) from the [versioned release](https://github.com/vincent-zhang0116/stable-ou-identification/releases/tag/v0.22). The `code/` directory provides a browsable copy of the Python sources and configurations; reproduction requires the complete archive, including its recorded random streams and results.

The archive contains 6,050 independent latent paths: 1,500 baseline, 700 diagnostic, 2,500 confirmation and 1,350 finite-variance and cusp paths. Its 125,520 method records include failed fits and constrained boundary returns. Methods, error laws and sample prefixes are paired within paths. Six supplementary diagnostic tables and source-to-display mappings are included.

## Reproduction

Extract the research archive and run from `Stable_OU_Research_Materials/`, using Python 3.13.16 and the pinned dependencies:

```text
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python reproduce.py --destination rebuilt
```

On Unix-like systems, replace `.venv\Scripts\python` with `.venv/bin/python`. The default command reconstructs the article displays and supplementary diagnostic tables. To rerun all 6,050 paths and reconstruct the displays:

```text
.venv\Scripts\python reproduce.py --destination full_rerun --fits
```

Use a new destination for each run. The archive checks source and data hashes before reconstruction. Historical directory names are retained to preserve imports and recorded provenance.

## Verification and scope

All 6,050 paths were rerun internally in an isolated environment on the same operating system. Return, coverage, boundary and accuracy decisions matched the stored records. The original Python 3.12 baseline has small floating-point differences under Python 3.13; use the numerical tolerances documented in the archive. Later studies matched their checked numerical values and array hashes. No third-party or second-operating-system reproduction is claimed.

Complete arrays are retained for selected paths; remaining observations regenerate from recorded random streams. Fixed-bandwidth cusp experiments assess drift recovery, not the separate joint estimator's asymptotic rate. Optimizer convergence does not certify the global objective-gap condition of the statistical theorem.

## Archive integrity

```text
101b2dc7b6cec528ef848388bd8f722c834506f2de7c258ddd6aa0b2e50e85dc  Stable_OU_Research_Materials_v0.22.zip
```

The research archive is byte-identical to the computational supplement supplied with the article. GitHub's automatically generated source archives contain repository files only; use the named release asset for the complete reproducibility package.

## Earlier materials

[Version 0.15](https://github.com/vincent-zhang0116/stable-ou-identification/releases/tag/v0.15) preserves the earlier baseline and diagnostic materials. `Stable_OU_v0.15.tex`, `diagnostics.png` and `literature_matrix.csv` in the repository are historical files from that version.

## Citation

ZHANG WENHAN (2026). *Identification of Stable Ornstein-Uhlenbeck Dynamics under Unknown Observation Noise: research materials*. Version 0.22. https://github.com/vincent-zhang0116/stable-ou-identification/releases/tag/v0.22
