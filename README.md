# Stable OU identification under unknown observation noise

Research materials accompanying *Finite-Feature Identification of Stable Ornstein-Uhlenbeck Dynamics with Unknown Observation Noise*.

ZHANG WENHAN  
Institute of Mathematical Sciences, Faculty of Science, Universiti Malaya, Malaysia  
Contact: 24084017@siswa.um.edu.my

## Research materials

- [Release v0.15](https://github.com/vincent-zhang0116/stable-ou-identification/releases/tag/v0.15)
- [Complete code and simulation data](https://github.com/vincent-zhang0116/stable-ou-identification/releases/download/v0.15/Stable_OU_v0.15_research_materials.zip)
- [Article source](Stable_OU_v0.15.tex)
- [Related-literature comparison](literature_matrix.csv)

The release contains the simulation programs, fixed configurations, per-path results, selected complete sample paths, numerical checks and reproduction instructions. Download the named research-materials ZIP; GitHub's automatically generated source archives contain only the repository files.

## Simulation studies

The baseline study uses 1,500 independent latent paths and records 36,000 paired method/scenario outcomes across 108 summary scenarios. An additional study uses 700 independent paths for frequency selection, a scalar moment comparator and correlated-Gaussian-noise diagnostics. Multiple outcomes from the same path are paired observations.

A complete baseline rerun reproduced all point-status and coverage decisions. Maximum absolute parameter-estimate differences were approximately 4.96e-13. Floating-point arrays are compared numerically across environments. Twelve complete baseline paths with observation variants are included; the remaining paths can be regenerated from the recorded random-generator inputs.

## Reproduction

Extract the research-materials ZIP and run the following commands from its top-level directory. The diagnostic environment uses Python 3.13.16, NumPy 2.5.4 and SciPy 1.18.1.

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python -m pip install -r requirements.txt
python verify_public_release.py
python analysis/check_added_results.py
```

Rerun the baseline simulation into a separate directory:

```powershell
python original_v013/source/experiment.py --config original_v013/config.json --out independent_rerun --workers 4
```

The archive README describes the additional simulations, recorded variables and numerical tolerances. The article source is self-contained and compiles with a standard LaTeX distribution.

## Diagnostic results

![Frequency and Gaussian-endpoint diagnostics](diagnostics.png)

Short-record coverage and inversion near the Gaussian endpoint can be poor. The article states the assumptions and limitations of each inferential result. Version-specific access limits for related work are recorded in the literature comparison.

## Citation

ZHANG WENHAN. *Finite-Feature Identification of Stable Ornstein-Uhlenbeck Dynamics with Unknown Observation Noise*. Research materials, version 0.15 (2026). https://github.com/vincent-zhang0116/stable-ou-identification/releases/tag/v0.15

Funding, competing interests and AI-assistance declarations are provided in the article.
