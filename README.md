# Modelling vortex interactions reveals distinct patterns of brain reconfiguration following psilocybin and escitalopram

Code accompanying the manuscript by Acosta Gordo and colleagues. The study compares subject-specific Hopf whole-brain models fitted in conventional BOLD signal space and in a mesoscopic vortex space derived from local synchronisation dynamics. Resting-state fMRI was acquired before and after psilocybin or escitalopram treatment in 42 participants with major depressive disorder (ClinicalTrials.gov: [NCT03429075](https://clinicaltrials.gov/study/NCT03429075)).

The repository contains the whole-brain model-fitting, machine-learning, and statistical analysis code used in the manuscript. It does not contain participant-level neuroimaging data, clinical scores, response assignments, structural connectivity, fitted connectivity matrices, or analysis outputs. The group-average structural connectivity matrix used to constrain the models was derived from  diffusion MRI data from 32 healthy participants in the Human Connectome Project (HCP) and is not redistributed here ([Setsompop et al., 2013](https://doi.org/10.1016/j.neuroimage.2013.05.078)).

## Repository contents

| Directory | Contents |
| --- | --- |
| `01_vortex_space_model` | Four Slurm/MATLAB jobs for subject-specific GECVO fitting, together with the Gaussian-filter helper and Schaefer 1000-to-100 mapping. |
| `02_signal_space_model` | Four Slurm/MATLAB jobs for subject-specific GEC fitting, with the same local helper files. |
| `03_machine_learning` | XGBoost classifiers for psilocybin pre versus post, escitalopram pre versus post, post-treatment psilocybin versus escitalopram, and baseline escitalopram response. |
| `04_regional_gbc` | Regional global brain connectivity analysis and Schaefer-100 labels. |
| `05_connectome_based_modelling` | In-sample association between baseline regional GBC and subsequent BDI improvement. |
| `06_network_modularity` | Newman modularity and module-count analysis using repeated Louvain optimisation. |
| `07_turbulence` | Longitudinal turbulence analysis in responders and correlations with clinical change. |

## Software

The model-fitting jobs require MATLAB, a Slurm workload manager, and the MATLAB Signal Processing, Statistics and Machine Learning, and Image Processing toolboxes. Downstream MATLAB analyses use the Statistics and Machine Learning Toolbox. `community_louvain.m`, required for the modularity analysis, is included in `06_network_modularity`.

The machine-learning analyses require Python and the packages listed in `requirements.txt`. A local environment can be created with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Model inputs and outputs

The model-fitting jobs are intended to be submitted from their own directory. The relevant trial time-series file and the structural-connectivity file must be placed in that same directory by an authorised user. The scripts expect the following input filenames:

- `psilodep2_1000_before_psilocybin.mat`
- `psilodep2_1000_after_psilocybin.mat`
- `psilodep2_1000_before_escitalopram.mat`
- `psilodep2_1000_after_escitalopram.mat`
- `SC_schaefer100_17Networks_32fold_groupconnectome_2mm_symm.mat`, containing the variable `SC`

Each Slurm array task fits one participant. Vortex-space jobs write `model_vortex_partition_social_###.mat`; signal-space jobs write `model_hopf_FC_###.mat`. Both output types contain `Ceff_task1`. Vortex-space outputs additionally contain the empirical turbulence variable `Turbu_emp_task1` used in the turbulence analysis.

For example:

```bash
cd 01_vortex_space_model
sbatch fit_gecvo_psilocybin_pre.sbatch
```

The fitting procedure is stochastic. Each array task starts from an independently initialised random state and averages simulated observables over 50 repetitions, as described in the manuscript.

## Machine-learning analyses

All classifiers use the 4,950 unique lower-triangular connections from each Schaefer-100 matrix. Feature ranking is recomputed within each training fold using Wilcoxon statistics with 10,000 permutations. The longitudinal analyses use leave-one-subject-out cross-validation. Independent-group analyses use balanced leave-pair-out sampling repeated over 1,000 shuffles. XGBoost hyperparameters are fixed across tasks and connectivity spaces.

Example longitudinal run:

```bash
python 03_machine_learning/classify_psilocybin_pre_post.py \
  --space vortex \
  --before-dir <psilocybin_pre_models> \
  --after-dir <psilocybin_post_models>
```

Example post-treatment comparison:

```bash
python 03_machine_learning/classify_post_treatment.py \
  --space signal \
  --psilocybin-dir <psilocybin_post_models> \
  --escitalopram-dir <escitalopram_post_models>
```

For the escitalopram response analysis, baseline model files must be supplied in separate responder and non-responder directories. The response assignment itself is not stored in this repository:

```bash
python 03_machine_learning/classify_escitalopram_response.py \
  --space vortex \
  --responders-dir <escitalopram_pre_responders> \
  --non-responders-dir <escitalopram_pre_nonresponders>
```

Each script accepts an optional `--out` argument and otherwise writes a compressed MATLAB results file to the current directory.

## MATLAB analyses

The downstream functions take input directories and clinical vectors as arguments; they do not impose a separate data-directory structure. Run `help <function_name>` in MATLAB for the full argument list.

```matlab
regional_gbc_analysis(...)
cpm_insample_gbc(...)
newman_modularity_analysis(...)
turbulence_analysis(...)
```

`regional_gbc_analysis` and `newman_modularity_analysis` are each run once for `signal` and once for `vortex`. Regional and modularity changes are defined as post-treatment minus pre-treatment. The connectome-based model uses pre-treatment minus post-treatment BDI as improvement, so larger positive values indicate greater improvement. The turbulence analysis uses post-treatment minus pre-treatment BDI, so more negative values indicate greater improvement.

The modularity analysis uses the Newman objective at resolution parameter γ = 1. Louvain optimisation is repeated 100 times for each matrix, and the partition with the highest Q is retained. Only Q and the corresponding number of modules are analysed.

All inferential Wilcoxon analyses use 10,000 permutations. The analysis functions save numerical results and tables but do not generate manuscript figures.

## Data availability

Requests for access to raw and analysed neuroimaging data from the clinical trial are reviewed by the chief investigator of the original study. Patient-related clinical data may be restricted to protect participant confidentiality. Structural-connectivity data were derived from the Human Connectome Project and are available from the original source. No participant-level or derived trial data are distributed with this repository.

## Citation

If you use this code, please cite:

> Acosta Gordo, Á., Dagnino, P. C., Sanz Perl, Y., Nutt, D. J., Carhart-Harris, R. L., Kringelbach, M. L., & Deco, G. (2026). *Modelling vortex interactions reveals distinct patterns of brain reconfiguration following psilocybin and escitalopram*. Manuscript in preparation.

The citation metadata in `CITATION.cff` should be updated with the journal DOI, volume, issue, and page or article number after publication.

## License and third-party material

Repository-authored code is released under the MIT License. `community_louvain.m`, `gaussfilt.m`, and atlas-derived mapping and label files retain their original attribution and applicable terms; see `THIRD_PARTY_NOTICES.md`.

## Contact

Álvaro Acosta Gordo: alvaro.acosta01@estudiant.upf.edu  
Gustavo Deco: gustavo.deco@upf.edu
