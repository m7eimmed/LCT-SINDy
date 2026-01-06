# LCT–SINDy Codes (Paper Reproducibility Package)

MATLAB implementation of the **Linear Chain Trick (LCT)** + **SINDy** workflow used in the accompanying paper.
The repository contains runnable scripts to reproduce the main numerical examples and figures.

## Folder structure

- 'Run_Logistic_example.m' — Logistic distributed-delay example illustrating full model identification from noisy data using the LCT–SINDy method.
- 'Run_Hes1_mRNA_Example.m' — Hes1–mRNA distributed-delay model example.
- 'Run_Ikeda_example.m' — Ikeda delay system example identified via an LCT (linear-chain) approximation.
- 'Run_Figure1DiscreteSINDy.m' — Discrete-delay SINDy example using a library augmented only with a discrete delayed state (no distributed-delay approximation).
- 'Run_Figure4LCTStability.m' — LCT stability/robustness experiment (Figure 4-style experiment).
- 'utils/' — Helper functions (LCT chain builders, library construction, STRidge, rollouts, derivative estimation, RHS functions).

## Requirements

- **MATLAB** (tested on **R2025b**).
- **Signal Processing Toolbox** (required for `sgolayfilt`, used in denoising and derivative estimation).


## Quick start

1. Clone the repository and open MATLAB in the repository root.
2. Run any of the scripts below. Most scripts automatically add the `utils/` folder to the MATLAB path.

## Reproducing paper results

### Logistic example
```matlab
run('Run_Logistic_example.m')


### Ikeda example
```matlab
run('Run_Ikeda_example.m')

Notes:
- The Logistic and Ikeda examples use the same derivative-estimation helper function.  
For exact reproducibility, ensure that the Savitzky–Golay window is set to (0.5/dt) when running the Ikeda example and to (2.5/dt) when running the Logistic example.  

### Hes1–mRNA example
```matlab
run('Run_Hes1_mRNA_Example.m')


### Figure-style scripts
```matlab
run('Run_Figure1DiscreteSINDy.m')
run('Run_Figure4LCTStability.m')

## Reproducibility

Random seeds are explicitly set inside the main scripts using `rng(...)`.
If you remove or modify `rng`, results will change.

## Common troubleshooting

- **`sgolayfilt` not found**  
  Install/enable **Signal Processing Toolbox**, or switch derivative mode to `'raw'` in the scripts.

## Citation

If you use this code, please cite the accompanying paper.

## Contact
Majid Bani-Yaghoub   baniyaghoubm@umkc.edu
Mohammed Alanazi     mnavy3@umkc.edu
