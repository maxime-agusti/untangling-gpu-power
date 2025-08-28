# Artifact for the paper: Untangling GPU Power Consumption: Job-Level Inference in Cloud Shared Settings

This document describes the artifact submitted for the research paper "Untangling GPU Power Consumption: Job-Level Inference in Cloud Shared Settings," accepted at the EuroSys 2026 conference.

This artifact aims to obtain the following badges:

  * **Artifact Available**: The source code, data, and documentation are permanently accessible.
  * **Artifact Functional**: The artifact is well-documented, and the provided code can be used to generate the figures and results presented in the paper.

## 1\. Overview

The artifact contains the necessary scripts and tools to process the raw experimental data and generate the figures presented in our paper.

**Important Notice on Reproducibility**: Due to the requirement of highly specific hardware (including precise multi-GPU configurations and IPMI power measurement systems) that is generally unavailable to evaluators, this artifact **is not intended for the full reproduction of the experiments** described in the paper. However, we provide the complete source code used for our experiments and for generating the figures to ensure maximum transparency.

## 2\. Artifact Structure

The project is structured as follows:

```
.
├── src/              # Python scripts to generate figures from the data
├── data/             # Raw data from the experiments
├── brench-res/       # Raw results from the benchmarks
├── figures/          # Output figures directory
├── experiments/      # Scripts to run the experiments
├── README.md         # This file
└── requirements.txt  # Required Python dependencies
```

## 3\. Installation and Dependencies

To use this artifact, you will need **Python 3.10** or a newer version. The Python dependencies can be installed using `pip`.

1.  **Clone the Git repository** for the artifact or unzip the archive.
2.  **Install the dependencies** using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## 4\. User Guide: Generating the Figures

The scripts to generate each figure are located in the `src/` directory. Each script corresponds to one or more figures in the paper.

For example, to generate **Figure 2** from the paper:

```bash
python3 src/Sec-03_temporally-shared/Fig-02_TS-power.py
```

Similarly for the other figures:

  * **Figure 3**: `python3 src/Sec-03_temporally-shared/Fig-03_TS-perf-power.py`
  * **Figure 5**: `python3 src/Sec-04_spatially-shared/Fig-05_MIG-GI-power-all.py`
  * **Figure 6**: `python3 src/Sec-04_spatially-shared/Fig-06_MIG-driver.py`
  * **Figure 7**: `python3 src/Sec-04_spatially-shared/Fig-07_MIG-GI-bench-all.py`
  * **Figure 8**: `python3 src/Sec-05_pass-through/Fig-08_PT-pearson-corr-ipmi.py`
  * **Figure 9**: `python3 src/Sec-05_pass-through/Fig-09_PT-density-4A100-4states-corrected.py`
  * **Figure 10**: `python3 src/Sec-05_pass-through/Fig-10_PT-density-8A100-2states-corrected.py`
  * **Figure 11**: `python3 src/Sec-05_pass-through/Fig-11_WC-GPU-temp-pwr-util.py`

Each script will output a pdf file into `figures/` directory.

## 5\. Description of Experiments (for informational purposes)

The `experiments/` directory contains the Python scripts and configurations we used to conduct our experiments. While reproducing these experiments is not the goal of this evaluation, we provide them for transparency. These scripts handle:

  * The deployment and execution of workloads (Blender, YOLO, etc.).
  * The monitoring and collection of power consumption data (via DCGM, IPMI, etc.).
  * The configuration of different GPU sharing modes (MIG, Time-slicing).
