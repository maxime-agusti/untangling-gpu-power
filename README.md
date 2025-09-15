# Artifact for the paper: Untangling GPU Power Consumption: Job-Level Inference in Cloud Shared Settings

This document describes the artifact submitted for the research paper "Untangling GPU Power Consumption: Job-Level Inference in Cloud Shared Settings," accepted at the EuroSys 2026 conference.

This artifact aims to obtain the following badges:

  * **Artifact Available**: The source code, data, and documentation are permanently accessible.
  * **Artifact Functional**: The artifact is well-documented, and the provided code can be used to generate the figures and results presented in the paper.

## 1\. Overview

The artifact contains the necessary scripts and tools to process the raw experimental data and generate the figures presented in our paper.

**Important Notice on Reproducibility**: Due to the requirement of highly specific hardware (including precise multi-GPU configurations and IPMI power measurement systems) that is generally unavailable to evaluators, this artifact **is not intended for the reproduction of the experiments** described in the paper. However, we provide the complete source code used for our experiments and for generating the figures to ensure maximum transparency.

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

To use this artifact, you will need a general-purpose computer running **Ubuntu 22.04** (or later) or **macOS 13** (or later).

You will also need **Python 3.11** or a newer version. The Python dependencies can be installed using `pip`.

1.  **Clone the Git repository** for the artifact or unzip the archive.
2.  **Install the dependencies** using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## 4\. Resource Requirements

This section details the computational resources required to run the figure generation scripts.

* **Storage**: A total of **2.0 GiB** of disk space is required to store the repository, datasets, dependencies, and generated figures.
* **CPU**: All figure generation scripts are **single-threaded**. A single CPU core is therefore sufficient.
* **Memory**: The memory consumption varies by script. The most demanding script requires approximately **1.8 GiB of RAM**.

## 5\. User Guide: Generating the Figures

The scripts to generate each figure are located in the `src/` directory. Each script corresponds to one or more figures in the paper.

For example, to generate **Figure 2** from the paper:

```bash
python3 src/Sec-03_temporally-shared/Fig-02_TS-power.py
```

This script requires **296 MiB** of memory and takes **~3 seconds** to run. This was measured on an **Apple M3 Pro** processor and may vary depending on your hardware configuration.

Similarly for the other figures:


* **Figure 3**:
    * **Command**: `python3 src/Sec-03_temporally-shared/Fig-03_TS-perf-power.py`:
    * **Memory**: 1168 MiB
    * **Execution Time**: ~7 seconds
* **Figure 5**:
    * **Command**: `python3 src/Sec-04_spatially-shared/Fig-05_MIG-GI-power-all.py`:
    * **Memory**: 574 MiB
    * **Execution Time**: ~5 seconds
* **Figure 6**:
    * **Command**: `python3 src/Sec-04_spatially-shared/Fig-06_MIG-driver.py`:
    * **Memory**: 432 MiB
    * **Execution Time**: ~3 seconds
* **Figure 7**:
    * **Command**: `python3 src/Sec-04_spatially-shared/Fig-07_MIG-GI-bench-all.py`:
    * **Memory**: 1422 MiB
    * **Execution Time**: ~10 seconds
* **Figure 8**:
    * **Command**: `python3 src/Sec-05_pass-through/Fig-08_PT-pearson-corr-ipmi.py`:
    * **Memory**: 1173 MiB
    * **Execution Time**: ~5 seconds
* **Figure 9**:
    * **Command**: `python3 src/Sec-05_pass-through/Fig-09_PT-density-4A100-4states-corrected.py`:
    * **Memory**: 1760 MiB
    * **Execution Time**: ~6 minutes
* **Figure 10**:
    * **Command**: `python3 src/Sec-05_pass-through/Fig-10_PT-density-8A100-2states-corrected.py`:
    * **Memory**: 944 MiB
    * **Execution Time**: ~1 minute and 30 seconds
* **Figure 11**:
    * **Command**: `python3 src/Sec-05_pass-through/Fig-11_WC-GPU-temp-pwr-util.py`:
    * **Memory**: 705 MiB
    * **Execution Time**: ~4 seconds


Each script will output a pdf file into `figures/` directory.

## 6\. Description of Experiments (for informational purposes)

The `experiments/` directory contains the Python scripts and configurations we used to conduct our experiments. While reproducing these experiments is not the goal of this evaluation, we provide them for transparency. These scripts handle:

  * The deployment and execution of workloads (Blender, YOLO, etc.).
  * The monitoring and collection of power consumption data (via DCGM, IPMI, etc.).
  * The configuration of different GPU sharing modes (MIG, Time-slicing).
