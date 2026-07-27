# Supporting Data and Analysis Code for
# “Deciphering Majorana Zero Modes in Topological Superconductor FeTe0.55Se0.45 with Machine-Learning-Assisted Spectral Deconvolution”

## Dataset title

Supporting Data and Analysis Code for “Deciphering Majorana Zero Modes in Topological Superconductor FeTe0.55Se0.45 with Machine-Learning-Assisted Spectral Deconvolution”

## Authors

Jewook Park, Hoyeon Jeon, Dongwon Shin, Guannan Zhang, Michael A. McGuire, Brian C. Sales, and An-Ping Li

Corresponding authors:

- Jewook Park, Oak Ridge National Laboratory, parkj1@ornl.gov
- An-Ping Li, Oak Ridge National Laboratory, apli@ornl.gov

## Dataset DOI

[DOI TO BE INSERTED UPON PUBLICATION]

## Related manuscript

This dataset supports the manuscript:

“Deciphering Majorana Zero Modes in Topological Superconductor FeTe0.55Se0.45 with Machine-Learning-Assisted Spectral Deconvolution”

The manuscript reports millikelvin scanning tunneling microscopy/spectroscopy (STM/S) measurements and machine-learning-assisted spectral deconvolution of vortex-core local density of states (LDOS) data in FeTe0.55Se0.45.

## Purpose of this dataset

This dataset provides the raw STM/S grid spectroscopy data and analysis scripts used to support the figures and conclusions of the associated manuscript. The data and code are intended to allow readers to inspect the original spectroscopy dataset, reproduce the major analysis workflow, and understand how zero-bias-peak-related spectral components were extracted from complex in-gap states.

The analysis workflow includes:

1. Loading STM/S grid spectroscopy data.
2. Converting scanning probe microscopy data into analysis-ready data structures.
3. Visualizing topography and spectroscopy data.
4. Performing peak detection and multi-peak Lorentzian fitting.
5. Constructing spectral feature sets from fitted peak parameters.
6. Applying dimensionality reduction and unsupervised clustering.
7. Reconstructing zero-bias-peak-related LDOS contributions.
8. Generating figure panels used in the manuscript.

## Files included

The dataset contains the following main files:

```text
README.md
Anaconda_FTS_fig1_2023data_2026_0727.py
FTS_fig_2T003_2026_0727.py
Grid Spectroscopy(X0.01)_2T_40mK_003.3ds
```

## File descriptions

### `Grid Spectroscopy(X0.01)_2T_40mK_003.3ds`

Raw grid spectroscopy data acquired by millikelvin STM/S on FeTe0.55Se0.45.

This file contains spatially resolved tunneling spectroscopy data used to construct grid LDOS maps and zero-bias conductance maps.

Relevant experimental conditions include:

- Sample: FeTe0.55Se0.45 single crystal
- Cleaving temperature: approximately 83 K
- Measurement temperature: 40 mK
- Tip: normal-metal PtIr tip
- Magnetic field: out-of-plane magnetic field, including B = 2 T condition used for vortex-core spectroscopy
- Measurement type: grid spectroscopy / spatially resolved dI/dV spectroscopy

### `FTS_fig_2T003_2026_0727.py`

Python analysis script, exported in Jupytext-compatible format, used for analysis and figure generation related to the 2 T grid spectroscopy dataset.

This script includes procedures for:

- Loading and processing STM/S grid spectroscopy data.
- Visualizing zero-bias conductance maps.
- Performing spectral analysis of vortex-core LDOS data.
- Extracting and plotting fitted spectral features.
- Preparing figure panels for the associated manuscript.

### `Anaconda_FTS_fig1_2023data_2026_0727.py`

Python analysis script, exported in Jupytext-compatible format, used for STM/S data processing, environment setup, visualization, and figure-generation steps related to the FeTe0.55Se0.45 analysis.

This script includes:

- Python package import and environment checks.
- STM/S data handling routines.
- Visualization of topography and spectroscopy datasets.
- Processing steps used in the manuscript figure workflow.

## Software environment

The scripts were developed and executed in a Python/Jupyter environment. They are saved as Jupytext-compatible `.py` files and may be opened either as plain Python scripts or paired with Jupyter/Jupytext workflows.

The following Python packages are used or expected by the analysis scripts:

```text
numpy
scipy
xarray
pandas
matplotlib
plotly
ipywidgets
scikit-image
scikit-learn
umap-learn
hdbscan
lmfit
nanonispy
jupyter
jupyterlab
jupytext
```

Depending on the local Python installation, additional packages may be required by SPMpy-related analysis functions.

A typical environment can be prepared with conda or mamba, for example:

```bash
conda create -n zbp_fts python=3.11
conda activate zbp_fts
conda install -c conda-forge numpy scipy pandas xarray matplotlib scikit-image scikit-learn jupyterlab jupytext ipywidgets plotly lmfit umap-learn hdbscan
pip install nanonispy
```

Package versions used during manuscript preparation may differ from the latest available versions. For exact reproduction, users should review the package installation and import sections near the beginning of the Python scripts.

## Suggested usage

1. Download all files into a single working directory.
2. Create and activate a Python environment with the required packages.
3. Open the `.py` scripts in JupyterLab using Jupytext, or run selected code blocks interactively.
4. Confirm that the raw `.3ds` file is located in the same directory expected by the scripts.
5. Execute the analysis cells in order, beginning with package imports and data loading.
6. Reproduce the spectroscopy maps, Lorentzian fitting results, clustering analysis, and figure-generation steps.

Example:

```bash
conda activate zbp_fts
jupyter lab
```

Then open:

```text
FTS_fig_2T003_2026_0727.py
Anaconda_FTS_fig1_2023data_2026_0727.py
```

as Jupytext notebooks or Python scripts.

## Data format notes

The `.3ds` file is a grid spectroscopy file generated by an STM/S controller. It contains spatially resolved spectroscopy data, including tunneling conductance as a function of bias voltage and spatial position.

The analysis scripts convert the spectroscopy data into Python data structures suitable for multidimensional analysis, including xarray-based representations. The scripts then perform visualization, fitting, clustering, and reconstruction of selected spectral components.

## Reproducibility notes

The scripts are provided in the form used during manuscript preparation. They may include exploratory cells, plotting routines, and environment checks. Users interested in reproducing specific manuscript figures should follow the relevant sections of the scripts corresponding to each figure.

Some intermediate results may depend on user-selected paths, plotting parameters, package versions, or interactive widget settings. Users should verify local file paths and package availability before running the scripts.

## Update notes

On 2026-07-27, following the figure-generation work carried out on the actual experimental dataset, the code comments in both analysis scripts were reviewed and updated for clarity and reuse (local file paths were anonymized and Korean-language comments were translated to English); the underlying analysis logic and figure outputs are unchanged.

Since the completion of this manuscript's analysis, the SPMpy project has been undergoing a broader redefinition, including a major update to its data-processing pipeline; the scripts archived here reflect the version of the code used for this manuscript and predate that redevelopment.

## Relationship to GitHub repository

A public version of the analysis code is also available at:

```text
https://github.com/jewook-park/ZBPs_in_FTS
```

The files deposited with this DOI record provide a citable archival snapshot associated with the manuscript. If the GitHub repository is updated after publication, the DOI-deposited files should be regarded as the reference version corresponding to the published article.

## Funding and facility acknowledgment

This research was supported by the Center for Nanophase Materials Sciences (CNMS), a U.S. Department of Energy, Office of Science User Facility at Oak Ridge National Laboratory.

Crystal growth and characterization were supported by the U.S. Department of Energy, Office of Science, Basic Energy Sciences, Materials Sciences and Engineering Division.

## DOE notice

This manuscript has been authored by UT-Battelle, LLC, under Contract No. DE-AC05-00OR22725 with the U.S. Department of Energy. The United States Government retains and the publisher, by accepting the article for publication, acknowledges that the United States Government retains a non-exclusive, paid-up, irrevocable, world-wide license to publish or reproduce the published form of this manuscript, or allow others to do so, for United States Government purposes. The Department of Energy will provide public access to these results of federally sponsored research in accordance with the DOE Public Access Plan.

## License and reuse

Please refer to the license and reuse terms specified in the Constellation DOI record and/or the associated GitHub repository. For questions about reuse of the raw STM/S data or analysis scripts, please contact the corresponding authors.

## Contact

For questions about the dataset or analysis workflow, please contact:

Jewook Park  
Center for Nanophase Materials Sciences  
Oak Ridge National Laboratory  
parkj1@ornl.gov
