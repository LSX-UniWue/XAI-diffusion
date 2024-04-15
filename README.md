# Generative Inpainting for Shapley-Value-Based Anomaly Explanation

This repository contains the code for the paper "Generative Inpainting for Shapley-Value-Based Anomaly Explanation" published as part of The second World Conference on eXplainable Artificial Intelligence (xAI 2024).

[Tritscher, Julian, et al. "Generative Inpainting for Shapley-Value-Based Anomaly Explanation" 2nd World Conference on eXplainable Artificial Intelligence - xAI (2024).]

## Data

### CIDDS-001
This repository uses the CIDDS-001 dataset:

[Ring, Markus, et al. "Technical Report CIDDS-001 data set." J. Inf. Warfare 13 (2017).]

Fully preprocessed data can be downloaded [here](https://drive.google.com/drive/folders/1yMaciiXPSbqnJrJHRrLeoiwMIiCzRF_I?usp=sharing) and needs to be placed into `data/cidds/data_prep`. The original dataset can be downloaded from https://www.hs-coburg.de/cidds.

Ground truth explanations are found under `data/cidds/data_raw`.

### ERP
The ERP fraud detection dataset is available at https://professor-x.de/erp-fraud-data (directory joint_datasets) and needs to be placed in `data/erp_fraud`.


## TabDDPM Training
TabDDPM Code is based on the official implementation: https://github.com/yandex-research/tab-ddpm

### Hyperparameter search
Parameter studies can be conducted using `tune_ddpm.py` with arguments for 'ds_name' (e.g. `cidds`) and 'prefix' (e.g. `ddpm_cidds`).

Config files for the best models found during our hyperparameter search can be found under `exp/cidds/ddpm_cidds_best` and `exp/erp_normal2/ddpm_erp_best`.

### Training single models
Training of a single TabDDPM model, given a config file, can be conducted using `pipeline.py` with arguments for 'config' (e.g. exp/cidds/ddpm_cidds_best/config.toml) and setting the argument '--train'.

## XAI Evaluation

Inpainting is based on the official RePaint implementation: https://github.com/andreas128/RePaint

`erp_xai.py` and `cidds_xai.py` contain the code for generating SHAP explanations for the ERP and CIDDS-001 datasets respectively.
Both take the arguments 'conf_path' (e.g. `repaint/confs/tabddpm_erp.yml`) and 'job_name'.

Both scripts need a prior trained TabDDPM model to work.
The path to the corresponding model.pt file needs to be indicated in the .yml file under `repaint/confs`.

The evaluated fully trained anomaly detectors are located in `xai/outputs/models`.

**Note**: Additional setup is required for running SHAP with *optimized* and *diffusion* replacement data.
To integrate the optimization procedure directly within kernel-SHAP,
this implementation requires to manually override the `shap/explainer/_kernel.py` script within the SHAP package.
For this, either override the contents of `shap/explainer/_kernel.py` entirely
with the backup file provided in `xai/backups/shap_kernel_backup.py`
or add the small segments marked with `# NEWCODE` within `xai/backups/shap_kernel_backup.py` in the
original library file of `shap/explainer/_kernel.py`.