# Master Thesis – Tree Crown Detection and Downstream Forest Metrics

This repository contains all scripts used for the experimental workflows of the master thesis
"Evaluation of Tree Crown Delineation Models with Subsequent Species, Vitality, and Allometric Classification".

The code is organized into two main processing pipelines and accompanying R-based statistical analyses.

---

## Pipeline 1 – Tree Crown Detection

Pipeline 1 performs individual tree crown delineation from UAV-derived RGB imagery using three
deep learning frameworks:

- Detectree2 (Mask R-CNN)
- U-Net–based Tree Crown Delineation (TCD)
- DeepTrees

### Training
Training scripts were adapted from the original frameworks to:
- log loss curves and convergence statistics
- enable consistent comparison across models
- export intermediate metrics for downstream R-based evaluation

Scripts:
- `detectree2_train_instances_v4.3.1.py`
- `training_tcd_rgb_v2.1.py`

### Inference
Inference scripts generate crown polygons (GPKG) used as input for Pipeline 2.
Minor modifications were applied to:
- standardize output formats
- ensure compatibility with feature extraction scripts

Scripts:
- `detectree2_infer_instances_to_gpkg_for_v4.3.1.py`
- `inference.py`

---

## Pipeline 2 – Feature Extraction, Classification, and Allometric Modelling

Pipeline 2 operates at the level of detected tree crown polygons.

### Feature Extraction
For each crown polygon, spectral, textural, geometric, and height-related features are extracted.
Feature extraction includes:
- descriptive statistics (mean, sd)
- percentile-based features
- higher-order moments (skewness, kurtosis)
- vegetation indices (NDVI, GNDVI, SAVI, VARI, EVI, MSI, NDWI)
- GLCM texture metrics

Script:
- `Terminator_extractor_2.py`

### Classification
Supervised classification is applied for:
- tree species
- tree condition

Scripts:
- `art_zustand_klassifikator.py`
- `chm_klassifikator.py`

### Integration
Final integration combines outputs from multiple detection methods and supports
both standard tiles and the Marteloscope reference plot.

Scripts:
- `multimethod_integration_v7.py`
- `multimethod_integration_v7_martel.py`

---

## R-Based Statistical Analysis

All quantitative analyses and figure generation were performed in R.

Included analyses:
- training and validation diagnostics
- inference comparison across models
- allometric DBH and height evaluation
- species-specific scatter analyses
- Marteloscope vs UAV tile comparison

Scripts are organized by topic under `r_analysis/`.

---

## Reproducibility Notes

- Python and R scripts were executed sequentially:
  1. Pipeline 1 training
  2. Pipeline 1 inference
  3. Pipeline 2 feature extraction and modelling
  4. R-based statistical evaluation
- Environment details are provided in the `environment/` directory.

---

## License
This repository is provided for academic and reproducibility purposes.

