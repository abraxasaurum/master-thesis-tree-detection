# Data Structure Overview

This document describes the expected input and output data structure used by the scripts in this repository.
No raw data are included.

---

## Input Data

### UAV RGB Imagery
- Format: GeoTIFF
- Resolution: ~1 cm
- Bands: Red, Green, Blue
- Usage: Input for Pipeline 1 (tree crown detection)

### NIR Imagery
- Format: GeoTIFF
- Resolution: Resampled to match RGB
- Usage: Feature extraction in Pipeline 2

### Canopy Height Model (CHM / nDSM)
- Format: GeoTIFF
- Resolution: Matched to RGB
- Usage: Height-based feature extraction in Pipeline 2

### Ground Truth Data
- Format: GeoPackage (.gpkg) or CSV
- Content:
  - Manually delineated tree crown polygons
  - Species labels
  - Tree condition labels
  - Field-measured DBH and height (where available)

---

## Intermediate Data

### Detection Outputs (Pipeline 1)
- Format: GeoPackage (.gpkg)
- Content:
  - Predicted tree crown polygons
  - Detection method identifier
  - Confidence scores (where applicable)

### Feature Tables (Pipeline 2)
- Format: CSV
- Content:
  - One row per detected tree
  - Extracted spectral, textural, geometric, and height features

---

## Output Data

### Classification Results
- Format: GeoPackage (.gpkg) and CSV
- Content:
  - Tree species predictions
  - Tree condition predictions

### Allometric Estimates
- Format: CSV
- Content:
  - Predicted DBH
  - Predicted tree height (where applicable)
  - Associated crown geometry metrics

---

## Notes
- File paths and exact filenames are defined within individual scripts.
- All outputs were generated using the scripts provided in this repository.

