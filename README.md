# Urban Typology and Amenity Classification Using Deep CNN

This project investigates how deep convolutional neural networks (CNNs) can be used to classify urban land typologies and detect amenities from Sentinel-2 satellite imagery using automated labels derived from OpenStreetMap.

## Table of Contents
- [Introduction](#introduction)
- [Data Sources](#data-sources)
- [Specification](#specification)
- [Programming](#programming)
- [Methods](#methods)
- [Results](#results)
- [Discussion and Visuals](#discussion-and-visuals)
- [Limitations and Future Work](#limitation-and-future-work)

## Introduction

Urban land classification is crucial for urban planning, sustainability analysis, and geospatial monitoring. This project aims to classify satellite image tiles by typology (residential, industrial, etc.) and detect amenities (parks, schools, etc.) using a custom-trained deep CNN. The task involves geospatial data extraction, automated labeling via spatial intersection, and supervised CNN training.

## Data Sources

- **Satellite Imagery**: Sentinel-2 Level-2A RGB data accessed via Google Earth Engine (June–August 2023).
- **Labels**: Automatically generated using OpenStreetMap data for `landuse`, `amenity`, `leisure`, `tourism`, and `natural` tags, using the `osmnx` and `geopandas` Python libraries.

Each 512x512m tile was extracted as a GeoTIFF image and intersected with OSM features to generate single typology labels and multi-label amenity presence.

## Specification

- **Tile Extraction**: ~4200 tiles for the Chicago metro area, at 10m resolution.
- **Labeling**: Spatial intersection used to assign each tile one dominant land use and four binary amenity labels.
- **Model**: A custom CNN with shared convolutional layers and two output heads:
  - Softmax for typology classification
  - Sigmoid for multi-label amenity detection

![CNN Diagram](CNN_typography_flow_diagram.png)

## Programming

The code is written in Python using:
- `earthengine-api` for image access
- `rasterio` for GeoTIFF handling
- `torch` and `torchvision` for model development
- `sklearn`, `matplotlib`, and `seaborn` for evaluation and visualization

All processes, including dataset generation, preprocessing, CNN setup, training and evaluation is within this [jupyter notebook](dataset_generation_and_CNN.ipynb)

## Methods

The project follows a multi-stage pipeline:

1. **Tile Generation**: Sentinel-2 RGB imagery from Google Earth Engine is clipped into 512x512m tiles over Chicago, saved locally at 10m resolution.
2. **Automated Labeling**: Using `osmnx`, land use and amenity labels are assigned to each tile based on spatial intersection with OpenStreetMap vector data.
3. **Dataset Preprocessing**: Images are normalized and center-cropped to 48×48 pixels, and paired with typology and multi-label amenity tags.
4. **Model Training**: A custom CNN is trained jointly on typology (single-label) and amenity (multi-label) outputs, using cross-entropy and binary cross-entropy losses.

## Results

- **Typology accuracy**: ~56% on validation set
- **Amenity ROC AUC**:
  - Water: 0.89
  - Park: 0.77
  - School: 0.71
  - Museum: 0.69

![Confusion Matrix](confusion_matrix_typology.png)

Confusion matrix shows strong performance for common classes like residential and industrial, with lower recall on rare typologies.

## Discussion and Visuals

![CNN Architecture](CNN_architecture.png)

![Amenity ROC Curves](roc_curves_amenities.png)

- **Training**: Conducted on CPU; 10 epochs took ~45 minutes.
- **Challenges**: Label imbalance and low-resolution imagery made rare class detection difficult.

The model performs best on well-represented classes and easily detectable features. Label imbalance and semantic ambiguity in OSM data present challenges, which are discussed in the thesis [here](Hamdi_Kucukengin_CNN_Topography.pdf).

## Limitation and Future Work

- **Label Imbalance**: Rare land-use classes like quarry or railway were underrepresented, leading to low F1 scores.
- **Model Simplicity**: The architecture was intentionally lightweight, limiting its ability to capture finer details.
- **Next Steps**:
  - Experiment with deeper CNNs or pretrained models
  - Apply to other cities
  - Use self-supervised learning for feature extraction