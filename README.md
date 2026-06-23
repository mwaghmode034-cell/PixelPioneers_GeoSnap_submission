# Geo Snap — PixelPioneers Submission

**SOI × Cosmosoc — Land-Use Classification & Explainability from Space**

**Team PixelPioneers**
Priyanshu • Shravan • Mahesh • Chinmay
1st Year B.Tech , IIT Dharwad

---

## Project Overview

Geo Snap is a deep learning project for **land-use classification from satellite imagery** using the **EuroSAT** dataset in two imaging modalities:

* **RGB images (3 bands)** – what a camera sees.
* **Multispectral Sentinel-2 images (13 bands)** – information beyond human vision.

The project investigates whether additional spectral information improves land-use classification performance and provides explainability and environmental insights from satellite imagery.

---

## Tasks Completed

| Task        | Description                                     |
| ----------- | ----------------------------------------------- |
| **Task 1A** | RGB Land-Use Classification                     |
| **Task 1B** | Multispectral (13-band) Land-Use Classification |
| **Task 2**  | Explainability and Model Interpretation         |
| **Task 3**  | Environmental Insights from Spectral Data       |
| **Bonus**   | Interactive Streamlit Web Application           |

---

# Results Summary

| Model           | Modality                | Validation Accuracy |
| --------------- | ----------------------- | ------------------- |
| EfficientNet-B2 | RGB (3-band)            | **97.33%**          |
| EfficientNet-B2 | Multispectral (13-band) | **98.79%**          |

The multispectral model improves accuracy by **+1.46 percentage points** over RGB alone.

---

# Features

## Classification

* RGB land-use classification using EfficientNet-B2.
* 13-band Sentinel-2 multispectral classification.
* Top-3 prediction probabilities.
* Full class probability distribution chart.

## Explainability

* Grad-CAM visual explanations.
* Band-importance analysis using occlusion.
* Confusion matrix and error analysis.
* Class-wise performance evaluation.

## Environmental Insights

* NDVI computation.
* NDWI computation.
* NDBI analysis.
* Moisture Index estimation.
* Land suitability and geo-planning insights.

## Deployment

* Automatic image-type detection.
* Interactive Streamlit interface.
* Downloadable prediction report.

---

# Dataset

Dataset: **EuroSAT**

### Classes

1. AnnualCrop
2. Forest
3. HerbaceousVegetation
4. Pasture
5. River
6. Highway
7. Industrial
8. PermanentCrop
9. Residential
10. SeaLake

### Input Dimensions

* RGB: `64 × 64 × 3`
* Multispectral: `64 × 64 × 13`

---

# Repository Structure

```text
PixelPioneer_GeoSnap_submission
│
├── Deployment/
│   ├── .streamlit/
│   │   └── config.toml
│   ├── assets/
│   │   └── logo.png
│   ├── app.py
│   ├── requirements.txt
│   └── runtime.txt
│
├── models/
│   ├── efficientnet_b2_ms_final.pth
│   └── efficientnet_b2_rgb_final.pth
│
├── predictions/
│   ├── ms_predictions.csv
│   ├── ms_predictions_proba.csv
│   ├── rgb_predictions.csv
│   └── rgb_predictions_proba.csv
│
├── PixelPioneers_GeoSnap.ipynb
├── PixelPioneers_Technical_Report.pdf
└── README.md
```

---

# Model Architecture

Both classifiers use:

* **EfficientNet-B2 backbone**
* AdamW optimizer
* CosineAnnealingWarmRestarts scheduler
* Label smoothing
* Gradient clipping
* Early stopping
* Dropout regularization

For the multispectral model:

* First convolution expanded from **3 → 13 channels**
* RGB pretrained weights reused for initialization.

---

# Environment and Dependencies

### Python Version

```text
Python 3.10+
```

### Main Libraries

```bash
pip install torch torchvision tifffile numpy pandas matplotlib seaborn scikit-learn Pillow rasterio streamlit
```

---

# Running the Notebook (Google Colab)

The notebook was developed and tested on:

* Google Colab 
* NVIDIA T4 GPU
* Python 3.10

---

## Step 1: Open Notebook

Open: 
```text
 https://colab.research.google.com/drive/1URYcL6djkYvssyWp6nmNKN3spcLTYpeG
```
in Google Colab.

Or Upload PixelPioneers_GeoSnap.ipynb to Drive

---

## Step 2: Upload Dataset

Place:

```text
EuroSAT_Dataset.zip
```

inside:

```text
drive/MyDrive/
```

---

## Step 3: Mount Google Drive

The notebook automatically mounts Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## Step 4: Run All Cells

The notebook is organized into sections:

### Sections 0–1

* Environment setup
* Dataset extraction
* Dependency installation

### Sections 2–6

* RGB model training
* Validation
* Checkpoint saving

### Sections 7–9

* Multispectral model training
* Validation
* Checkpoint saving

### Section 10

* Generate prediction CSV files.

### Sections 11–16

* Confusion matrices
* Grad-CAM
* Band importance analysis
* Error analysis
* Spectral signatures

### Sections 17–18

* NDVI analysis
* NDWI analysis
* NDBI analysis
* Environmental insights
* Final summary

---

## Expected Runtime

On Google Colab T4 GPU:

| Task                   | Runtime     |
| ---------------------- | ----------- |
| RGB Training           | ~13 minutes |
| Multispectral Training | ~10 minutes |
| Total                  | ~25 minutes |

---

# Reproducing Results Using Saved Models

## RGB Model

```python
ckpt = torch.load(
    'models/efficientnet_b2_rgb_final.pth',
    map_location=device
)

model.load_state_dict(
    ckpt['model_state_dict']
)
model.eval()
```

---

## Multispectral Model

```python
ckpt = torch.load(
    'models/efficientnet_b2_ms_final.pth',
    map_location=device
)

band_mean = ckpt['band_mean']
band_std = ckpt['band_std']
```

Normalize the input:

```python
img = (
    raw_13band_tensor
    - band_mean.view(-1,1,1)
) / (
    band_std.view(-1,1,1)
    + 1e-6
)
```

---

# Prediction Files

## Task 1A

```text
predictions/rgb_predictions.csv
predictions/rgb_predictions_proba.csv
```

## Task 1B

```text
predictions/ms_predictions.csv
predictions/ms_predictions_proba.csv
```

---

# Running the Web Application Locally

Move into the deployment directory:

```bash
cd Deployment
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# Web Application Features

* Automatic RGB / Multispectral image detection.

* Supports:

  * JPG
  * JPEG
  * PNG
  * BMP
  * WEBP
  * TIF
  * TIFF

* Top-3 predictions.

* Class probability visualization.

* NDVI computation.

* NDWI computation.

* Moisture Index computation.

* Downloadable prediction report.

---

# Live Demo

**Streamlit Deployment**

https://geosnap-pixelpioneers.streamlit.app/

---

# Environmental Insights

Using Sentinel-2 spectral information, the project derives:

## NDVI

* Vegetation health monitoring
* Crop stress estimation

## NDWI

* Water-body identification
* Moisture monitoring

## NDBI

* Urban expansion analysis
* Built-up area assessment

## Applications

* Agriculture monitoring
* Drought assessment
* Land-use planning
* Climate monitoring
* Geo-planning and land suitability analysis

---

# Deliverables

| File                                 | Description                 |
| ------------------------------------ | --------------------------- |
| `PixelPioneers_GeoSnap.ipynb`        | Full pipeline notebook      |
| `PixelPioneers_Technical_Report.pdf` | Technical report            |
| `efficientnet_b2_rgb_final.pth`      | Best RGB model              |
| `efficientnet_b2_ms_final.pth`       | Best multispectral model    |
| `rgb_predictions.csv`                | RGB predictions             |
| `ms_predictions.csv`                 | Multispectral predictions   |
| `rgb_predictions_proba.csv`          | RGB probabilities           |
| `ms_predictions_proba.csv`           | Multispectral probabilities |
| `Deployment/`                        | Streamlit web application   |

---

# Technology Stack

* Python
* PyTorch
* EfficientNet-B2
* Streamlit
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Rasterio
* Pillow
* tifffile

---

# Acknowledgements

* SOI × Cosmosoc
* EuroSAT Dataset
* Sentinel-2 Mission
* IIT Dharwad

---

# Team PixelPioneers

**Land-Use Classification & Explainability from Space**

SOI × Cosmosoc Hackathon Submission.
