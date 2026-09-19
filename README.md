# 🌍 Climate Temperature Forecasting using ConvLSTM

[![Project Status](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![Dataset](https://img.shields.io/badge/Dataset-ERA5-green.svg)
![Deep Learning](https://img.shields.io/badge/Model-ConvLSTM-orange.svg)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-brightgreen?logo=streamlit)](https://climate-forecasting-9mfng65j47sierrx5jm6pe.streamlit.app/)

---

# 📖 Climate Temperature Forecasting

A deep learning framework for **next-day temperature forecasting** using the **ERA5 Climate Reanalysis Dataset** and **Convolutional Long Short-Term Memory (ConvLSTM)** networks.

The project provides a complete end-to-end pipeline for:

- 🌍 Climate data acquisition
- 🛰 Data preprocessing
- 📊 Sequence generation
- 🧠 Deep learning model training
- 📈 Model evaluation
- 🌡 Temperature prediction
- 📉 Performance visualization

The framework is modular, reproducible, and configurable, making it suitable for research and real-world climate forecasting applications.

---

# 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Workflow](#-workflow)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Data Pipeline](#-data-pipeline)
- [Available Models](#-available-models)
- [Usage](#-usage)
- [Troubleshooting](#-troubleshooting)
- [Dependencies](#-dependencies)
- [Roadmap](#-roadmap)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)

---

# 🎯 Project Overview

Climate temperature prediction is essential for agriculture, disaster management, environmental monitoring, renewable energy planning, and weather forecasting.

Traditional numerical weather prediction methods require significant computational resources and may struggle to accurately capture localized temperature variations over space and time.

This project introduces a **ConvLSTM-based spatiotemporal deep learning framework** that learns both spatial and temporal dependencies directly from historical ERA5 climate observations.

The framework automates the complete forecasting pipeline, including:

- ERA5 climate data download
- Data preprocessing
- Geographic region extraction
# 📊 Results

## Research Status

The repository is currently **not publication-ready**. The raw ERA5 files and
the regenerated regional subset have finite `t2m` values, but the downstream
daily NetCDF rebuild is still blocked by a Windows NetCDF/HDF5 file-write
problem. Until daily files, normalization statistics, and sequence tensors are
rebuilt in one successful run, no benchmark value should be described as a
final paper result.

The authoritative reviewer audit is [research.md](research.md). It records the
reviewer-by-reviewer status and separates verified facts from provisional
artifacts.

## Verified Dataset Definition

- Raw ERA5 coverage: 2019-01-01 00:00 through 2025-12-31 23:00
- Variable: hourly 2-meter air temperature (`t2m`), in Kelvin
- Study region: 5°N to 35°N and 65°E to 100°E
- Grid: 121 × 141 at 0.25° resolution
- Configured split: train 2019–2023, validation 2024, test 2025
- Input window: 7 daily fields; forecast horizon: 1 day
- Aggregation: daily maximum, as implemented in `preprocessing/resample_time.py`

## Provisional Artifact Metrics

The following values are preserved from the previously saved checkpoint and
test arrays. They are reproducible artifact values, not final research metrics,
because the saved normalization artifact currently contains non-finite
statistics and the raw-derived preprocessing chain has not completed
successfully.

| Model | Test RMSE | Test MAE | Test R² | Test correlation |
|-------|------------|----------|---------|------------------|
| Persistence | 1.1277 | 0.9304 | 0.9794 | 0.9897 |
| Saved Conv3D encoder-decoder checkpoint | 1.8823 | 1.6127 | 0.9425 | 0.9720 |

The class named `ConvLSTMModel` is implemented as a Conv3D encoder-decoder,
not an LSTM-cell model. It has two 3 × 3 × 3 encoder convolutions with 32
hidden channels, one decoder convolution, ReLU activations, and 29,441
trainable parameters. These artifact metrics do not establish superiority over
persistence. Independent station or operational forecast validation is not
included.

## Regenerating Reviewer Outputs

After the preprocessing artifacts are successfully rebuilt and verified, run:

```bash
python -m preprocessing.subset_region
python -m preprocessing.split_by_year
python -m preprocessing.resample_time
python -m preprocessing.normalize
python -m preprocessing.create_sequences
python reviewer_analysis.py
python run_baseline_experiments.py --epochs 1 --batch-size 1 --device cpu
```

The analysis script writes metadata, metrics, seasonal and regional tables,
and spatial error maps to `outputs/reviewer_revision/`. Do not update the
metrics table above with new values until `mean_std.json`, all NumPy arrays,
and all sequence tensors pass finite-value checks.

### Current Reviewer Closure

- Reviewer 1: study region and architecture are **DONE**; dates, leakage,
  normalization, and figures remain **PARTIAL** pending the rebuild.
- Reviewer 2: t2m scope and metric definitions are **DONE**; baselines and
  seasonal/regional/spatial analysis remain **PARTIAL**.
- Reviewer 3: independent validation is **NOT DONE**; citations, literature
  comparison, Figure 3 explanation, and conclusion moderation remain
  **PARTIAL**.

The full evidence and unresolved items are maintained in
[research.md](research.md).
| Provider | ECMWF (Copernicus C3S) |
| Variable | 2-meter Air Temperature (`t2m`) |
| Temporal Resolution | Hourly (Resampled to Daily) |
| Spatial Coverage | Global (Region Subset) |
| Data Format | NetCDF (`.nc`) |
| Default Study Region | India |
| Download Method | CDS API |

> **Note:** The ERA5 dataset is not included in this repository because of its large size. Data can be downloaded using the provided preprocessing scripts.

---

# 🔬 Methodology

The forecasting framework is based on a **Convolutional Long Short-Term Memory (ConvLSTM)** architecture.

Unlike traditional LSTMs, ConvLSTM replaces fully connected operations with convolutional operations, enabling the network to preserve spatial information while simultaneously learning temporal dependencies.

The overall methodology consists of:

1. Download ERA5 climate data
2. Select the target geographic region
3. Convert hourly observations into daily maxima (as implemented in `preprocessing/resample_time.py`)
4. Normalize temperature values
5. Generate sliding-window sequences
6. Train the ConvLSTM model
7. Predict next-day temperature maps
8. Evaluate prediction accuracy
9. Visualize forecasting results

## Why 2-meter air temperature only?

The project focuses on `t2m` because it is the single target variable used by
the download, preprocessing, tensor-generation, model, dashboard, and
evaluation paths. Keeping one spatially gridded target makes the current study
a focused test of next-day spatiotemporal temperature prediction and avoids
introducing unsupported assumptions about how humidity, pressure, wind, or
precipitation should be encoded. This is a scope decision, not evidence that
additional variables are unnecessary. Multivariable inputs and independent
observations remain important future work.

---

# 📊 Results

## Current Research Status

The project is currently **not publication-ready**. The raw ERA5 files,
regional subset, and hourly train/validation/test split files have verified
finite `t2m` values. The daily resampling stage is still blocked by a Windows
NetCDF/HDF5 write or file-lock problem, so the normalized arrays, sequence
tensors, benchmarks, and reviewer figures have not been regenerated from the
raw-derived pipeline.

The authoritative reviewer audit is [research.md](research.md). It separates
verified repository facts from provisional saved artifacts and records the
reviewer-by-reviewer closure status.

## Verified Dataset Definition

- Raw ERA5 coverage: 2019-01-01 00:00 through 2025-12-31 23:00
- Variable: hourly 2-meter air temperature (`t2m`), in Kelvin
- Study region: 5°N to 35°N and 65°E to 100°E
- Grid: 121 × 141 at 0.25° resolution
- Configured split: train 2019–2023, validation 2024, test 2025
- Input window: 7 daily fields; forecast horizon: 1 day
- Aggregation: daily maximum, as implemented in `preprocessing/resample_time.py`

## Reviewer Revision Evidence

The repository includes an artifact-backed audit in
`reviewer_analysis.py`. Run it from the project root with:

```bash
python reviewer_analysis.py
```

It writes verified metadata, model metrics, seasonal and regional tables, and
latitude/longitude spatial error maps to `outputs/reviewer_revision/`.

Verified from the checked-in daily files:

- Study grid: latitude 5°N to 35°N and longitude 65°E to 100°E, with a 121 ×
      141 grid. The repository does not contain a separate scientific rationale
      for selecting this rectangle; that rationale must be added to the paper.
- The configured chronology is train 2019-2023, validation 2024, and test
      2025. Final sequence counts and target dates are not reportable until the
      daily files and normalized tensors are regenerated successfully.
- The class named `ConvLSTMModel` is implemented as a three-layer Conv3D
      encoder/decoder: two 3 × 3 × 3 convolutions with 32 hidden channels and a
      one-channel 3 × 3 × 3 decoder, ReLU after the encoder convolutions, no
      dropout, and 29,441 trainable parameters. It is not an LSTM-cell
      implementation.
- The sequence generator creates each split independently, so a sequence does
      not cross a split boundary. This must be reverified after the successful
      raw-derived rebuild.
- The current preprocessing code uses a training-split scalar z-score and
      replaces remaining non-finite values with zero. The saved normalization
      statistic `296.488...` is Kelvin-scale while the current script computes
      statistics after Kelvin-to-Celsius conversion, so the preprocessing outputs
      should be regenerated before publication.

The saved artifact checkpoint gives correlation `0.9720` and R² `0.9425`; these
are different statistics, not an inconsistency in the recalculation. On that
provisional test artifact, persistence is stronger than the checkpoint (RMSE
`1.128` versus `1.882`), so no superiority claim is justified. These values
must not be presented as final paper metrics because the current normalization
artifact contains non-finite statistics. Independent station or operational
forecast comparisons are not included in the repository.

The checked-in checkpoint supports a reproducible next-day grid forecast
evaluation, but the current evidence does not establish superiority over a
simple persistence baseline.

### Evaluation Metrics

| Model | Test RMSE | Test MAE | Test R² | Test correlation |
|-------|------------|----------|---------|------------------|
| Persistence | 1.1277 | 0.9304 | 0.9794 | 0.9897 |
| Saved Conv3D encoder-decoder checkpoint | 1.8823 | 1.6127 | 0.9425 | 0.9720 |

### Highlights

- ✅ Reproducible next-day spatial evaluation
- ✅ Seasonal, regional, and grid-cell error artifacts
- ⚠️ Persistence is stronger than the saved checkpoint on the checked-in test set
- ⚠️ Full baseline comparisons must be rerun after preprocessing succeeds
- ⚠️ Independent station or operational forecast validation is not yet available

> **Prediction maps, evaluation graphs, and additional performance metrics are available in the `outputs/` directory.**
>
> ---

# 📂 Project Structure

The project is organized into modular components for easy development, maintenance, and experimentation.

```text
climate2/
│
├── models/                    # Deep Learning Models
│   ├── convlstm.py
│   ├── cnn_lstm.py
│   ├── transformer.py
│   └── model_utils.py
│
├── preprocessing/             # Data preprocessing pipeline
│   ├── data_download/
│   ├── merge_years.py
│   ├── subset_region.py
│   ├── split_by_year.py
│   ├── resample_time.py
│   ├── normalize.py
│   ├── create_sequences.py
│   └── main.py
│
├── training/                  # Model training and evaluation
│   ├── train.py
│   ├── validate.py
│   ├── test.py
│   ├── metrics.py
│   └── losses.py
│
├── data_loader/               # PyTorch DataLoader
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── metadata/
│   └── splits/
│
├── experiments/               # Saved experiments
├── outputs/                   # Predictions & Visualizations
├── notebooks/                 # Jupyter notebooks
│
├── dashboard.py               # Streamlit Dashboard
├── predict_by_date.py
├── evaluate_model.py
├── future_predict.py
├── config.yaml
├── main.py
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

## Prerequisites

Before running the project, ensure your system has:

- Python 3.10 or later
- Git
- 8 GB RAM (16 GB recommended)
- CUDA-enabled GPU (optional but recommended)
- Copernicus Climate Data Store (CDS) API account

---

## Clone Repository

```bash
git clone https://github.com/your-username/climate2.git

cd climate2
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install --upgrade pip

pip install -r requirements.txt
```

---

## Verify Installation

```bash
python --version

python -c "import torch; print(torch.__version__)"

python -c "import xarray"
```

---

# ⚙ Configuration

All project settings are managed through a single configuration file.

```
config.yaml
```

The configuration includes:

| Category | Description |
|-----------|-------------|
| Variable | Climate variable (t2m) |
| Region | Latitude & Longitude bounds |
| Sequence Length | Number of historical days |
| Batch Size | Training batch size |
| Learning Rate | Optimizer learning rate |
| Epochs | Maximum training epochs |
| Model | ConvLSTM / CNN-LSTM / Transformer |
| Device | CUDA or CPU |

---

## Example Configuration

```yaml
variable: t2m

sequence_length: 7

training:
  batch_size: 8
  epochs: 100
  learning_rate: 0.0001

model:
  name: convlstm
  hidden_dim: 32
  num_layers: 2
```

---

# 🌍 Data Pipeline

The preprocessing pipeline converts raw ERA5 climate data into model-ready tensors.

```text
ERA5 Dataset
      │
      ▼
Download Climate Data
      │
      ▼
Merge Yearly Files
      │
      ▼
Region Selection
      │
      ▼
Train / Validation / Test Split
      │
      ▼
Daily Resampling
      │
      ▼
Normalization
      │
      ▼
Sliding Window Sequence Generation
      │
      ▼
Model Ready Dataset
```

---

## Preprocessing Steps

### 1. Download ERA5 Data

```bash
python preprocessing/data_download/download_era5.py
```

---

### 2. Merge Yearly Files

```bash
python preprocessing/merge_years.py
```

---

### 3. Extract Study Region

```bash
python preprocessing/subset_region.py
```

---

### 4. Split Dataset

```bash
python preprocessing/split_by_year.py
```

---

### 5. Daily Resampling

```bash
python preprocessing/resample_time.py
```

---

### 6. Normalize Data

```bash
python preprocessing/normalize.py
```

---

### 7. Generate Sequences

```bash
python preprocessing/create_sequences.py
```

---

### Complete Preprocessing Pipeline

Instead of executing every step individually:

```bash
python preprocessing/main.py --preprocess
```

---

# 🧠 Available Models

The project supports multiple deep learning architectures for spatiotemporal forecasting.

---

## 1. ConvLSTM (Recommended)

**Architecture**

Convolutional Long Short-Term Memory

**Advantages**

- Learns spatial and temporal dependencies simultaneously
- High prediction accuracy
- Efficient training
- Ideal for temperature forecasting

**Configuration**

```yaml
model:
  name: convlstm
```

---

## 2. CNN-LSTM

**Architecture**

CNN Feature Extractor + LSTM Temporal Model

**Advantages**

- Strong spatial feature extraction
- Good temporal learning
- Effective for complex climate patterns

**Configuration**

```yaml
model:
  name: cnn_lstm
```

---

## 3. Transformer

**Architecture**

Multi-Head Self Attention

**Advantages**

- Captures long-range dependencies
- Parallel computation
- Suitable for long climate sequences

**Configuration**

```yaml
model:
  name: transformer
```

---

## Model Comparison

| Feature | ConvLSTM | CNN-LSTM | Transformer |
|----------|----------|----------|-------------|
| Spatial Learning | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Temporal Learning | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Memory Usage | Low | Medium | High |
| Training Speed | Fast | Medium | Medium |
| Long-Term Dependencies | Good | Good | Excellent |
| Recommended | ✅ Yes | Yes | Research |

> **ConvLSTM is the default and recommended model for this project due to its balance of prediction accuracy and computational efficiency.**
> ---

# 💻 Usage

After completing the installation and preprocessing steps, you can train, evaluate, and visualize the model using the commands below.

---

## 🚀 Train the Model

Train the default ConvLSTM model using the configuration specified in `config.yaml`.

```bash
python main.py
```

Or specify a custom configuration file.

```bash
python main.py --config config.yaml
```

The training pipeline automatically performs:

- Dataset loading
- Model initialization
- Training
- Validation
- Checkpoint saving
- Performance logging

All trained models are stored inside the `experiments/` directory.

---

# 🔍 Evaluate the Model

Evaluate the trained model on the test dataset.

```bash
python evaluate_model.py \
--model-path experiments/exp_02_convlstm/best_model.pth
```

The evaluation reports:

- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Temperature prediction maps
- Performance graphs

Results are automatically saved in:

```text
outputs/
```

---

# 🌡 Make Temperature Predictions

Predict temperature for a specific day.

```bash
python predict_by_date.py --date 2023-06-15
```

Predict temperatures for multiple days.

```bash
python predict_by_date.py \
--start-date 2023-06-01 \
--end-date 2023-06-30
```

Predictions are stored inside

```text
outputs/predictions/
```

---

# 🔮 Future Forecasting

Generate forecasts for future days.

```bash
python future_predict.py --days-ahead 30
```

Example

```text
Input :
Last 7 Days Temperature Maps

↓

Output :

Next 30 Days Temperature Forecast
```

---

# 📊 Interactive Dashboard

Launch the Streamlit dashboard.

```bash
streamlit run dashboard.py
```

The dashboard provides

- Prediction visualization
- Temperature maps
- Model comparison
- Evaluation metrics
- Interactive charts
- Regional forecasting

---

# 🧪 Quick Testing

Generate a synthetic dataset.

```bash
python generate_synthetic_data.py --samples 1000
```

Run a quick validation.

```bash
python quick_test.py
```

Verify prediction outputs.

```bash
python verify_results.py
```

Debug predictions.

```bash
python debug_predictions.py
```

---

# 🔎 Data Validation

Validate the processed dataset.

```bash
python check_available_data.py

python check_data.py

python check_merged.py

python check_tensors.py
```

These scripts verify

- Missing files
- Dataset integrity
- Tensor dimensions
- Data consistency
- Merge correctness

---

# 🛠 Troubleshooting

## Installation Issues

### Missing PyTorch

```bash
pip install torch torchvision torchaudio
```

---

### Missing Dependencies

```bash
pip install -r requirements.txt --upgrade
```

---

## GPU Issues

### CUDA Not Available

Check CUDA availability.

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If CUDA is unavailable, switch to CPU.

```yaml
training:
    device: cpu
```

---

### CUDA Out of Memory

Reduce batch size.

```yaml
training:
    batch_size: 4
```

---

## Data Issues

### Missing ERA5 Files

Download again.

```bash
python preprocessing/data_download/download_era5.py
```

---

### Corrupted Data

Run

```bash
python check_data.py

python fix_data.py
```

---

### Tensor Errors

```bash
python check_tensors.py

python preprocessing/create_sequences.py
```

---

## Training Issues

### Loss Becomes NaN

Reduce learning rate.

```yaml
learning_rate: 0.00001
```

Increase regularization.

```yaml
weight_decay: 1e-5
```

---

### Overfitting

Increase

```yaml
dropout: 0.3
```

or

```yaml
weight_decay: 1e-4
```

---

### Slow Training

Increase

```yaml
batch_size: 16
```

if GPU memory permits.

---

# 📦 Dependencies

Major libraries used in this project.

| Library | Purpose |
|----------|----------|
| PyTorch | Deep Learning |
| Xarray | NetCDF Processing |
| NumPy | Numerical Computing |
| Pandas | Data Analysis |
| NetCDF4 | Climate Data |
| Scikit-learn | Metrics & Preprocessing |
| Matplotlib | Visualization |
| CDS API | ERA5 Download |
| Streamlit | Dashboard |
| PyYAML | Configuration |

Install all dependencies.

```bash
pip install -r requirements.txt
```

---

# 🚀 Deployment

The project can be deployed using Streamlit.

## Streamlit Cloud

1. Push repository to GitHub.
2. Create a Streamlit Cloud application.
3. Select

```
dashboard.py
```

as the entry point.

---

## Render

Build Command

```bash
pip install -r requirements.txt
```

Start Command

```bash
streamlit run dashboard.py
```

---

## Recommended Deployment

| Platform | Status |
|----------|--------|
| Streamlit Cloud | ✅ Recommended |
| Render | ✅ Recommended |
| Railway | ✅ Supported |
| Docker | ✅ Supported |
| Vercel | ❌ Not Recommended |

> **Note:** Vercel is not suitable for this project because of its serverless execution limits and the large machine learning dependencies required by PyTorch and Streamlit.
>
> ---

# 🗺️ Project Roadmap

The project is continuously evolving with additional models, optimization techniques, and deployment capabilities.

## ✅ Completed

- [x] ERA5 Climate Data Integration
- [x] Automated Data Preprocessing Pipeline
- [x] ConvLSTM Model Implementation
- [x] Model Training & Validation Pipeline
- [x] Temperature Prediction
- [x] Model Evaluation Metrics
- [x] Streamlit Dashboard
- [x] Configuration Management
- [x] Experiment Tracking

---

## 🚧 In Progress

- [ ] CNN-LSTM Performance Optimization
- [ ] Transformer Model Training
- [ ] Real-Time Temperature Prediction
- [ ] Model Performance Benchmarking
- [ ] Improved Visualization Dashboard

---

## 📌 Planned Features

- [ ] Multi-Step Temperature Forecasting
- [ ] Extreme Weather Event Prediction
- [ ] Ensemble Learning Models
- [ ] Hyperparameter Optimization
- [ ] Model Explainability (XAI)
- [ ] Docker Support
- [ ] REST API using FastAPI
- [ ] Cloud Deployment
- [ ] CI/CD Pipeline
- [ ] Automated Experiment Tracking

---

# 📈 Performance Summary

| Metric | Value |
|----------|--------|
| Dataset | ERA5 Climate Reanalysis |
| Climate Variable | 2m Air Temperature (t2m) |
| Input Sequence Length | 7 Days |
| Forecast Horizon | Next Day |
| Framework | PyTorch |
| Primary Model | ConvLSTM |
| Spatial Resolution | 121 × 141 Grid |
| Data Format | NetCDF (.nc) |

---

# 📚 Documentation

Additional project documentation is available in the repository.

| Document | Description |
|----------|-------------|
| `RESULTS_SUMMARY.md` | Experimental Results |
| `RESULTS_SUMMARY_UPDATED.md` | Updated Performance |
| `IMPLEMENTATION_ROADMAP.md` | Development Timeline |
| `DECISION_TREE.md` | Workflow Guide |
| `LIVE_STREAMING_GUIDE.md` | Streaming Prediction |
| `STATIC_VS_STREAMING_GUIDE.md` | Architecture Comparison |
| `SIMPLE_AUTOMATION_GUIDE.md` | Automation Guide |

---

# 📓 Jupyter Notebooks

The repository also includes notebooks for experimentation and debugging.

| Notebook | Purpose |
|-----------|----------|
| `01_explore_data.ipynb` | Data Exploration |
| `02_preprocessing_debug.ipynb` | Preprocessing Pipeline |
| `03_model_debug.ipynb` | Model Debugging |

---

# 📜 Citation

If you use this repository in your research or publications, please cite it as:

```bibtex
@software{ClimateForecasting2026,
  author = {Nidhi Dhameliya},
  title = {AI-Powered Climate Temperature Forecasting using ConvLSTM},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/your-username/your-repository}
}
```

---

## ERA5 Dataset Citation

```bibtex
@article{hersbach2020era5,
  title={The ERA5 Global Reanalysis},
  author={Hersbach, H. and others},
  journal={Quarterly Journal of the Royal Meteorological Society},
  year={2020}
}
```

---

# 🙏 Acknowledgements

This project was made possible through the support of the following open-source tools and datasets:

- **ECMWF** – ERA5 Climate Reanalysis Dataset
- **Copernicus Climate Change Service (C3S)**
- **PyTorch**
- **Xarray**
- **NetCDF4**
- **NumPy**
- **Pandas**
- **Scikit-learn**
- **Matplotlib**
- **Streamlit**
- **CDS API**

Special thanks to the open-source community for providing high-quality tools that made this research possible.

---

# 🤝 Contributing

Contributions are welcome!

If you would like to improve the project:

1. Fork the repository.
2. Create a feature branch.

```bash
git checkout -b feature/new-feature
```

3. Commit your changes.

```bash
git commit -m "Add new feature"
```

4. Push the branch.

```bash
git push origin feature/new-feature
```

5. Open a Pull Request.

Please ensure your code follows the existing project structure and coding style.

---

# 🐛 Reporting Issues

If you encounter a bug or have a feature request, please open a GitHub Issue with:

- Problem description
- Steps to reproduce
- Expected behavior
- Screenshots (if applicable)
- System information

---

# 📄 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for more details.

---

# ⭐ Support the Project

If you find this repository useful for your research, learning, or development, please consider giving it a ⭐ on GitHub.

Your support helps improve the project and encourages future development.

---

## 👩‍💻 Author

**Nidhi Dhameliya**

M.Tech – Data Science & Machine Learning

Deep Learning • Climate Informatics • Artificial Intelligence • Spatiotemporal Forecasting

---

<div align="center">

### ⭐ If you like this project, don't forget to star the repository! ⭐

**Happy Forecasting! 🌍📈**

</div>
