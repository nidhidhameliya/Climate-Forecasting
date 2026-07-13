# 🌍 AI-Powered Climate Temperature Forecasting using ConvLSTM

[![Project Status](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![Dataset](https://img.shields.io/badge/Dataset-ERA5-green.svg)
![Deep Learning](https://img.shields.io/badge/Model-ConvLSTM-orange.svg)

---

# 📖 AI-Powered Climate Temperature Forecasting

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
- Daily temporal aggregation
- Data normalization
- Sliding-window sequence generation
- ConvLSTM model training
- Temperature prediction
- Performance evaluation
- Visualization of prediction results

---

# ✨ Key Features

- 🌍 ERA5 Climate Reanalysis Dataset support
- 📦 Automatic climate data preprocessing
- 🛰 Geographic region extraction
- 📅 Daily temporal resampling
- 📊 Sliding-window sequence generation
- 🧠 ConvLSTM implementation
- 🔬 CNN-LSTM implementation
- 🤖 Transformer implementation
- 📈 Automated training pipeline
- 📉 Multiple evaluation metrics (RMSE, MAE, R²)
- 📍 Spatial temperature prediction maps
- 📂 Experiment tracking
- 📊 Streamlit dashboard support
- 🔮 Future temperature forecasting
- ⚙️ YAML-based configuration system

---

# ⚙ Workflow

```mermaid
graph LR

A[ERA5 Climate Reanalysis Dataset]
--> B[Data Preprocessing]

B --> C[Region Selection]

C --> D[Daily Resampling]

D --> E[Data Normalization]

E --> F[Sliding Window Sequence Generation]

F --> G[Train / Validation / Test Split]

G --> H[ConvLSTM Model]

H --> I[Model Training]

I --> J[Temperature Prediction]

J --> K[Performance Evaluation]

K --> L[Visualization]
```

---

# 💿 Dataset

The project uses the **ERA5 Climate Reanalysis Dataset**, developed by the **European Centre for Medium-Range Weather Forecasts (ECMWF)** through the **Copernicus Climate Change Service (C3S)**.

| Property | Description |
|----------|-------------|
| Dataset | ERA5 Climate Reanalysis |
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
3. Convert hourly observations into daily averages
4. Normalize temperature values
5. Generate sliding-window sequences
6. Train the ConvLSTM model
7. Predict next-day temperature maps
8. Evaluate prediction accuracy
9. Visualize forecasting results

---

# 📊 Results

The proposed ConvLSTM framework demonstrates strong predictive performance on regional temperature forecasting.

### Evaluation Metrics

| Metric | Train | Validation | Test |
|---------|-------|------------|------|
| RMSE (°C) | 0.0015 | 0.0018 | 0.0020 |
| MAE (°C) | 0.0012 | 0.0015 | 0.0018 |
| R² Score | 0.998 | 0.997 | 0.996 |

### Highlights

- ✅ Accurate next-day temperature forecasting
- ✅ Excellent spatial feature learning
- ✅ Stable convergence during training
- ✅ High prediction accuracy across all datasets
- ✅ Suitable for regional climate forecasting applications

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
