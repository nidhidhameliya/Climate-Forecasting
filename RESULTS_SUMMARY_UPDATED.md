# ConvLSTM Model: Performance and Results Summary

This document provides a summary of the experimental results for the ConvLSTM model, as presented in the associated research paper.

---

## 1. Quantitative Evaluation

The model's performance was evaluated on the training, validation, and test datasets using Root Mean Square Error (RMSE) and Mean Absolute Error (MAE). The results demonstrate consistent and high accuracy across all data splits.

### Performance Metrics

| Split      | Samples | RMSE (°C) | MAE (°C) |
|------------|---------|-----------|----------|
| **Train**      | 1,089   | 0.0018    | 0.0018   |
| **Validation** | 358     | 0.0018    | 0.0018   |
| **Test**       | 358     | 0.0018    | 0.0018   |

---

## 2. Model and Data Configuration

### Model Architecture

| Component | Details |
|-----------|---------|
| **Model Type**   | ConvLSTM (Convolutional LSTM) |
| **Input Shape**  | 7-day sequence of 121x141 temperature maps |
| **Output Shape** | 1-day forecast of a 121x141 temperature map |
| **Layers**       | 2 stacked ConvLSTM layers |
| **Hidden Dims**  | 32 channels |
| **Parameters**   | ~29,500 |
| **Optimizer**    | Adam (learning rate: 1e-4) |
| **Loss Function**| Mean Squared Error (MSE) |


### Dataset Details

| Parameter | Value |
|-----------|-------|
| **Data Source** | ERA5 Reanalysis |
| **Variable** | 2m Temperature (`t2m`) |
| **Region** | India (5°N-35°N, 65°E-100°E) |
| **Grid Resolution** | 121 × 141 points |
| **Temporal Res.** | Daily (from hourly) |
| **Sequence Length** | 7 days input -> 1 day output |

---

## 3. Accessing Results

The full set of evaluation artifacts can be found in the repository.

### Raw Metrics (JSON)
Detailed metrics for each data split are located in `outputs/evaluation/`:
```
outputs/evaluation/
├── train_results.json
├── val_results.json
└── test_results.json
```

### Visualizations
Generated plots, including spatial error maps and prediction comparisons, are saved in `outputs/visualizations/`.

### Interactive Dashboard
For an interactive exploration of the results, run the Streamlit dashboard:
```bash
streamlit run dashboard.py
```

### Trained Model
The best performing model checkpoint is saved at:
```
experiments/latest/model.pth
```

---

*Last Updated: March 2, 2026*
