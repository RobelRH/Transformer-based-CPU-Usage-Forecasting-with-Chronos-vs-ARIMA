# CPU Usage Forecasting with ARIMA & Transformer (Chronos-style)

## Overview
This project demonstrates **time series forecasting of CPU usage** using:

- **ARIMA**: Classical statistical time series model  
- **Transformer (Chronos-style)**: Modern deep learning model for time series

We generate **synthetic CPU usage data** with trend, seasonality, noise, and random spikes, then compare model performance using **RMSE** and **MAE**.

---

## Features

- Synthetic CPU usage dataset  
- Baseline ARIMA forecasting  
- Transformer-based forecasting  
- Visual comparison of forecasts vs actual  
- Performance metrics table  

---

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# Install dependencies
pip install statsmodels matplotlib torch scikit-learn
```

---

## Usage

1. Open `CPU_Forecasting_Transformer.ipynb` in **Jupyter Notebook** or **Google Colab**.  
2. Run all cells sequentially:  
   - Generate synthetic CPU data  
   - Fit ARIMA model and forecast  
   - Fit Transformer model and forecast  
   - Evaluate performance and plot results  
3. Check **comparison table** for RMSE and MAE.

---

## Results

**Example metrics on synthetic data**:

| Model       | RMSE   | MAE    |
|------------|--------|--------|
| ARIMA      | 11.01  | 8.02   |
| Transformer| 11.04  | 7.79   |

**Plots include**:

- Synthetic CPU usage data  
- ARIMA forecast vs actual  
- Transformer forecast vs actual  

---

## Future Improvements

- Use **real CPU usage datasets** for realistic evaluation  
- Increase sequence length for transformer input  
- Multivariate forecasting (CPU + Memory + Network)  
- Hyperparameter tuning for transformer model  

---

## License
This project is released under the MIT License.
