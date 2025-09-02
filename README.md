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
git clone https://github.com/RobelRH/Transformer-based-CPU-Usage-Forecasting-with-Chronos-vs-ARIMA.git
cd Transformer-based-CPU-Usage-Forecasting-with-Chronos-vs-ARIMA

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
- Hyperparameter tuning for the transformer model  

---

## License
This project is released under the MIT License.

## MIT LICENSE
```
MIT License

Copyright (c) 2025 Robel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
