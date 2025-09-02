# Transformer-based CPU Usage Forecasting with Chronos
This project demonstrates **time series forecasting of CPU usage** using:

1. **ARIMA** – classical statistical model
2. **Transformer-based model (Chronos-style)** – modern deep learning approach

We also **compare performance** using RMSE and MAE.

---

## Dataset
We generate **synthetic CPU usage data** including:
- Trend
- Seasonality
- Random spikes

# ===============================
# 1. Install Dependencies
# ===============================
!pip install statsmodels matplotlib torch -q

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import warnings
warnings.filterwarnings("ignore")


# ===============================
# 2. Generate Synthetic CPU Usage Data
# ===============================
np.random.seed(42)
time = pd.date_range(start="2023-01-01", periods=500, freq="H")

cpu_usage = (
    30 + 10 * np.sin(np.linspace(0, 20, 500))   # seasonality
    + np.linspace(0, 15, 500)                   # upward trend
    + np.random.normal(0, 5, 500)               # noise
)

# Add random spikes
spike_indices = np.random.choice(len(cpu_usage), 15, replace=False)
cpu_usage[spike_indices] += np.random.randint(20, 40, size=15)

data = pd.DataFrame({"time": time, "cpu": cpu_usage})
data.set_index("time", inplace=True)

plt.figure(figsize=(12,5))
plt.plot(data.index, data["cpu"], label="CPU Usage")
plt.title("Synthetic CPU Usage Data")
plt.xlabel("Time")
plt.ylabel("CPU %")
plt.legend()
plt.show()

# Train/Test Split
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

## ARIMA Model
We use **statsmodels’ ARIMA** to build a baseline forecasting model.


from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA (order p=5, d=1, q=0)
arima_model = ARIMA(train["cpu"], order=(5,1,0))
arima_result = arima_model.fit()

# Forecast
n_periods = len(test)
arima_forecast = arima_result.forecast(steps=n_periods)

# Evaluation
rmse_arima = math.sqrt(mean_squared_error(test["cpu"], arima_forecast))
mae_arima = mean_absolute_error(test["cpu"], arima_forecast)
print(f"ARIMA RMSE: {rmse_arima:.2f}, MAE: {mae_arima:.2f}")

# Plot
plt.figure(figsize=(12,5))
plt.plot(train.index, train["cpu"], label="Train")
plt.plot(test.index, test["cpu"], label="Test")
plt.plot(test.index, arima_forecast, label="ARIMA Forecast")
plt.title("ARIMA Forecast vs Actual CPU Usage")
plt.xlabel("Time")
plt.ylabel("CPU %")
plt.legend()
plt.show()


## Transformer-based Forecasting (Chronos-style)
We implement a **simple transformer model** for time series forecasting.

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

SEQ_LEN = 24  # look-back steps
PRED_LEN = 1  # predict 1 step ahead

# Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_len=SEQ_LEN):
        self.series = series
        self.seq_len = seq_len
    def __len__(self):
        return len(self.series) - self.seq_len
    def __getitem__(self, idx):
        x = self.series[idx:idx+self.seq_len]
        y = self.series[idx+self.seq_len]
        return x, y

train_values = torch.tensor(train["cpu"].values, dtype=torch.float32).unsqueeze(1)
test_values = torch.tensor(test["cpu"].values, dtype=torch.float32).unsqueeze(1)

train_dataset = TimeSeriesDataset(train_values)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Transformer Model
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers
        )
        self.fc = nn.Linear(d_model, 1)
    def forward(self, src):
        src = self.input_proj(src).permute(1,0,2)
        out = self.transformer(src, src)
        out = out[-1,:,:]  # last time step
        return self.fc(out)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Training
EPOCHS = 30
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch+1)%5==0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}")

# Forecasting
model.eval()
preds = []
input_seq = train_values[-SEQ_LEN:].unsqueeze(0).to(device)

with torch.no_grad():
    for _ in range(len(test)):
        y_pred = model(input_seq)
        preds.append(y_pred.item())
        input_seq = torch.cat([input_seq[:,1:,:], y_pred.unsqueeze(0).unsqueeze(2)], dim=1)

# Evaluation
rmse_tf = math.sqrt(mean_squared_error(test["cpu"], preds))
mae_tf = mean_absolute_error(test["cpu"], preds)
print(f"Transformer RMSE: {rmse_tf:.2f}, MAE: {mae_tf:.2f}")

# Plot
plt.figure(figsize=(12,5))
plt.plot(train.index, train["cpu"], label="Train")
plt.plot(test.index, test["cpu"], label="Test")
plt.plot(test.index, preds, label="Transformer Forecast")
plt.title("Transformer Forecast vs Actual CPU Usage")
plt.xlabel("Time")
plt.ylabel("CPU %")
plt.legend()
plt.show()

## Comparison
We summarize the results of ARIMA vs Transformer.

results = pd.DataFrame({
    "Model": ["ARIMA", "Transformer"],
    "RMSE": [rmse_arima, rmse_tf],
    "MAE": [mae_arima, mae_tf]
})
results

## Conclusion
- Both models provide reasonable forecasts on synthetic CPU data.
- Transformer slightly improves MAE, but ARIMA remains competitive.
- Future improvements:
  - Use real CPU usage data
  - Longer sequences or multivariate features
  - Hyperparameter tuning for the transformer 
