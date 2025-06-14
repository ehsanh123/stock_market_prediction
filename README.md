# 📈 FTSE 100 Stock Price Prediction

## 🔍 Overview

Predicts FTSE 100 stock prices using GRU-based time series models and explains predictions with LIME to highlight key features. Evaluates model performance to find the best approach.

---

## 🎯 Objectives

- Forecast FTSE 100 stock closing prices 
- Use GRU models for time series analysis  
- Explain predictions with LIME for feature importance  
- Compare model outputs for effectiveness  

---

## 🗂 Structure

- `Get_stocks.ipynb`: Fetches FTSE 100 stock data (e.g., `BP.L`) from Yahoo Finance, saves as CSVs  
- `prediction.py`: Contains functions for:
  - Preprocessing (scaling, 60-day sequences)
  - GRU model (64/32 units, dropout, dense layers)
  - Plotting loss and trends
  - Rolling predictions and LIME explainability
- `predict1.ipynb`: Runs full prediction pipeline for `BP.L` (2015–2022), trains the model, applies LIME

---

## 🚀 Features

- **Data**: Daily stock prices (2015–2022) from Yahoo Finance  
- **Model**: GRU with 60-day input, two GRU layers, dropout (0.2), dense output  
- **Explainability**: LIME shows impact of past prices (e.g., `Day59_0`)  
- **Metrics**: MSE, RMSE; includes plots for loss and predictions  

---

## ⚙️ Setup

**Python 3.10+ required**

Install dependencies:


- **pip install yfinance pandas numpy tensorflow scikit-learn matplotlib lime pydot


- **Install GraphViz for model visualization:

Windows: Download from graphviz.org, add bin to PATH

## Linux:

sudo apt install graphviz

Mac:

brew install graphviz

## 📦 Usage

Run Get_stocks.ipynb to download stock data

Use predict1.ipynb to preprocess, train, and predict for BP.L

Apply LIME in predict1.ipynb for feature importance

Visualize results with prediction.py using:

plot_loss(history)
display_trends(...)

To view model structure:

from tensorflow.keras.utils import plot_model
plot_model(model3, to_file='model3.png', show_shapes=True)

## 📊 Example Output

Model:

Input_Layer:     (None, 60, 1)
GRU_Layer1:      (None, 60, 64)
GRU_Layer2:      (None, 32)
Dense_Layer:     (None, 16)
Output_Layer:    (None, 1)
Total params: 22817

## LIME Explanation:

Day59_0: -0.30
Day58_0: -0.11

Visuals:

Loss curves

Actual vs. predicted prices

## 📜 License

This project is licensed under the MIT License.
