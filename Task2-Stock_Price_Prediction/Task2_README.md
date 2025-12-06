# Task 2: Predict Future Stock Prices (Short-Term)

## 📊 Overview
This task implements regression models to predict the next day's closing stock price using historical OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance.

## 🎯 Objective
Use historical stock data to predict the next day's closing price using machine learning regression models.

## 📁 Dataset
**Apple (AAPL) Stock Data**
- **Source:** Yahoo Finance (via yfinance library)
- **Date Range:** 2020-01-01 to 2025-01-01 (5 years)
- **Frequency:** Daily trading data
- **Features:**
  - Open Price
  - High Price
  - Low Price
  - Close Price
  - Volume
- **Target:** Next Day's Closing Price

## 🛠️ Technologies Used
- **Python 3.8+**
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **yfinance** - Financial data API
- **matplotlib** - Plotting
- **seaborn** - Visualizations
- **sklearn** - Machine learning models

## 📋 Requirements Checklist

### What This Notebook Includes:
- ✅ Select a stock (AAPL - Apple)
- ✅ Load historical data using yfinance
- ✅ Use Open, High, Low, Volume features
- ✅ Predict next day's Close price
- ✅ Train Linear Regression model
- ✅ Train Random Forest model
- ✅ Plot actual vs predicted prices
- ✅ Compare model performance

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install pandas numpy yfinance matplotlib seaborn scikit-learn
```

### 2. Run the Notebook
```bash
jupyter notebook Task2_Stock_Price_Prediction.ipynb
```

### 3. Note on Data Download
The notebook automatically downloads stock data from Yahoo Finance. Ensure you have an active internet connection.

## 📈 Key Outputs

### Models Implemented:
1. **Linear Regression** - Baseline model
2. **Random Forest Regressor** - Ensemble model (200 trees)

### Evaluation Metrics:
- **MAE (Mean Absolute Error)** - Average prediction error in dollars
- **RMSE (Root Mean Squared Error)** - Penalizes large errors
- **R² Score** - Model fit quality (0-1, higher is better)

### Visualizations:
1. Time series plot: Actual vs Predicted (Linear Regression)
2. Time series plot: Actual vs Predicted (Random Forest)
3. Scatter plots: Actual vs Predicted for both models
4. Feature importance chart (Random Forest)
5. Residual analysis plots
6. Model comparison charts

### Expected Results:
- **Linear Regression:**
  - MAE: ~$3-4
  - RMSE: ~$4-5
  - R² Score: ~0.78-0.82

- **Random Forest:**
  - MAE: ~$2-3
  - RMSE: ~$3-4
  - R² Score: ~0.82-0.87

### Key Findings:
- 📈 Random Forest outperforms Linear Regression
- 💰 Close price is the most important feature
- 📊 Historical prices moderately predict future prices
- ⚠️ Model may miss sudden price spikes
- 🎯 Suitable for trend prediction, not exact values

## 💡 Skills Demonstrated
- Time series data handling
- Regression modeling
- API integration (yfinance)
- Feature engineering (creating next-day target)
- Model evaluation and comparison
- Financial data analysis
- Residual analysis

## 📊 Notebook Structure
1. Import Libraries
2. Download Stock Data
3. Data Inspection
4. Feature Engineering (Next_Close target)
5. Train-Test Split (80-20, time-ordered)
6. Feature Scaling (StandardScaler)
7. Model Training
   - Linear Regression
   - Random Forest
8. Model Evaluation (MAE, RMSE, R²)
9. Visualizations
10. Feature Importance
11. Residual Analysis
12. Key Findings

## 🎓 Learning Outcomes
- Fetch financial data using APIs
- Handle time series data properly
- Build regression models for prediction
- Evaluate and compare multiple models
- Understand limitations of price prediction

## ⚠️ Important Notes
- **Not Financial Advice:** This is an educational project only
- **Model Limitations:** Past performance ≠ future results
- **Real Trading:** Consider many more factors (news, sentiment, fundamentals)
- **Data Availability:** Yahoo Finance may rate-limit requests

## 🔗 Additional Resources
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [Time Series Forecasting Guide](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [Stock Market Analysis Tutorial](https://www.investopedia.com/terms/t/technical-analysis.asp)

---

**Status:** ✅ Complete  
**Estimated Time:** 45-60 minutes  
**Difficulty:** Intermediate
