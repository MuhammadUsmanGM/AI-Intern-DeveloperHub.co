# Task 6: House Price Prediction

## 🌐 Live Deployment
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-intern-developerappco-6b4gz4tvvfxs5rkvgpzf3e.streamlit.app/)

## 📊 Overview
This task builds regression models to predict house prices based on property features such as square footage, bedrooms, bathrooms, and other real estate characteristics.

## 🎯 Objective
Predict house prices using property features and evaluate model performance with regression metrics.

## 📁 Dataset
**House Price Prediction Dataset**
- **Source:** Kaggle / Generated Sample Data
- **Size:** 500 samples (sample), or use real Kaggle dataset
- **Features Include:**
  - Square Footage
  - Number of Bedrooms
  - Number of Bathrooms
  - Year Built
  - Garage Spaces
  - Pool (Yes/No)
  - Lot Size (acres)

**Optional Kaggle Dataset:**
https://www.kaggle.com/datasets/harlfoxem/housesalesprediction

## 🛠️ Technologies Used
- **Python 3.8+**
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **matplotlib** - Plotting
- **seaborn** - Visualizations
- **sklearn** - Machine learning models

## 📋 Requirements Checklist

### What This Notebook Includes:
- ✅ Preprocess features (square footage, bedrooms, location)
- ✅ Train Linear Regression model
- ✅ Train Gradient Boosting model
- ✅ Visualize predicted vs actual prices
- ✅ Evaluate with MAE and RMSE
- ✅ Feature importance analysis
- ✅ Residual analysis

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Data Options

**Option A: Use Generated Sample Data (Default)**
- No download needed
- Runs immediately
- Good for learning

**Option B: Use Real Kaggle Dataset**
- Download from Kaggle
- More realistic prices
- Better for portfolio

### 3. Run the Notebook
```bash
jupyter notebook Task6_House_Price_Prediction.ipynb
```

## 📈 Key Outputs

### Models Implemented:
1. **Linear Regression** - Baseline model
2. **Gradient Boosting Regressor** - Ensemble model

### Evaluation Metrics:
- **MAE (Mean Absolute Error)** - Average price error
- **RMSE (Root Mean Squared Error)** - Penalizes large errors
- **R² Score** - Model fit quality (0-1)

### Visualizations Generated:
1. Price distribution (histogram & box plot)
2. Correlation heatmap
3. Feature vs Price scatter plots (6 features)
4. Actual vs Predicted (line plots for both models)
5. Actual vs Predicted (scatter plots)
6. Feature importance charts
7. Residual analysis plots
8. Model metrics comparison

### Expected Results:
- **Linear Regression:**
  - MAE: ~$12,000-15,000
  - RMSE: ~$15,000-18,000
  - R² Score: ~0.80-0.84

- **Gradient Boosting:**
  - MAE: ~$8,000-12,000
  - RMSE: ~$10,000-14,000
  - R² Score: ~0.85-0.90

### Key Findings:
- 🏠 Square footage is the strongest price predictor
- 🛏️ Bedrooms significantly affect price
- 🏊 Pool adds substantial value
- 📅 Newer homes command higher prices
- 🎯 Gradient Boosting outperforms Linear Regression

## 💡 Skills Demonstrated
- Regression modeling
- Feature scaling and selection
- Model evaluation (MAE, RMSE, R²)
- Real estate data understanding
- Ensemble methods (Gradient Boosting)
- Feature importance analysis
- Residual analysis

## 📊 Notebook Structure
1. Import Libraries
2. Load/Generate Dataset
3. Data Inspection
4. Exploratory Data Analysis
   - Price distribution
   - Correlation analysis
   - Feature relationships
5. Data Preprocessing
   - Handle missing values
   - Feature-target separation
   - Train-test split
   - Feature scaling
6. Model Training
   - Linear Regression
   - Gradient Boosting
7. Model Evaluation
8. Visualizations
9. Feature Importance
10. Residual Analysis
11. Price Predictions on New Examples
12. Key Findings

## 🎓 Learning Outcomes
- Build regression models for real estate
- Preprocess and scale features
- Evaluate models with appropriate metrics
- Interpret feature importance
- Analyze residuals for model diagnostics
- Compare multiple regression algorithms

## 🏡 Real Estate Features Explained

### Primary Features:
- **Square Feet:** Total living area
- **Bedrooms:** Number of sleeping rooms
- **Bathrooms:** Full + half bathrooms
- **Year Built:** Age affects price
- **Garage:** Parking space value
- **Pool:** Premium amenity
- **Lot Size:** Land area in acres

### Price Relationships:
- ↑ More sqft = ↑ Higher price
- ↑ More bedrooms = ↑ Higher price
- ↑ Newer homes = ↑ Higher price
- ✓ Pool presence = Premium boost

## 📈 Sample Predictions

The notebook includes example predictions:
```
House 1: 2000 sqft, 3 bed → $X predicted
House 2: 3000 sqft, 4 bed → $Y predicted
House 3: 1500 sqft, 2 bed → $Z predicted
```

## 🔧 Customization

### Add More Features:
```python
df['new_feature'] = your_data
X = df[['SquareFeet', 'Bedrooms', 'new_feature']]
```

### Try Different Models:
```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
```

### Adjust Train-Test Split:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3  # 70-30 split
)
```

## ⚠️ Important Notes
- **Sample Data:** Default uses generated data for learning
- **Real Data:** For portfolios, use actual Kaggle datasets
- **Market Factors:** Real prices depend on many more factors
- **Location:** Not included in sample (add if using real data)

## 🔗 Additional Resources
- [House Sales Dataset](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)
- [Gradient Boosting Guide](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
- [Real Estate ML Analysis](https://towardsdatascience.com/predicting-house-prices-with-machine-learning-62d5bcd0d68f)

---

**Status:** ✅ Complete  
**Estimated Time:** 60-75 minutes  
**Difficulty:** Intermediate
