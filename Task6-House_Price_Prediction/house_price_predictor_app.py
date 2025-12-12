import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set up the page configuration
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Title and description
st.title("🏠 House Price Prediction App")
st.markdown("""
This application predicts house prices based on various property features.
Fill in the details of the house below to get an estimated price.
""")

# Create a sample dataset to train the model
@st.cache_data
def create_sample_data():
    np.random.seed(42)
    
    # Generate sample house price dataset
    n_samples = 500
    
    df = pd.DataFrame({
        'SquareFeet': np.random.uniform(800, 5000, n_samples),
        'Bedrooms': np.random.randint(1, 6, n_samples),
        'Bathrooms': np.random.uniform(1, 4, n_samples),
        'YearBuilt': np.random.randint(1950, 2023, n_samples),
        'Garage': np.random.randint(0, 4, n_samples),
        'Pool': np.random.randint(0, 2, n_samples),
        'Lot_Size': np.random.uniform(0.5, 2, n_samples),
    })
    
    # Generate prices based on features with some correlation
    df['Price'] = (
        150 * df['SquareFeet'] +
        50000 * df['Bedrooms'] +
        30000 * df['Bathrooms'] +
        2000 * (2023 - df['YearBuilt']) +  # Negative: older houses worth less
        20000 * df['Garage'] +
        100000 * df['Pool'] +
        50000 * df['Lot_Size'] +
        np.random.normal(0, 50000, n_samples)  # Add noise
    )
    
    # Ensure prices are positive
    df['Price'] = np.abs(df['Price'])
    
    return df

# Train models
@st.cache_resource
def train_models(df):
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    # Split the data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Linear Regression model
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(X_train_scaled, y_train)
    
    # Train Gradient Boosting model
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    
    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_pred_lin = lin_reg_model.predict(X_test_scaled)
    y_pred_gb = gb_model.predict(X_test_scaled)
    
    lin_mae = mean_absolute_error(y_test, y_pred_lin)
    gb_mae = mean_absolute_error(y_test, y_pred_gb)
    lin_r2 = r2_score(y_test, y_pred_lin)
    gb_r2 = r2_score(y_test, y_pred_gb)
    
    return lin_reg_model, gb_model, scaler, lin_mae, gb_mae, lin_r2, gb_r2

# Load data and models
df = create_sample_data()
lin_reg_model, gb_model, scaler, lin_mae, gb_mae, lin_r2, gb_r2 = train_models(df)

# Sidebar for inputs
st.sidebar.header("House Features")

# Input widgets
square_feet = st.sidebar.slider("Square Feet", 500, 8000, 2000, step=50)
bedrooms = st.sidebar.slider("Bedrooms", 1, 8, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1.0, 6.0, 2.0, step=0.5)
year_built = st.sidebar.slider("Year Built", 1900, 2023, 2000)
garage = st.sidebar.slider("Garage Spaces", 0, 5, 2)
pool = st.sidebar.selectbox("Pool", ["No", "Yes"], index=0)
lot_size = st.sidebar.slider("Lot Size (acres)", 0.1, 5.0, 0.5, step=0.1)

# Convert pool selection to binary
pool_binary = 1 if pool == "Yes" else 0

# Create input dataframe
input_data = pd.DataFrame({
    'SquareFeet': [square_feet],
    'Bedrooms': [bedrooms],
    'Bathrooms': [bathrooms],
    'YearBuilt': [year_built],
    'Garage': [garage],
    'Pool': [pool_binary],
    'Lot_Size': [lot_size]
})

# Scale the input data
input_scaled = scaler.transform(input_data)

# Make predictions
lin_pred = lin_reg_model.predict(input_scaled)[0]
gb_pred = gb_model.predict(input_scaled)[0]

# Calculate average prediction
avg_pred = (lin_pred + gb_pred) / 2

# Display predictions
st.subheader("Price Predictions")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Linear Regression Prediction", 
        value=f"${lin_pred:,.2f}",
        delta=f"{(lin_pred - avg_pred)/avg_pred*100:.1f}% vs Average"
    )

with col2:
    st.metric(
        label="Gradient Boosting Prediction", 
        value=f"${gb_pred:,.2f}",
        delta=f"{(gb_pred - avg_pred)/avg_pred*100:.1f}% vs Average"
    )

with col3:
    st.metric(
        label="Average Prediction", 
        value=f"${avg_pred:,.2f}"
    )

# Display input details
st.subheader("House Details")
input_df = pd.DataFrame({
    'Feature': ['Square Feet', 'Bedrooms', 'Bathrooms', 'Year Built', 'Garage', 'Pool', 'Lot Size (acres)'],
    'Value': [square_feet, bedrooms, bathrooms, year_built, garage, pool, lot_size]
})
st.table(input_df)

# Model information
st.subheader("Model Information")
st.write(f"""
- **Linear Regression** R² Score: {lin_r2:.4f} (MAE: ${lin_mae:,.2f})
- **Gradient Boosting** R² Score: {gb_r2:.4f} (MAE: ${gb_mae:,.2f})
""")

st.info("""
**Note:** This prediction is based on a model trained on synthetic data. 
Real estate prices depend on many more factors including location, market conditions, and neighborhood amenities.
""")

# Feature importance visualization
st.subheader("Feature Impact Analysis")

# Get feature importance from the models
feature_names = ['SquareFeet', 'Bedrooms', 'Bathrooms', 'YearBuilt', 'Garage', 'Pool', 'Lot_Size']
lin_coef = np.abs(lin_reg_model.coef_)
gb_imp = gb_model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Linear Regression': lin_coef,
    'Gradient Boosting': gb_imp
})

# Display the dataframe
st.dataframe(importance_df.style.format({
    'Linear Regression': '{:.4f}',
    'Gradient Boosting': '{:.4f}'
}))

# Explanation section
st.subheader("How It Works")
st.write("""
The House Price Predictor uses machine learning to estimate property values based on key features:

1. **Square Feet**: Larger homes typically have higher values
2. **Bedrooms/Bathrooms**: More bedrooms and bathrooms generally increase value
3. **Year Built**: Newer homes often command higher prices
4. **Garage Spaces**: Additional parking adds value
5. **Pool**: Swimming pools provide a premium
6. **Lot Size**: Larger lots are generally more valuable

The model was trained on a synthetic dataset with realistic price relationships between features.
""")

st.sidebar.markdown("---")
st.sidebar.info("🏠 House Price Predictor App\n\nBased on machine learning models trained on synthetic real estate data.")

# Footer
st.markdown("---")
st.markdown("Developed as part of AI/ML Engineering Internship Tasks")