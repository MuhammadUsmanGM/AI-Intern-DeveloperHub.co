import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import yfinance, but provide fallback if not available
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Set up the page configuration
st.set_page_config(
    page_title="📈 Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Stock Price Prediction Dashboard")
st.markdown("""
This application predicts future stock prices using machine learning models trained on historical data.
Select a popular stock from the sidebar or enter your own ticker symbol to get started.
""")

# Add a section with example stocks to try
st.markdown("""
### 📋 Popular Stocks to Try:
- **Technology:** AAPL (Apple), MSFT (Microsoft), GOOGL (Google), NVDA (NVIDIA)
- **Electric Vehicles:** TSLA (Tesla)
- **E-commerce:** AMZN (Amazon)
- **Social Media:** META (Meta/Facebook)
- **Streaming:** NFLX (Netflix)
""")

st.markdown("---")

# Sidebar for inputs
st.sidebar.header("📈 Stock Prediction Settings")

# Popular stock options for easy selection
popular_stocks = {
    "Apple Inc.": "AAPL",
    "Microsoft Corp.": "MSFT",
    "Amazon.com Inc.": "AMZN",
    "Alphabet Inc.": "GOOGL",
    "Tesla Inc.": "TSLA",
    "Meta Platforms": "META",
    "NVIDIA Corp.": "NVDA",
    "Netflix Inc.": "NFLX",
    "JPMorgan Chase": "JPM",
    "Procter & Gamble": "PG"
}

# Create a selection box with popular stocks
selected_stock_name = st.sidebar.selectbox(
    "Select a popular stock or enter custom ticker",
    options=list(popular_stocks.keys()) + ["Custom"]
)

# Set ticker based on selection
if selected_stock_name == "Custom":
    ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper()
else:
    ticker = popular_stocks[selected_stock_name]

# Date range selection
import datetime
today = datetime.date.today()
default_start = today - datetime.timedelta(days=365)  # 1 year ago (shorter for better data availability)

start_date = st.sidebar.date_input("Start Date", value=default_start)
end_date = st.sidebar.date_input("End Date", value=today)

# Model selection
model_type = st.sidebar.selectbox("Select Model", ["Linear Regression", "Random Forest"])

# Show instructions to user
st.sidebar.info(f"""
### How to Use:
1. Select a popular stock or enter a ticker
2. Adjust date range if needed
3. Choose prediction model
4. View analysis and predictions

**Example tickers to try:** AAPL, MSFT, GOOGL, TSLA, NVDA
""")

# Function to create sample stock data
def create_sample_stock_data(start_date, end_date):
    """Create sample stock data that mimics realistic stock movements"""
    import datetime as dt
    # Calculate the date range in days
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Create date range
    dates = pd.date_range(start=start, end=end, freq='D')
    
    # Filter to weekdays only (stock market is closed weekends)
    dates = dates[dates.weekday < 5]
    
    n = len(dates)
    
    if n == 0:
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'], index=pd.DatetimeIndex([]))
    
    # Simulate realistic stock price movements
    # Random walks with some drift and volatility
    returns = np.random.normal(0.0005, 0.02, n)  # Daily returns (0.05% avg, 2% std)
    prices = 150 * np.exp(np.cumsum(returns))  # Start at $150 with compound returns
    
    # Create realistic OHLCV data
    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    closes = prices
    
    # The first open is just the first price
    opens[0] = closes[0]
    
    # Calculate opens, highs, lows based on closes with some variation
    for i in range(1, n):
        # Use previous close as the base for next open
        opens[i] = closes[i-1]
        # Add small random variation for open
        opening_variation = np.random.normal(0, 0.005) * closes[i-1]
        opens[i] = max(0.1, opens[i] + opening_variation)  # Ensure positive
        
        # High is max of open, close, and some variation
        high_variation = abs(np.random.normal(0, 0.01) * closes[i])
        highs[i] = max(opens[i], closes[i]) + high_variation
        
        # Low is min of open, close, and some variation
        low_variation = abs(np.random.normal(0, 0.01) * closes[i])
        lows[i] = min(opens[i], closes[i]) - low_variation
        lows[i] = max(0.1, lows[i])  # Ensure positive
        
        # Ensure high >= open, close and low <= open, close
        highs[i] = max(highs[i], opens[i], closes[i])
        lows[i] = min(lows[i], opens[i], closes[i])
    
    # If we have only 1 element, fill with reasonable defaults
    if n == 1:
        opens[0] = prices[0]
        highs[0] = prices[0] * 1.02
        lows[0] = prices[0] * 0.98
        closes[0] = prices[0]
    
    # Generate volume data (correlates somewhat with volatility)
    volatility = np.abs(np.diff(closes, prepend=closes[0])) / closes
    volumes = 1_000_000 + (volatility * 10_000_000)  # Base volume with volatility factor
    volumes = np.clip(volumes, 100_000, 100_000_000)  # Reasonable volume bounds
    
    df = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    }, index=dates)
    
    return df

# Download stock data with fallback
@st.cache_data
def load_stock_data(ticker, start, end):
    # Check if the end date is in the future and adjust if necessary
    actual_end = min(end, datetime.date.today())
    
    if YFINANCE_AVAILABLE:
        try:
            # Attempt to download data for a valid past range
            data = yf.download(ticker, start=start, end=actual_end, progress=False)
            
            # Additional check: if data is empty or has very few entries
            if data.empty or len(data) < 10:
                st.warning(f"Insufficient data for {ticker}. Using sample data instead.")
                return create_sample_stock_data(start, actual_end)
            
            return data
        except Exception as e:
            st.warning(f"Could not retrieve data for {ticker}. Using sample data instead. Error: {str(e)}")
            return create_sample_stock_data(start, actual_end)
    else:
        # Create sample data if yfinance is not available
        st.info("yfinance not available. Using sample data.")
        return create_sample_stock_data(start, actual_end)

# Simple RSI calculation with error handling
def calculate_rsi(prices, window=14):
    try:
        prices = pd.Series(prices)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except:
        # Return NaN series if calculation fails
        return pd.Series([np.nan] * len(prices), index=prices.index)

# Prepare features for prediction
def prepare_features(df):
    df_features = df.copy()

    # Create technical indicators with minimal period requirements
    df_features['SMA_5'] = df_features['Close'].rolling(window=5, min_periods=2).mean()
    df_features['SMA_10'] = df_features['Close'].rolling(window=10, min_periods=2).mean()
    df_features['SMA_20'] = df_features['Close'].rolling(window=20, min_periods=2).mean()

    # Calculate RSI with error handling
    df_features['RSI'] = calculate_rsi(df_features['Close'])

    # Calculate percentage changes with fill method
    df_features['Price_Change'] = df_features['Close'].pct_change()
    df_features['Volume_Change'] = df_features['Volume'].pct_change()

    # Create target variable (next day's closing price)
    df_features['Next_Close'] = df_features['Close'].shift(-1)

    # Select features for modeling
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_5', 'SMA_10', 'SMA_20', 'RSI',
        'Price_Change', 'Volume_Change'
    ]

    # Return features and target, ensuring they're properly aligned
    features_df = df_features[feature_columns]
    target_series = df_features['Next_Close']

    return features_df, target_series

# Validate date range
if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# Load the stock data
try:
    data = load_stock_data(ticker, start_date, end_date)
    
    # Check if we have any data
    if data.empty or len(data) < 10:
        st.error("No data available for the selected stock and date range.")
        st.stop()
    
    # Clean data - remove any existing NaN
    data = data.dropna()
    
    if len(data) < 20:  # Check if we have sufficient data
        st.error("Insufficient data for the selected date range. Please try a broader range.")
        st.stop()
    
    # Prepare features
    X, y = prepare_features(data)
    
    # Handle NaN values in features and target
    # Combine X and y to align them, then remove rows with NaN
    df_combined = pd.concat([X, y], axis=1).dropna()
    
    # Check if combined dataframe is empty after dropping NaN
    if df_combined.empty or len(df_combined) < 10:
        # If there's not enough data with technical indicators, try with basic features only
        st.warning("Not enough data for all technical indicators. Using basic features only...")

        # Use only basic features that don't require rolling windows
        basic_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        basic_X = df[basic_features]
        basic_y = df['Close'].shift(-1)  # Predict next day's close

        # Align X and y
        basic_combined = pd.concat([basic_X, basic_y], axis=1).dropna()

        if basic_combined.empty or len(basic_combined) < 10:
            st.error("Not enough valid data. Please try a broader date range or a different stock.")
            st.stop()

        X = basic_combined[basic_features]
        y = basic_combined[basic_y.name]
    else:
        # Separate back into X and y after removing NaN
        feature_columns = [col for col in X.columns]
        X = df_combined[feature_columns]
        y = df_combined[y.name]
    
    if len(X) < 20:  # Make sure we have enough data for training
        st.error(f"Not enough data after feature engineering. Available: {len(X)} rows. Please try a broader date range.")
        st.stop()

    # Check if we have any data after cleaning
    if X.empty or y.empty or len(X) == 0 or len(y) == 0:
        st.error("No valid data available after processing. Please try a different stock or date range.")
        st.stop()

    # Split the data ensuring we maintain time series order
    split_idx = int(len(X) * 0.8)  # Use 80% for training
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    # Check if train/test sets are valid
    if len(X_train) == 0 or len(X_test) == 0 or len(y_train) == 0 or len(y_test) == 0:
        st.error("Not enough data to create train/test split. Please try a broader date range.")
        st.stop()

    # Further validation: Check for any remaining NaN or infinite values
    X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
    X_test = X_test.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Restrict y to match the X indices after cleaning
    y_train = y_train[y_train.index.intersection(X_train.index)]
    y_test = y_test[y_test.index.intersection(X_test.index)]
    
    # Align X and y again after cleaning
    X_train = X_train.loc[y_train.index]
    X_test = X_test.loc[y_test.index]
    
    if X_train.empty or X_test.empty:
        st.error("Data contains invalid values that could not be processed. Please try a different date range.")
        st.stop()

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model based on user selection
    if model_type == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:  # Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Display stock information
    st.subheader(f"Stock Information: {ticker.upper()}")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Price", f"${data['Close'][-1]:.2f}", 
                 f"{((data['Close'][-1] - data['Close'][-2]) / data['Close'][-2] * 100):+.2f}%")
    with col2:
        st.metric("Volume", f"{data['Volume'][-1]:,.0f}")
    with col3:
        st.metric("Date Range", f"{start_date} to {end_date}")
    
    # Plot historical prices
    st.subheader(f"Historical Prices for {ticker.upper()}")
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(data.index, data['Close'], label='Close Price', linewidth=1)
    ax.set_title(f'{ticker.upper()} Closing Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Model Performance
    st.subheader(f"Model Performance: {model_type}")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean Absolute Error", f"${mae:.2f}")
    with col2:
        st.metric("RMSE", f"${rmse:.2f}")
    with col3:
        st.metric("R² Score", f"{r2:.4f}")
    
    # Plot predictions vs actual
    st.subheader("Model Predictions vs Actual")
    fig, ax = plt.subplots(figsize=(15, 6))
    test_dates = y_test.index
    ax.plot(test_dates, y_test.values, label='Actual Price', linewidth=2)
    ax.plot(test_dates, y_pred, label='Predicted Price', linewidth=2, alpha=0.8)
    ax.set_title(f'{ticker.upper()} - {model_type}: Actual vs Predicted')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Feature importance (for Random Forest)
    if model_type == "Random Forest":
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        st.bar_chart(feature_importance.set_index('Feature'))
    
    # Prediction for next day
    st.subheader("Next Day Prediction")
    try:
        latest_features = X.iloc[-1:].values
        latest_features_scaled = scaler.transform(latest_features)
        next_day_pred = model.predict(latest_features_scaled)[0]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Next Close", f"${next_day_pred:.2f}")
        with col2:
            st.metric("Current Close", f"${data['Close'][-1]:.2f}")
    except:
        st.warning("Could not generate next day prediction due to data issues.")
    
    # Data overview
    st.subheader("Data Overview")
    st.write(f"Total Days: {len(data)}")
    st.write(f"Date Range: {data.index[0].date()} to {data.index[-1].date()}")
    st.write(f"Price Range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    
    # Display last few rows of data
    st.subheader("Recent Data")
    st.dataframe(data[['Open', 'High', 'Low', 'Close', 'Volume']].tail())
    
    st.info(f"""
    **Model Information:** {model_type} trained on {len(X_train)} days of historical data 
    to predict the next day's closing price. Performance metrics are calculated on 
    the hold-out test set ({len(X_test)} days).
    """)
    
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    
    if not YFINANCE_AVAILABLE:
        st.warning("The yfinance library is not installed. Please install it using: `pip install yfinance`")
        st.info("The app will run with sample data instead of real stock data.")
    else:
        st.info("Please check your internet connection and the stock ticker symbol. You can also try a different date range.")
        st.info("The error might also be due to insufficient data or market holidays in the selected date range.")

# Footer
st.markdown("---")
st.markdown("Developed as part of AI/ML Engineering Internship Tasks")
st.markdown("This application uses historical stock data to predict future prices using machine learning models.")