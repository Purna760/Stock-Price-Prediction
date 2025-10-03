import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-positive {
        color: #00ff00;
        font-weight: bold;
    }
    .prediction-negative {
        color: #ff0000;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">🚀 Advanced Stock Price Predictor</h1>', unsafe_html=True)

# Sidebar
st.sidebar.header("📊 Configuration")

# Stock selection
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))

# Model selection
model_choice = st.sidebar.selectbox(
    "Select ML Model",
    ["Random Forest", "Gradient Boosting", "SVM", "Ensemble"]
)

# Parameters
lookback_days = st.sidebar.slider("Lookback Days", 5, 60, 30)
forecast_days = st.sidebar.slider("Forecast Days", 1, 30, 7)

# Technical Indicators without TA-Lib
def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band

def calculate_momentum(prices, window=10):
    """Calculate Price Momentum"""
    return prices.diff(window)

def calculate_price_roc(prices, window=10):
    """Calculate Price Rate of Change"""
    return ((prices - prices.shift(window)) / prices.shift(window)) * 100

@st.cache_data
def load_stock_data(ticker, start, end):
    """Load stock data from Yahoo Finance"""
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            st.error(f"No data found for ticker {ticker}")
            return None
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def calculate_technical_indicators(df):
    """Calculate technical indicators without TA-Lib"""
    df = df.copy()
    
    # Price-based indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['RSI'] = calculate_rsi(df['Close'], 14)
    
    # MACD
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = calculate_macd(df['Close'])
    
    # Bollinger Bands
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = calculate_bollinger_bands(df['Close'])
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    
    # Price trends
    df['Price_Rate_Of_Change'] = calculate_price_roc(df['Close'], 10)
    df['Momentum'] = calculate_momentum(df['Close'], 10)
    
    return df

def create_features(df, lookback=30):
    """Create features for machine learning"""
    df = df.copy()
    
    # Price features
    df['Price_Lag_1'] = df['Close'].shift(1)
    df['Price_Lag_5'] = df['Close'].shift(5)
    df['Price_Lag_10'] = df['Close'].shift(10)
    
    # Rolling statistics
    df['Rolling_Mean_7'] = df['Close'].rolling(window=7).mean()
    df['Rolling_Std_7'] = df['Close'].rolling(window=7).std()
    df['Rolling_Mean_21'] = df['Close'].rolling(window=21).mean()
    df['Rolling_Std_21'] = df['Close'].rolling(window=21).std()
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=21).std()
    
    # Price change features
    df['Daily_Return'] = df['Close'].pct_change()
    df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
    
    # Additional technical features
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Open_Close_Ratio'] = df['Open'] / df['Close']
    
    # Drop NaN values
    df = df.dropna()
    
    return df

def prepare_ml_data(df, lookback=30, forecast_days=1):
    """Prepare data for machine learning"""
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'EMA_20', 
                'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower',
                'Volume_SMA', 'Price_Rate_Of_Change', 'Momentum',
                'Price_Lag_1', 'Price_Lag_5', 'Price_Lag_10',
                'Rolling_Mean_7', 'Rolling_Std_7', 'Rolling_Mean_21', 'Rolling_Std_21',
                'Volatility', 'Daily_Return', 'Price_Range',
                'High_Low_Ratio', 'Open_Close_Ratio']
    
    # Select available features
    available_features = [f for f in features if f in df.columns]
    X = df[available_features]
    
    # Create target (future price)
    y = df['Close'].shift(-forecast_days)
    
    # Remove rows with NaN in target
    valid_indices = ~y.isna()
    X = X[valid_indices]
    y = y[valid_indices]
    
    return X, y, available_features

def train_model(X, y, model_type):
    """Train the selected machine learning model"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize model
    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
    elif model_type == "SVM":
        model = SVR(kernel='rbf', C=1.0)
    elif model_type == "Ensemble":
        # Simple ensemble of multiple models
        rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
        gb = GradientBoostingRegressor(n_estimators=50, random_state=42, max_depth=5)
        
        # For simplicity, we'll use Random Forest as ensemble
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    else:  # Default to Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    return model, scaler, X_test, y_test, y_pred

def calculate_metrics(y_true, y_pred):
    """Calculate prediction metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Direction accuracy
    if len(y_true) > 1:
        direction_true = np.diff(y_true) > 0
        direction_pred = np.diff(y_pred) > 0
        direction_accuracy = np.mean(direction_true == direction_pred)
    else:
        direction_accuracy = 0
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2 Score': r2,
        'Direction Accuracy': direction_accuracy
    }

def create_interactive_chart(df, predictions=None):
    """Create interactive Plotly chart"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price Chart with Technical Indicators', 'Volume & RSI'),
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3]
    )
    
    # Price data
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add moving averages
    if 'SMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange')),
            row=1, col=1
        )
    
    if 'EMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['EMA_20'], name='EMA 20', line=dict(color='red')),
            row=1, col=1
        )
    
    # Add Bollinger Bands
    if all(col in df.columns for col in ['BB_upper', 'BB_lower']):
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper', line=dict(color='gray', dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower', line=dict(color='gray', dash='dash')),
            row=1, col=1
        )
    
    # Volume
    colors = ['red' if row['Open'] - row['Close'] >= 0 else 'green' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors),
        row=2, col=1
    )
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        # Add RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig

# Main app logic
def main():
    # Load data
    with st.spinner('Loading stock data...'):
        data = load_stock_data(ticker, start_date, end_date)
    
    if data is None:
        st.error("Failed to load data. Please check the ticker symbol and dates.")
        return
    
    if data.empty:
        st.error("No data available for the selected ticker and date range.")
        return
    
    # Calculate technical indicators
    with st.spinner('Calculating technical indicators...'):
        data = calculate_technical_indicators(data)
    
    # Display basic info
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = data['Close'].iloc[-1]
    if len(data) > 1:
        prev_price = data['Close'].iloc[-2]
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
    else:
        prev_price = current_price
        price_change = 0
        price_change_pct = 0
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    with col2:
        st.metric("Daily Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
    with col3:
        st.metric("52W High", f"${data['High'].max():.2f}")
    with col4:
        st.metric("52W Low", f"${data['Low'].min():.2f}")
    
    # Display interactive chart
    st.plotly_chart(create_interactive_chart(data), use_container_width=True)
    
    # Machine Learning Prediction Section
    st.header("🤖 Machine Learning Predictions")
    
    # Prepare data for ML
    with st.spinner('Preparing data for machine learning...'):
        data_ml = create_features(data, lookback_days)
        if len(data_ml) == 0:
            st.error("Not enough data for machine learning after feature engineering. Try increasing the date range.")
            return
            
        X, y, feature_names = prepare_ml_data(data_ml, lookback_days, forecast_days)
    
    if len(X) == 0:
        st.error("Not enough data for machine learning. Try increasing the date range.")
        return
    
    # Train model and get predictions
    with st.spinner(f'Training {model_choice} model...'):
        try:
            model, scaler, X_test, y_test, y_pred = train_model(X, y, model_choice)
        except Exception as e:
            st.error(f"Error training model: {e}")
            return
    
    # Calculate metrics
    metrics = calculate_metrics(y_test.values, y_pred)
    
    # Display metrics
    st.subheader("📊 Model Performance")
    metric_cols = st.columns(5)
    with metric_cols[0]:
        st.metric("MAE", f"${metrics['MAE']:.2f}")
    with metric_cols[1]:
        st.metric("RMSE", f"${metrics['RMSE']:.2f}")
    with metric_cols[2]:
        st.metric("R² Score", f"{metrics['R2 Score']:.4f}")
    with metric_cols[3]:
        st.metric("Direction Accuracy", f"{metrics['Direction Accuracy']:.1%}")
    
    # Prediction vs Actual chart
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=y_test.index, y=y_test.values, name='Actual', line=dict(color='blue')))
    fig_pred.add_trace(go.Scatter(x=y_test.index, y=y_pred, name='Predicted', line=dict(color='red', dash='dash')))
    fig_pred.update_layout(title='Actual vs Predicted Prices', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # Future Prediction
    st.subheader("🔮 Future Price Prediction")
    
    try:
        # Use latest data for prediction
        latest_data = X.iloc[-1:].copy()
        latest_scaled = scaler.transform(latest_data)
        future_prediction = model.predict(latest_scaled)[0]
        
        current_price = data['Close'].iloc[-1]
        pred_change = future_prediction - current_price
        pred_change_pct = (pred_change / current_price) * 100
        
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        
        with pred_col1:
            st.metric("Current Price", f"${current_price:.2f}")
        with pred_col2:
            st.metric(f"Predicted in {forecast_days} days", f"${future_prediction:.2f}")
        with pred_col3:
            change_color = "prediction-positive" if pred_change > 0 else "prediction-negative"
            st.markdown(f'<div class="metric-card {change_color}">Expected Change: ${pred_change:.2f} ({pred_change_pct:.2f}%)</div>', unsafe_allow_html=True)
        
        # Feature Importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            st.subheader("🎯 Feature Importance")
            feature_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig_imp = go.Figure(go.Bar(
                x=feature_imp['importance'],
                y=feature_imp['feature'],
                orientation='h'
            ))
            fig_imp.update_layout(title='Feature Importance', xaxis_title='Importance')
            st.plotly_chart(fig_imp, use_container_width=True)
        
        # Trading Suggestions
        st.subheader("💡 Trading Suggestions")
        
        # Simple trading logic based on predictions and technical indicators
        suggestion = "HOLD"
        confidence = "Medium"
        
        if pred_change_pct > 2:
            suggestion = "BUY"
            confidence = "High" if metrics['Direction Accuracy'] > 0.7 else "Medium"
        elif pred_change_pct < -2:
            suggestion = "SELL"
            confidence = "High" if metrics['Direction Accuracy'] > 0.7 else "Medium"
        
        # Check RSI for overbought/oversold
        if 'RSI' in data.columns and not pd.isna(data['RSI'].iloc[-1]):
            current_rsi = data['RSI'].iloc[-1]
            if current_rsi > 70:
                suggestion = "SELL (Overbought)"
            elif current_rsi < 30:
                suggestion = "BUY (Oversold)"
        
        sug_col1, sug_col2 = st.columns(2)
        with sug_col1:
            st.info(f"**Action:** {suggestion}")
        with sug_col2:
            st.info(f"**Confidence:** {confidence}")
            
    except Exception as e:
        st.error(f"Error making prediction: {e}")
    
    # Risk Disclaimer
    st.warning("""
    **⚠️ Risk Disclaimer:** 
    This is for educational purposes only. Stock predictions are inherently uncertain. 
    Always do your own research and consult with financial advisors before making investment decisions.
    Past performance is not indicative of future results.
    """)

if __name__ == "__main__":
    main()
