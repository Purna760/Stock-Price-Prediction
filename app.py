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
        font-size: 2.5rem;
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
        color: green;
        font-weight: bold;
    }
    .prediction-negative {
        color: red;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">🚀 Advanced Stock Price Predictor</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📊 Configuration")

# Stock selection with default value
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper()

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
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except:
        return pd.Series([np.nan] * len(prices), index=prices.index)

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    try:
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    except:
        nan_series = pd.Series([np.nan] * len(prices), index=prices.index)
        return nan_series, nan_series, nan_series

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    try:
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, rolling_mean, lower_band
    except:
        nan_series = pd.Series([np.nan] * len(prices), index=prices.index)
        return nan_series, nan_series, nan_series

def calculate_momentum(prices, window=10):
    """Calculate Price Momentum"""
    try:
        return prices.diff(window)
    except:
        return pd.Series([np.nan] * len(prices), index=prices.index)

def calculate_price_roc(prices, window=10):
    """Calculate Price Rate of Change"""
    try:
        return ((prices - prices.shift(window)) / prices.shift(window)) * 100
    except:
        return pd.Series([np.nan] * len(prices), index=prices.index)

def load_stock_data_simple(ticker_symbol):
    """Simple and robust stock data loading - NO DATE FILTERING"""
    try:
        # Create ticker object
        stock = yf.Ticker(ticker_symbol)
        
        # Get historical data for 3 years (enough data for ML)
        # Using period instead of dates to avoid timezone issues
        data = stock.history(period="3y")
        
        if data.empty:
            return None, f"No data found for {ticker_symbol}"
            
        # Remove timezone information completely
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        # Ensure numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove any rows with NaN in essential columns
        data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        if len(data) < 30:
            return None, f"Only {len(data)} days of data available. Need at least 30 days."
            
        return data, "Success"
        
    except Exception as e:
        return None, f"Error loading {ticker_symbol}: {str(e)}"

@st.cache_data
def load_stock_data(_ticker):
    """Cached version of stock data loading"""
    return load_stock_data_simple(_ticker)

@st.cache_data
def calculate_technical_indicators(df):
    """Calculate technical indicators without TA-Lib"""
    try:
        df = df.copy()
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            return df
        
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
    except Exception as e:
        return df

def create_features(df, lookback=30):
    """Create features for machine learning"""
    try:
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
    except Exception as e:
        return df

def prepare_ml_data(df, lookback=30, forecast_days=1):
    """Prepare data for machine learning"""
    try:
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'EMA_20', 
                    'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower',
                    'Volume_SMA', 'Price_Rate_Of_Change', 'Momentum',
                    'Price_Lag_1', 'Price_Lag_5', 'Price_Lag_10',
                    'Rolling_Mean_7', 'Rolling_Std_7', 'Rolling_Mean_21', 'Rolling_Std_21',
                    'Volatility', 'Daily_Return', 'Price_Range',
                    'High_Low_Ratio', 'Open_Close_Ratio']
        
        # Select available features
        available_features = [f for f in features if f in df.columns]
        if not available_features:
            return pd.DataFrame(), pd.Series(), []
            
        X = df[available_features]
        
        # Create target (future price)
        y = df['Close'].shift(-forecast_days)
        
        # Remove rows with NaN in target
        valid_indices = ~y.isna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        return X, y, available_features
    except Exception as e:
        return pd.DataFrame(), pd.Series(), []

def train_model(X, y, model_type):
    """Train the selected machine learning model"""
    try:
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
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        return model, scaler, X_test, y_test, y_pred
    except Exception as e:
        return None, None, None, None, None

def calculate_metrics(y_true, y_pred):
    """Calculate prediction metrics"""
    try:
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
    except:
        return {
            'MAE': 0, 'MSE': 0, 'RMSE': 0, 'R2 Score': 0, 'Direction Accuracy': 0
        }

def create_interactive_chart(df):
    """Create interactive Plotly chart"""
    try:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Price Chart', 'Volume'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Price data
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'],
                name='Close Price',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Add moving averages if available
        if 'SMA_20' in df.columns and not df['SMA_20'].isna().all():
            fig.add_trace(
                go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange')),
                row=1, col=1
            )
        
        # Volume
        colors = ['red' if row['Open'] - row['Close'] >= 0 else 'green' for _, row in df.iterrows()]
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    except Exception as e:
        return go.Figure()

def safe_metric_display(metric_name, value, format_str="${:.2f}"):
    """Safely display a metric with error handling"""
    try:
        if pd.isna(value) or value is None:
            st.metric(metric_name, "N/A")
        else:
            st.metric(metric_name, format_str.format(float(value)))
    except:
        st.metric(metric_name, "Error")

# Main app logic
def main():
    # Quick test buttons
    st.sidebar.header("🚀 Quick Test")
    popular_tickers = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "NFLX"]
    
    # Display popular tickers as buttons
    selected_ticker = ticker
    
    # Create buttons for popular tickers
    for i in range(0, len(popular_tickers), 2):
        cols = st.sidebar.columns(2)
        tick1 = popular_tickers[i]
        with cols[0]:
            if st.button(tick1, key=f"btn_{tick1}", use_container_width=True):
                selected_ticker = tick1
        if i + 1 < len(popular_tickers):
            tick2 = popular_tickers[i + 1]
            with cols[1]:
                if st.button(tick2, key=f"btn_{tick2}", use_container_width=True):
                    selected_ticker = tick2
    
    # Use the selected ticker
    current_ticker = selected_ticker
    
    try:
        # Load data
        with st.spinner(f'📥 Loading data for {current_ticker}...'):
            data, message = load_stock_data(current_ticker)
        
        if data is None:
            st.error(f"❌ {message}")
            
            st.info("💡 **Try these popular stocks:**")
            # Display popular tickers in the main area
            main_cols = st.columns(4)
            for i, tick in enumerate(popular_tickers):
                with main_cols[i % 4]:
                    if st.button(f"📈 {tick}", key=f"main_btn_{tick}", use_container_width=True):
                        # Use session state to remember the selection
                        st.session_state.selected_ticker = tick
                        st.rerun()
            return
        
        st.success(f"✅ Successfully loaded {len(data)} days of data for {current_ticker}")
        
        # Show data preview
        with st.expander("📋 Data Preview (Last 10 days)"):
            st.dataframe(data.tail(10))
        
        # Calculate technical indicators
        with st.spinner('Calculating technical indicators...'):
            data = calculate_technical_indicators(data)
        
        # Display basic info
        st.subheader("📈 Stock Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        # Safely get current price and handle potential NaN values
        current_price = data['Close'].iloc[-1] if not data['Close'].empty else 0
        current_price = float(current_price) if not pd.isna(current_price) else 0
        
        # Calculate price change safely
        if len(data) > 1:
            prev_price = data['Close'].iloc[-2] if not data['Close'].empty else current_price
            prev_price = float(prev_price) if not pd.isna(prev_price) else current_price
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
        else:
            prev_price = current_price
            price_change = 0
            price_change_pct = 0
        
        with col1:
            safe_metric_display("Current Price", current_price)
        with col2:
            try:
                st.metric("Daily Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
            except:
                st.metric("Daily Change", "N/A")
        with col3:
            safe_metric_display("All Time High", data['High'].max())
        with col4:
            safe_metric_display("All Time Low", data['Low'].min())
        
        # Display interactive chart
        st.subheader("📊 Price Chart")
        chart = create_interactive_chart(data)
        st.plotly_chart(chart, use_container_width=True)
        
        # Check if we have enough data for ML
        if len(data) < 50:
            st.warning("⚠️ Limited data available. ML predictions work better with more historical data.")
            return
        
        # Machine Learning Prediction Section
        st.header("🤖 Machine Learning Predictions")
        
        # Prepare data for ML
        with st.spinner('Preparing data for machine learning...'):
            data_ml = create_features(data, lookback_days)
            if data_ml.empty:
                st.warning("⚠️ Not enough data for machine learning after processing.")
                return
                
            X, y, feature_names = prepare_ml_data(data_ml, lookback_days, forecast_days)
        
        if X.empty or y.empty or len(X) < 20:
            st.warning("⚠️ Insufficient data for training. Need at least 20 data points.")
            return
        
        # Train model and get predictions
        with st.spinner(f'Training {model_choice} model...'):
            model, scaler, X_test, y_test, y_pred = train_model(X, y, model_choice)
            
            if model is None:
                st.error("❌ Model training failed")
                return
        
        # Calculate metrics
        metrics = calculate_metrics(y_test.values, y_pred)
        
        # Display metrics
        st.subheader("📊 Model Performance")
        metric_cols = st.columns(5)
        with metric_cols[0]:
            safe_metric_display("MAE", metrics['MAE'])
        with metric_cols[1]:
            safe_metric_display("RMSE", metrics['RMSE'])
        with metric_cols[2]:
            st.metric("R² Score", f"{metrics['R2 Score']:.4f}")
        with metric_cols[3]:
            st.metric("Direction Accuracy", f"{metrics['Direction Accuracy']:.1%}")
        
        # Prediction vs Actual chart
        if len(y_test) > 0:
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
            
            pred_change = future_prediction - current_price
            pred_change_pct = (pred_change / current_price) * 100 if current_price != 0 else 0
            
            pred_col1, pred_col2, pred_col3 = st.columns(3)
            
            with pred_col1:
                safe_metric_display("Current Price", current_price)
            with pred_col2:
                safe_metric_display(f"Predicted Price", future_prediction)
            with pred_col3:
                change_class = "prediction-positive" if pred_change > 0 else "prediction-negative"
                st.markdown(f'<div class="{change_class}">Expected Change: ${pred_change:.2f} ({pred_change_pct:.2f}%)</div>', unsafe_allow_html=True)
            
            # Trading Suggestions
            st.subheader("💡 Trading Suggestions")
            
            suggestion = "HOLD"
            confidence = "Medium"
            
            if pred_change_pct > 3:
                suggestion = "STRONG BUY 🟢"
                confidence = "High"
            elif pred_change_pct > 1:
                suggestion = "BUY 🟢"
                confidence = "Medium"
            elif pred_change_pct < -3:
                suggestion = "STRONG SELL 🔴"
                confidence = "High"
            elif pred_change_pct < -1:
                suggestion = "SELL 🔴"
                confidence = "Medium"
            else:
                suggestion = "HOLD 🟡"
                confidence = "Low"
            
            sug_col1, sug_col2 = st.columns(2)
            with sug_col1:
                st.info(f"**Action:** {suggestion}")
            with sug_col2:
                st.info(f"**Confidence:** {confidence}")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
        
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        st.info("💡 Try clicking one of the popular stock buttons in the sidebar")
    
    # Risk Disclaimer
    st.warning("""
    **⚠️ Risk Disclaimer:** 
    This is for educational purposes only. Stock predictions are inherently uncertain. 
    Always do your own research and consult with financial advisors before making investment decisions.
    Past performance is not indicative of future results.
    """)

if __name__ == "__main__":
    main()
