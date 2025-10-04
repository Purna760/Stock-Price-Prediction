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
    .stock-button {
        width: 100%;
        margin: 2px 0;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">🚀 Live Stock Price Predictor</h1>', unsafe_allow_html=True)

# Initialize session state
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = "AAPL"

# Popular tickers with their actual current prices (for verification)
popular_tickers = {
    "AAPL": "Apple Inc.",
    "TSLA": "Tesla Inc.", 
    "GOOGL": "Google (Alphabet)",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "NVDA": "NVIDIA",
    "NFLX": "Netflix"
}

# Sidebar
st.sidebar.header("📊 Configuration")

# Display current selection prominently
st.sidebar.markdown(f"### 🔍 Analyzing: **{st.session_state.current_ticker}**")
st.sidebar.markdown(f"*{popular_tickers.get(st.session_state.current_ticker, '')}*")

# Quick stock buttons
st.sidebar.header("🚀 Quick Stocks")
st.sidebar.write("Click any stock to analyze:")

# Create buttons in a grid
cols = st.sidebar.columns(2)
button_clicked = False

for i, (ticker, name) in enumerate(popular_tickers.items()):
    col = cols[i % 2]
    with col:
        if st.button(ticker, key=f"btn_{ticker}", use_container_width=True):
            st.session_state.current_ticker = ticker
            button_clicked = True

# Manual ticker input
st.sidebar.header("🔍 Custom Stock")
manual_ticker = st.sidebar.text_input("Or enter any stock ticker:", value=st.session_state.current_ticker).upper()

if manual_ticker and manual_ticker != st.session_state.current_ticker:
    st.session_state.current_ticker = manual_ticker
    button_clicked = True

# Model settings
st.sidebar.header("🤖 ML Settings")
model_choice = st.sidebar.selectbox(
    "Select ML Model",
    ["Random Forest", "Gradient Boosting", "SVM", "Ensemble"]
)

lookback_days = st.sidebar.slider("Lookback Days", 5, 60, 30)
forecast_days = st.sidebar.slider("Forecast Days", 1, 30, 7)

# Force refresh if button was clicked
if button_clicked:
    st.rerun()

# NEW: Live data loading function without caching issues
def load_live_stock_data(ticker_symbol):
    """Load live stock data with proper error handling"""
    try:
        st.info(f"🔄 Fetching LIVE data for {ticker_symbol}...")
        
        # Create ticker object
        stock = yf.Ticker(ticker_symbol)
        
        # Get stock info to verify it's a valid ticker
        info = stock.info
        company_name = info.get('longName', ticker_symbol)
        st.success(f"✅ Found: {company_name}")
        
        # Get historical data - using multiple periods to ensure fresh data
        data_1y = stock.history(period="1y")
        data_6mo = stock.history(period="6mo")
        data_3mo = stock.history(period="3mo")
        
        # Use the most recent data available
        data = data_1y if not data_1y.empty else data_6mo if not data_6mo.empty else data_3mo
        
        if data.empty:
            return None, f"No historical data found for {ticker_symbol}"
        
        # Reset index and handle dates
        data = data.reset_index()
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return None, f"Missing data columns: {missing_cols}"
        
        # Convert to numeric and clean
        for col in required_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove rows with NaN in essential columns
        data = data.dropna(subset=required_cols)
        
        if len(data) < 30:
            return None, f"Only {len(data)} trading days available. Need at least 30 days."
        
        # Set Date as index
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.set_index('Date')
        
        # Get current live price
        live_data = stock.history(period="1d")
        if not live_data.empty:
            current_live_price = live_data['Close'].iloc[-1]
            st.success(f"📊 Live Price: ${current_live_price:.2f}")
        else:
            current_live_price = data['Close'].iloc[-1]
            st.info(f"📊 Latest Close: ${current_live_price:.2f}")
        
        return data, f"Successfully loaded {len(data)} trading days"
        
    except Exception as e:
        return None, f"Error loading {ticker_symbol}: {str(e)}"

# Technical indicators
def calculate_technical_indicators(df):
    """Calculate basic technical indicators"""
    try:
        df = df.copy()
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_upper'] = df['Close'].rolling(20).mean() + (df['Close'].rolling(20).std() * 2)
        df['BB_lower'] = df['Close'].rolling(20).mean() - (df['Close'].rolling(20).std() * 2)
        
        return df
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        return df

def create_features(df, lookback=30):
    """Create features for machine learning"""
    try:
        df = df.copy()
        
        # Price features
        for lag in [1, 2, 3, 5, 10]:
            df[f'Price_Lag_{lag}'] = df['Close'].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 21]:
            df[f'Rolling_Mean_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'Rolling_Std_{window}'] = df['Close'].rolling(window=window).std()
        
        # Price movements
        df['Daily_Return'] = df['Close'].pct_change()
        df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Volatility'] = df['Close'].rolling(window=21).std()
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    except Exception as e:
        st.error(f"Error creating features: {e}")
        return df

def prepare_ml_data(df, forecast_days=1):
    """Prepare data for machine learning"""
    try:
        # Select features
        feature_columns = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Volume']]
        X = df[feature_columns]
        
        # Create target (future price)
        y = df['Close'].shift(-forecast_days)
        
        # Remove rows with NaN in target
        valid_indices = ~y.isna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        return X, y, feature_columns
    except Exception as e:
        st.error(f"Error preparing ML data: {e}")
        return pd.DataFrame(), pd.Series(), []

def train_model(X, y, model_type):
    """Train machine learning model"""
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize and train model
        if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        elif model_type == "Gradient Boosting":
            model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
        elif model_type == "SVM":
            model = SVR(kernel='rbf', C=1.0)
        else:  # Ensemble
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        return model, scaler, X_test, y_test, y_pred
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None, None, None, None

def create_stock_chart(df, ticker):
    """Create interactive stock chart"""
    try:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'{ticker} Price Chart', 'Volume & RSI'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Price line
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Moving averages
        if 'SMA_20' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange')),
                row=1, col=1
            )
        
        # Volume
        colors = ['red' if row['Open'] > row['Close'] else 'green' for _, row in df.iterrows()]
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors, opacity=0.6),
            row=2, col=1
        )
        
        # RSI if available
        if 'RSI' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig.update_layout(height=700, showlegend=True, title=f"{ticker} Stock Analysis")
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return go.Figure()

# Main application
def main():
    current_ticker = st.session_state.current_ticker
    
    # Display header with current stock
    st.header(f"📈 Live Analysis: {current_ticker}")
    st.write(f"**Company:** {popular_tickers.get(current_ticker, 'Unknown')}")
    
    try:
        # Load live data
        data, message = load_live_stock_data(current_ticker)
        
        if data is None:
            st.error(f"❌ {message}")
            st.info("💡 Try one of these popular stocks:")
            
            # Show quick stock buttons in main area
            cols = st.columns(4)
            for i, (ticker, name) in enumerate(popular_tickers.items()):
                with cols[i % 4]:
                    if st.button(f"📈 {ticker}", key=f"main_btn_{ticker}", use_container_width=True):
                        st.session_state.current_ticker = ticker
                        st.rerun()
            return
        
        # Display data verification
        st.success(f"✅ {message}")
        
        # Show data statistics to verify it's different for each stock
        with st.expander("🔍 Data Verification & Statistics"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Data Points", len(data))
            with col2:
                st.metric("Date Range", f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            with col3:
                st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
            with col4:
                price_change = data['Close'].iloc[-1] - data['Close'].iloc[0]
                st.metric("Total Change", f"${price_change:.2f}")
            
            st.dataframe(data.tail(10))
        
        # Calculate technical indicators
        with st.spinner('Calculating technical indicators...'):
            data = calculate_technical_indicators(data)
        
        # Display key metrics
        st.subheader("📊 Key Metrics")
        metric_cols = st.columns(4)
        
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        
        with metric_cols[0]:
            st.metric("Current Price", f"${current_price:.2f}")
        with metric_cols[1]:
            st.metric("Daily Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
        with metric_cols[2]:
            st.metric("All Time High", f"${data['High'].max():.2f}")
        with metric_cols[3]:
            st.metric("All Time Low", f"${data['Low'].min():.2f}")
        
        # Display chart
        st.subheader("📈 Price Analysis")
        chart = create_stock_chart(data, current_ticker)
        st.plotly_chart(chart, use_container_width=True)
        
        # Machine Learning Section
        if len(data) >= 50:
            st.header("🤖 Machine Learning Predictions")
            
            # Prepare data for ML
            with st.spinner('Preparing data for machine learning...'):
                data_ml = create_features(data, lookback_days)
                if not data_ml.empty:
                    X, y, features = prepare_ml_data(data_ml, forecast_days)
                    
                    if not X.empty and len(X) >= 20:
                        # Train model
                        with st.spinner(f'Training {model_choice} model...'):
                            model, scaler, X_test, y_test, y_pred = train_model(X, y, model_choice)
                            
                            if model is not None:
                                # Calculate metrics
                                mae = mean_absolute_error(y_test, y_pred)
                                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                                r2 = r2_score(y_test, y_pred)
                                
                                # Display metrics
                                st.subheader("📊 Model Performance")
                                perf_cols = st.columns(4)
                                with perf_cols[0]:
                                    st.metric("MAE", f"${mae:.2f}")
                                with perf_cols[1]:
                                    st.metric("RMSE", f"${rmse:.2f}")
                                with perf_cols[2]:
                                    st.metric("R² Score", f"{r2:.4f}")
                                
                                # Future prediction
                                st.subheader("🔮 Future Price Prediction")
                                try:
                                    latest_data = X.iloc[-1:].copy()
                                    latest_scaled = scaler.transform(latest_data)
                                    future_price = model.predict(latest_scaled)[0]
                                    future_change = future_price - current_price
                                    future_change_pct = (future_change / current_price) * 100
                                    
                                    pred_cols = st.columns(3)
                                    with pred_cols[0]:
                                        st.metric("Current", f"${current_price:.2f}")
                                    with pred_cols[1]:
                                        st.metric(f"Predicted ({forecast_days} days)", f"${future_price:.2f}")
                                    with pred_cols[2]:
                                        change_type = "prediction-positive" if future_change > 0 else "prediction-negative"
                                        st.markdown(f'<div class="{change_type}">Change: ${future_change:.2f} ({future_change_pct:.2f}%)</div>', unsafe_allow_html=True)
                                    
                                    # Trading suggestion
                                    st.subheader("💡 Trading Suggestion")
                                    if future_change_pct > 2:
                                        st.success("**BUY** 🟢 - Positive momentum expected")
                                    elif future_change_pct < -2:
                                        st.error("**SELL** 🔴 - Negative momentum expected")
                                    else:
                                        st.info("**HOLD** 🟡 - Neutral outlook")
                                        
                                except Exception as e:
                                    st.error(f"Prediction error: {e}")
        
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        st.info("💡 Try selecting a different stock or refreshing the page")
    
    # Risk disclaimer
    st.warning("""
    **⚠️ Risk Disclaimer:** 
    This is for educational purposes only. Stock predictions are inherently uncertain. 
    Always do your own research and consult with financial advisors before making investment decisions.
    Past performance is not indicative of future results.
    """)

if __name__ == "__main__":
    main()
