# ui/app.py

import os
import sys
import streamlit as st
import yaml
import pandas as pd
import matplotlib.pyplot as plt

# Add the project root to sys.path.
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Load configuration.
config_path = os.path.join(parent_dir, "config", "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Resolve the historical data file path.
data_file_rel = config.get("historical_data", "data/historical_data.csv")
data_file = os.path.abspath(os.path.join(parent_dir, data_file_rel))
if not os.path.exists(data_file):
    st.warning(f"Data file not found: {data_file}")
else:
    st.write(f"Using data file: {data_file}")

from utils.data_access import load_historical_data
from the_backtesting.engine import BacktestEngine
from ml_models.sample_strategy import AdvancedStrategy
from ml_models.ml_trading_model import train_lstm_model, adjust_strategy_with_predictions

st.title("Robust Stock Backtesting & ML Trading Tool")

st.sidebar.header("Ticker Selection")
st.sidebar.markdown(
    """
    **Ticker:**  
    Enter the ticker symbol of the stock you wish to analyze.  
    Examples:
    - **AAPL** for Apple  
    - **MSFT** for Microsoft  
    - **GOOGL** for Alphabet  
    """
)
ticker = st.sidebar.text_input("Ticker Symbol", value=config.get("ticker", "AAPL"))

st.sidebar.header("Strategy & Risk Management")
short_window = st.sidebar.number_input("Short MA Window", min_value=1, value=50)
long_window = st.sidebar.number_input("Long MA Window", min_value=1, value=200)
base_trade_size = st.sidebar.number_input("Base Trade Size", min_value=1, value=config.get("trade_size", 100))
initial_cash = st.sidebar.number_input("Initial Cash", value=config.get("initial_cash", 100000))
commission = st.sidebar.number_input("Commission Rate", value=config.get("commission", 0.001))
stop_loss_pct = st.sidebar.slider("Stop-Loss (%)", min_value=0.5, max_value=10.0, value=2.0) / 100.0
take_profit_pct = st.sidebar.slider("Take-Profit (%)", min_value=0.5, max_value=20.0, value=4.0) / 100.0
entry_threshold = st.sidebar.slider("Entry Threshold (%)", min_value=0.1, max_value=5.0, value=1.0) / 100.0
exit_threshold = st.sidebar.slider("Exit Threshold (%)", min_value=0.1, max_value=5.0, value=0.5) / 100.0

# Optimization is now automatic unless disabled.
use_optimization = st.sidebar.checkbox("Use Automated Optimization", value=True)

data_file_input = st.sidebar.text_input("Historical Data File", value=config.get("historical_data", "data/historical_data.csv"))
if not os.path.isabs(data_file_input):
    data_file_input = os.path.abspath(os.path.join(parent_dir, data_file_input))

# Reset Button: Delete current CSV file.
if st.sidebar.button("Reset Data"):
    if os.path.exists(data_file):
        os.remove(data_file)
        st.success("Data file deleted. You can now fetch new data.")
    else:
        st.info("No data file exists to delete.")

st.sidebar.header("Actions")
if st.sidebar.button("Fetch Full Data & Enrich"):
    st.write("Fetching and enriching full data (~30 days) from Alpha Vantage & NewsAPI for ticker:", ticker)
    import subprocess
    subprocess.run(["python", os.path.join(parent_dir, "data_ingestion", "producer.py"), "--ticker", ticker])
    st.success("Data ingested and saved.")

if st.sidebar.button("Train ML Model"):
    st.write("Training LSTM model on full historical data...")
    model, scaler = train_lstm_model(data_file_input, look_back=60, epochs=5, batch_size=32)
    st.success("ML Model trained. See the plot for actual vs. predicted prices.")

if st.sidebar.button("Optimize & Run Backtest"):
    st.write("Loading full historical data...")
    data = load_historical_data(data_file_input)
    st.write("Data loaded:", data.shape[0], "rows")
    
    # Instantiate strategy with current slider inputs.
    strategy = AdvancedStrategy(short_window=short_window, long_window=long_window,
                                  entry_threshold=entry_threshold, exit_threshold=exit_threshold,
                                  base_trade_size=base_trade_size)
    
    # If optimization is enabled, automatically optimize the strategy parameters.
    if use_optimization:
        st.write("Optimizing strategy parameters automatically...")
        opt_params = strategy.optimize_parameters(data, initial_cash, commission, base_trade_size)
        st.write("Optimized Parameters:", opt_params)
    else:
        st.write("Using manual parameters from UI.")
    
    st.write("Adjusting strategy based on ML predictions...")
    model, scaler = train_lstm_model(data_file_input, look_back=60, epochs=3, batch_size=32)
    strategy = adjust_strategy_with_predictions(strategy, model, scaler, data, look_back=60)
    st.write("Final strategy parameters:", {
        "entry_threshold": strategy.entry_threshold,
        "exit_threshold": strategy.exit_threshold,
        "base_trade_size": strategy.base_trade_size
    })
    
    engine = BacktestEngine(data, strategy, initial_cash=initial_cash, commission=commission,
                              trade_size=strategy.base_trade_size, slippage_pct=0.001)
    performance = engine.run()
    
    st.write("### Backtest Performance")
    st.write("Final Cash: $", round(performance["Final Cash"], 2))
    st.write("Total Return: ", round(performance["Total Return"] * 100, 2), "%")
    st.write("Number of Trades: ", performance["Number of Trades"])
    st.write("Average Profit per Trade: $", round(performance["Average Profit per Trade"], 2))
    st.write("Max Profit: $", round(performance["Max Profit"], 2))
    st.write("Max Loss: $", round(performance["Max Loss"], 2))
    
    trades = performance["Trades"]
    if trades:
        trade_df = pd.DataFrame([{
            "Entry Time": t.entry_time,
            "Entry Price": t.entry_price,
            "Exit Time": t.exit_time,
            "Exit Price": t.exit_price,
            "Profit": t.profit(),
            "Log": str(t.log)
        } for t in trades if t.is_closed()])
        st.write("### Trade Details")
        st.dataframe(trade_df)
    
    profits = [t.profit() for t in trades if t.is_closed()]
    if profits:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(profits) + 1), profits, marker="o", color="red", label="Trade Profit")
        ax.set_title("Profit per Trade")
        ax.set_xlabel("Trade Number")
        ax.set_ylabel("Profit ($)")
        ax.legend()
        st.pyplot(fig)
