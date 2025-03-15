# Robust Stock Backtesting & ML Trading Tool

This production-ready tool ingests enriched historical data via Alpha Vantage and NewsAPI, publishes data to Kafka, and saves it to CSV. It then uses the enriched data to:

- Simulate historical trades via a backtesting engine (with realistic commission and trade size).
- Train an ML (LSTM) model on multiple features (price, volume, technical indicators, news sentiment) to help refine trading strategies.
- Provide an interactive Streamlit UI to adjust parameters, trigger ingestion, run ML training, and simulate backtests.

## Setup Instructions

1. **Obtain API Keys:**
   - [Alpha Vantage API Key](https://www.alphavantage.co/support/#api-key)
   - [NewsAPI.org Key](https://newsapi.org/)
   - Kafka broker access (e.g., free tier via [Confluent Cloud](https://www.confluent.io/get-started/))

2. **Configure the Tool:**
   - Edit `config/config.yaml` with your API keys, Kafka settings, and simulation parameters.

3. **Install Dependencies:**
      ```bash
      pip install -r requirements.txt
      ```

4. **Data Ingestion:**

   Run the ingestion script to fetch and enrich historical data:
      ```bash
      python data_ingestion/producer.py
      ```
   This script fetches data from Alpha Vantage, enriches it with news sentiment from NewsAPI, publishes each record to Kafka, and saves all data to data/historical_data.csv.

6. **Launch the User Interface:**
      ```bash
      streamlit run ui/app.py
      ```
   Use the sidebar to trigger data ingestion, train the ML model, or run a backtest.

7. **Backtesting & ML:**
   - Adjust strategy parameters (e.g., moving average windows) via the UI.
   - Run backtests to see simulated trade performance.
   - Train the ML model to forecast future prices and help refine strategies.
