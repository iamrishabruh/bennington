import pandas as pd

def load_historical_data(filepath):
    expected_headers = ["Time", "Open", "High", "Low", "Close", "Volume", "News_Sentiment", "Time_Str"]
    data = pd.read_csv(filepath, names=expected_headers, header=0, parse_dates=["Time"], index_col="Time")
    data = data.sort_index()
    return data

