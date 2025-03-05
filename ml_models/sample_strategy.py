# ml_models/sample_strategy.py

from the_backtesting.strategy import Strategy
import numpy as np
from the_backtesting.engine import BacktestEngine
import pandas as pd

class AdvancedStrategy(Strategy):
    def __init__(self, short_window=50, long_window=200, entry_threshold=0.01, exit_threshold=0.005, base_trade_size=100, force_trade_threshold=0.005):
        self.short_window = short_window
        self.long_window = long_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.base_trade_size = base_trade_size
        self.force_trade_threshold = force_trade_threshold
        self.last_signal = None
        self.max_buy_signal = (0, None, None)

    def on_bar(self, time, row, data):
        idx = data.index.get_indexer_for([time])[0]
        if idx < self.long_window:
            return None

        short_ma = data["Close"].iloc[idx - self.short_window:idx].mean()
        long_ma = data["Close"].iloc[idx - self.long_window:idx].mean()
        pct_diff = (short_ma - long_ma) / long_ma

        if pct_diff > 0 and pct_diff > self.max_buy_signal[0]:
            self.max_buy_signal = (pct_diff, time, row)

        if pct_diff > self.entry_threshold and self.last_signal != "buy":
            self.last_signal = "buy"
            return {
                "action": "buy",
                "size": self.base_trade_size,
                "stop_loss": row["Close"] * (1 - 0.02),
                "take_profit": row["Close"] * (1 + 0.04)
            }
        elif pct_diff < -self.exit_threshold and self.last_signal != "sell":
            self.last_signal = "sell"
            return {"action": "sell", "reason": "Exit threshold reached"}
        return None

    def optimize_parameters(self, data, initial_cash, commission, trade_size):
        best_return = -np.inf
        best_params = {
            "entry_threshold": self.entry_threshold,
            "exit_threshold": self.exit_threshold,
            "base_trade_size": self.base_trade_size
        }
        candidate_entry = [0.005, 0.01, 0.015]
        candidate_exit = [0.0025, 0.005, 0.0075]
        candidate_trade_size = [50, 100, 150]

        for et in candidate_entry:
            for ex in candidate_exit:
                for ts in candidate_trade_size:
                    test_strategy = AdvancedStrategy(
                        short_window=self.short_window,
                        long_window=self.long_window,
                        entry_threshold=et,
                        exit_threshold=ex,
                        base_trade_size=ts,
                        force_trade_threshold=self.force_trade_threshold
                    )
                    engine = BacktestEngine(data, test_strategy, initial_cash=initial_cash, 
                                            commission=commission, trade_size=ts, slippage_pct=0.001)
                    performance = engine.run()
                    ret = performance["Total Return"]
                    if ret > best_return:
                        best_return = ret
                        best_params = {"entry_threshold": et, "exit_threshold": ex, "base_trade_size": ts}
        self.entry_threshold = best_params["entry_threshold"]
        self.exit_threshold = best_params["exit_threshold"]
        self.base_trade_size = best_params["base_trade_size"]
        return best_params
