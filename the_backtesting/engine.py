# backtesting/engine.py

import pandas as pd
import numpy as np
import random
from the_backtesting.trade import Trade

class BacktestEngine:
    def __init__(self, data: pd.DataFrame, strategy, initial_cash=100000, commission=0.001, 
                 trade_size=100, slippage_pct=0.001):
        self.data = data.sort_index()
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0
        self.trade_size = trade_size
        self.commission = commission
        self.slippage_pct = slippage_pct
        self.trades = []
        self.logs = []

    def simulate_slippage(self, price):
        return price * (1 + random.uniform(-self.slippage_pct, self.slippage_pct))

    def run(self):
        for current_time, row in self.data.iterrows():
            signal = self.strategy.on_bar(current_time, row, self.data)
            self.logs.append((current_time, f"Signal: {signal}"))
            if signal and signal.get("action") == "buy" and self.position == 0:
                self.execute_buy(current_time, row["Close"], signal)
            elif signal and signal.get("action") == "sell" and self.position > 0:
                self.execute_sell(current_time, row["Close"], signal)
        if self.position > 0:
            last_time = self.data.index[-1]
            last_price = self.data.iloc[-1]["Close"]
            self.execute_sell(last_time, last_price, {"action": "sell", "reason": "End of Data"})

        # If no trade occurred, force a trade automatically if the best observed buy signal exceeds a threshold.
        if len(self.trades) == 0:
            max_signal, force_time, force_row = self.strategy.max_buy_signal
            if max_signal >= self.strategy.force_trade_threshold and force_time is not None:
                self.logs.append((force_time, f"No trade executed normally. Forcing BUY with signal {max_signal:.4f}"))
                self.execute_buy(force_time, force_row["Close"], {
                    "action": "buy",
                    "size": self.trade_size,
                    "stop_loss": force_row["Close"] * (1 - 0.02),
                    "take_profit": force_row["Close"] * (1 + 0.04),
                    "forced": True
                })
                if len(self.trades) > 0:
                    last_time = self.data.index[-1]
                    last_price = self.data.iloc[-1]["Close"]
                    self.execute_sell(last_time, last_price, {"action": "sell", "reason": "Forced trade at end"})
                    self.logs.append((last_time, "Forced trade executed due to insufficient signals."))
                else:
                    self.logs.append((self.data.index[-1], "Forced BUY failed (insufficient cash)."))
            else:
                self.logs.append((self.data.index[-1], "No forced trade executed: best signal below threshold."))

        return self.get_performance()

    def execute_buy(self, time, price, signal):
        exec_price = self.simulate_slippage(price)
        size = signal.get("size", self.trade_size)
        cost = exec_price * size * (1 + self.commission)
        if self.cash >= cost:
            self.position = size
            self.cash -= cost
            trade = Trade(entry_time=time, entry_price=exec_price, size=size,
                          stop_loss=signal.get("stop_loss"), take_profit=signal.get("take_profit"))
            trade.update_log(f"Bought at {exec_price}")
            self.trades.append(trade)
            self.logs.append((time, f"Executed BUY at {exec_price}"))
        else:
            self.logs.append((time, "Insufficient cash for BUY"))

    def execute_sell(self, time, price, signal):
        exec_price = self.simulate_slippage(price)
        proceeds = exec_price * self.position * (1 - self.commission)
        self.cash += proceeds
        trade = self.trades[-1]
        trade.close(time, exec_price)
        self.position = 0
        self.logs.append((time, f"Executed SELL at {exec_price}"))

    def get_performance(self):
        total_return = (self.cash - self.initial_cash) / self.initial_cash
        profits = [t.profit() for t in self.trades if t.is_closed()]
        num_trades = len(profits)
        avg_profit = np.mean(profits) if profits else 0
        max_profit = np.max(profits) if profits else 0
        max_loss = np.min(profits) if profits else 0
        return {
            "Final Cash": self.cash,
            "Total Return": total_return,
            "Number of Trades": num_trades,
            "Average Profit per Trade": avg_profit,
            "Max Profit": max_profit,
            "Max Loss": max_loss,
            "Trades": self.trades,
            "Logs": self.logs
        }
