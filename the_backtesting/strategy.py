# backtesting/strategy.py

class Strategy:
    def on_bar(self, time, row, data):
        raise NotImplementedError("Implement the on_bar method in your strategy subclass.")
