# technical.py
import pandas as pd
import pandas_ta as ta
from utils import log

class TechnicalAnalyzer:
    """Compute technical indicators and detect patterns using pandas-ta."""

    def __init__(self, ma_fast: int = 20, ma_slow: int = 50, volume_period: int = 20):
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.volume_period = volume_period

    def calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving average columns to df using rolling mean."""
        df['MA_fast'] = df['close'].rolling(self.ma_fast).mean()
        df['MA_slow'] = df['close'].rolling(self.ma_slow).mean()
        return df

    def detect_trend(self, df: pd.DataFrame) -> str:
        """Return 'up', 'down' or 'flat' based on MAs."""
        last = df.iloc[-1]
        fast = last['MA_fast']
        slow = last['MA_slow']
        if fast > slow:
            return "up"
        elif fast < slow:
            return "down"
        else:
            return "flat"

    def detect_breakout(self, df: pd.DataFrame) -> str:
        """Detect breakout above resistance or below support."""
        support = df['low'].min()
        resistance = df['high'].max()
        last_close = df['close'].iloc[-1]
        if last_close > resistance:
            return "breakout_up"
        elif last_close < support:
            return "breakout_down"
        return None

    def detect_candlestick_patterns(self, df: pd.DataFrame) -> list:
        """Return list of detected candlestick patterns using pandas-ta."""
        patterns = []
        df.ta.cdl_pattern(update=True)
        for col in df.columns:
            if col.startswith('CDL_') and df[col].iloc[-1] != 0:
                patterns.append((col, int(df[col].iloc[-1])))
        return patterns

    def support_resistance(self, df: pd.DataFrame, lookback: int = 50) -> tuple:
        """Return recent support and resistance levels."""
        support = df['low'].rolling(lookback).min().iloc[-1]
        resistance = df['high'].rolling(lookback).max().iloc[-1]
        return support, resistance

    def fibonacci_levels(self, df: pd.DataFrame) -> dict:
        """Return key Fibonacci retracement levels."""
        swing_high = df['high'].max()
        swing_low = df['low'].min()
        diff = swing_high - swing_low
        levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        return {f"fib_{int(l*100)}": swing_low + diff * l for l in levels}

    def draw_trendlines(self, df: pd.DataFrame) -> dict:
        """Return basic trendline points for LTA and LTB."""
        top_idx = df['high'].idxmax()
        bottom_idx = df['low'].idxmin()
        return {
            "LTA": (bottom_idx, df.at[bottom_idx, 'low']),
            "LTB": (top_idx, df.at[top_idx, 'high'])
        }

    def validate_candle_pattern(self, candle: pd.Series, pattern_name: str) -> bool:
        """Simple validation: trust pandas-ta detection for now."""
        return True
