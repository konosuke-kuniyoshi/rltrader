"""
マーケットデータから特徴量を抽出するユーティリティ

`FeatureExtractor` は OHLCV、最近の取引、板情報を受け取り、学習・推論用の
36次元特徴量 (`Features` dataclass) を生成します。

出力:
- 正規化済みのテクニカル指標、板深度指標、出来高指標等を含む `Features` インスタンス。

利用例:
- `extract_features(...)` を呼んで `Features` を取得し、`to_array()` で配列化します。
"""

import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd

from ..common import Trade, OrderBookSnapshot, OHLCV, Features


class FeatureExtractor:
    """Extract 36-dim feature vector from market data."""
    
    def __init__(self, lookback_min: int = 30):
        """初期化: 特徴量抽出の設定とバッファを準備します。

        - `lookback_min`: 参照する過去データの長さ（分）
        """
        self.lookback_min = lookback_min
        
        # Buffers for calculations
        self.trades_buffer: List[Trade] = []
        self.ohlcv_buffer: List[OHLCV] = []
        self.orderbook_buffer: List[OrderBookSnapshot] = []
        self.returns_buffer: List[float] = []
    
    def _calculate_returns(self, prices: List[float], periods: List[int]) -> dict:
        """Calculate log returns at different periods."""
        result = {}
        if len(prices) < 2:
            return {f"ret_{p}": 0.0 for p in periods}
        
        current_price = prices[-1]
        for p in periods:
            if len(prices) > p:
                past_price = prices[-(p + 1)]
                result[f"ret_{p}"] = np.log(current_price / past_price) if past_price > 0 else 0.0
            else:
                result[f"ret_{p}"] = 0.0
        return result
    
    def _calculate_volatility(self, returns: List[float], period: int) -> float:
        """Calculate realized volatility."""
        if len(returns) < period:
            return 0.0
        recent = returns[-period:]
        return float(np.std(recent)) if len(recent) > 0 else 0.0
    
    def _calculate_atr(self, ohlcv: List[OHLCV], period: int) -> float:
        """Calculate Average True Range."""
        if len(ohlcv) < period:
            return 0.0
        
        trs = []
        for i in range(len(ohlcv) - period, len(ohlcv)):
            if i == 0:
                tr = ohlcv[i].high - ohlcv[i].low
            else:
                tr = max(
                    ohlcv[i].high - ohlcv[i].low,
                    abs(ohlcv[i].high - ohlcv[i-1].close),
                    abs(ohlcv[i].low - ohlcv[i-1].close)
                )
            trs.append(tr)
        
        return float(np.mean(trs)) if len(trs) > 0 else 0.0
    
    def _calculate_sma(self, prices: List[float], period: int) -> float:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        return float(np.mean(prices[-period:]))
    
    def _calculate_macd(self, prices: List[float]) -> Tuple[float, float, float]:
        """Calculate MACD (12, 26, 9)."""
        if len(prices) < 26:
            return 0.0, 0.0, 0.0
        
        ema12 = self._ema(prices, 12)
        ema26 = self._ema(prices, 26)
        macd_line = ema12 - ema26
        
        # Signal line (9-period EMA of MACD)
        macd_values = []
        for i in range(26, len(prices)):
            e12 = self._ema(prices[:i+1], 12)
            e26 = self._ema(prices[:i+1], 26)
            macd_values.append(e12 - e26)
        
        signal = self._ema(macd_values, 9) if len(macd_values) >= 9 else macd_line
        histogram = macd_line - signal
        
        return float(macd_line), float(signal), float(histogram)
    
    def _ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return float(np.mean(prices))
        
        ema = np.mean(prices[:period])
        multiplier = 2.0 / (period + 1)
        
        for i in range(period, len(prices)):
            ema = prices[i] * multiplier + ema * (1 - multiplier)
        
        return float(ema)
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI (0..1)."""
        if len(prices) < period + 1:
            return 0.5
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 1.0 if avg_gain > 0 else 0.5
        
        rs = avg_gain / avg_loss
        rsi = 1.0 - (1.0 / (1.0 + rs))
        return float(np.clip(rsi, 0, 1))
    
    def _calculate_depth_metrics(self, ob: OrderBookSnapshot, recent_volume: float) -> dict:
        """Calculate order book depth metrics."""
        if not ob or not ob.bids or not ob.asks:
            return {
                "spread_in_ticks": 0.0,
                "depth_bid5": 0.0, "depth_ask5": 0.0,
                "depth_bid10": 0.0, "depth_ask10": 0.0,
                "order_imbalance": 0.0,
                "impact_price_10k": 0.0, "impact_price_50k": 0.0,
            }
        
        best_bid = ob.best_bid
        best_ask = ob.best_ask
        spread = best_ask - best_bid
        tick_size = 0.01  # BTC tick size
        spread_in_ticks = spread / tick_size if tick_size > 0 else 0
        
        # Cumulative depth
        bid5_vol = sum(q for _, q in ob.bids[:5])
        ask5_vol = sum(q for _, q in ob.asks[:5])
        bid10_vol = sum(q for _, q in ob.bids[:10])
        ask10_vol = sum(q for _, q in ob.asks[:10])
        
        # Normalize by recent volume
        if recent_volume > 0:
            depth_bid5 = bid5_vol / recent_volume
            depth_ask5 = ask5_vol / recent_volume
            depth_bid10 = bid10_vol / recent_volume
            depth_ask10 = ask10_vol / recent_volume
        else:
            depth_bid5 = depth_ask5 = depth_bid10 = depth_ask10 = 0.0
        
        # Order imbalance
        total = bid10_vol + ask10_vol + 1e-9
        order_imbalance = (bid10_vol - ask10_vol) / total
        
        # Impact price (simplified: cumulative depth to reach notional)
        impact_10k = self._impact_price(ob.asks, 10000, best_ask)
        impact_50k = self._impact_price(ob.asks, 50000, best_ask)
        
        return {
            "spread_in_ticks": float(spread_in_ticks),
            "depth_bid5": float(depth_bid5),
            "depth_ask5": float(depth_ask5),
            "depth_bid10": float(depth_bid10),
            "depth_ask10": float(depth_ask10),
            "order_imbalance": float(order_imbalance),
            "impact_price_10k": float(impact_10k),
            "impact_price_50k": float(impact_50k),
        }
    
    def _impact_price(self, asks: List[Tuple[float, float]], notional: float, mid: float) -> float:
        """Calculate impact price for given notional."""
        cumul = 0.0
        for price, qty in asks:
            qty_usd = price * qty
            if cumul + qty_usd >= notional:
                return float((price - mid) / mid) if mid > 0 else 0.0
            cumul += qty_usd
        return float((asks[-1][0] - mid) / mid) if asks and mid > 0 else 0.0
    
    def _calculate_trade_metrics(self, trades: List[Trade]) -> dict:
        """Calculate metrics from recent trades."""
        if not trades:
            return {
                "last10_trades_buy_ratio": 0.5,
                "last10_trades_avg_size": 0.0,
            }
        
        recent_trades = trades[-10:]
        buy_count = sum(1 for t in recent_trades if t.side == "BUY")
        buy_ratio = buy_count / len(recent_trades) if recent_trades else 0.5
        avg_size = np.mean([t.size for t in recent_trades])
        
        return {
            "last10_trades_buy_ratio": float(buy_ratio),
            "last10_trades_avg_size": float(avg_size),
        }
    
    def _get_session_flag(self, ts: int) -> int:
        """Get trading session flag (0=Asia, 1=Europe, 2=US)."""
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        hour = dt.hour
        
        if 0 <= hour < 8:
            return 0  # Asia
        elif 8 <= hour < 16:
            return 1  # Europe
        else:
            return 2  # US
    
    def extract_features(
        self,
        ohlcv: List[OHLCV],
        recent_trades: List[Trade],
        orderbook: OrderBookSnapshot,
        position_flag: int = 0,
        time_in_position_s: float = 0.0,
        funding_rate: float = 0.0,
    ) -> Features:
        """Extract 36-dim feature vector."""
        
        if not ohlcv:
            # Return safe defaults
            return self._default_features(position_flag, time_in_position_s)
        
        prices = [o.close for o in ohlcv]
        atr_scale = max(self._calculate_atr(ohlcv, 14), 1e-6)
        
        # Price metrics
        returns = [np.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]
        ret_dict = self._calculate_returns(prices, [15, 60, 300, 900])  # 15s, 1m, 5m, 15m in seconds
        
        # Replace time-based with volume-based
        ret_dict = {
            "ret_15s": returns[-1] if len(returns) >= 1 else 0.0,
            "ret_1m": returns[-4] if len(returns) >= 4 else 0.0,
            "ret_5m": returns[-20] if len(returns) >= 20 else 0.0,
            "ret_15m": returns[-60] if len(returns) >= 60 else 0.0,
        }
        
        vol_5m = self._calculate_volatility(returns, 5)
        vol_15m = self._calculate_volatility(returns, 15)
        
        # Technical indicators
        atr14_1m = self._calculate_atr(ohlcv[-60:], 14) if len(ohlcv) >= 60 else 0.0
        atr14_5m = self._calculate_atr(ohlcv[-300:], 14) if len(ohlcv) >= 300 else 0.0
        
        sma5 = self._calculate_sma(prices, 5)
        sma20 = self._calculate_sma(prices, 20)
        sma60 = self._calculate_sma(prices, 60)
        
        sma_ratio_5_20 = ((sma5 - sma20) / atr_scale) if atr_scale > 0 else 0.0
        sma_ratio_20_60 = ((sma20 - sma60) / atr_scale) if atr_scale > 0 else 0.0
        
        macd, macd_signal, macd_hist = self._calculate_macd(prices)
        macd_norm = (macd / atr_scale) if atr_scale > 0 else 0.0
        macd_signal_norm = (macd_signal / atr_scale) if atr_scale > 0 else 0.0
        macd_hist_norm = (macd_hist / atr_scale) if atr_scale > 0 else 0.0
        
        rsi = self._calculate_rsi(prices, 14)
        
        # Order book metrics
        recent_volume = sum(t.size for t in recent_trades[-60:]) if recent_trades else 1.0
        ob_metrics = self._calculate_depth_metrics(orderbook, max(recent_volume, 1.0))
        
        # Trade metrics
        trade_metrics = self._calculate_trade_metrics(recent_trades)
        
        # Volume metrics (z-score)
        vol_1m = sum(t.size for t in recent_trades[-60:]) if recent_trades else 0.0
        vol_5m = sum(t.size for t in recent_trades[-300:]) if recent_trades else 0.0
        
        # Simplified z-score (assume mean=1, std=0.5 for now)
        vol_1m_z = (vol_1m - 1.0) / 0.5 if vol_1m > 0 else 0.0
        vol_5m_z = (vol_5m - 1.0) / 0.5 if vol_5m > 0 else 0.0
        
        # Session
        session = self._get_session_flag(orderbook.ts if orderbook else int(datetime.now().timestamp() * 1000))
        
        # Skewness and kurtosis (simplified)
        realized_skew = float(np.skew(returns[-300:])) if len(returns) >= 300 else 0.0
        realized_kurt = float(np.kurtosis(returns[-300:])) if len(returns) >= 300 else 0.0
        
        # Time in position
        time_clipped = min(time_in_position_s / 3600.0, 1.0)  # Clip at 1 hour
        
        features = Features(
            ret_15s=ret_dict["ret_15s"],
            ret_1m=ret_dict["ret_1m"],
            ret_5m=ret_dict["ret_5m"],
            ret_15m=ret_dict["ret_15m"],
            realized_vol_5m=vol_5m,
            realized_vol_15m=vol_15m,
            atr14_1m=atr14_1m,
            atr14_5m=atr14_5m,
            sma_ratio_5_20=sma_ratio_5_20,
            sma_ratio_20_60=sma_ratio_20_60,
            macd=macd_norm,
            macd_signal=macd_signal_norm,
            macd_hist=macd_hist_norm,
            rsi14=rsi,
            spread_in_ticks=ob_metrics["spread_in_ticks"],
            depth_bid5=ob_metrics["depth_bid5"],
            depth_ask5=ob_metrics["depth_ask5"],
            depth_bid10=ob_metrics["depth_bid10"],
            depth_ask10=ob_metrics["depth_ask10"],
            order_imbalance=ob_metrics["order_imbalance"],
            impact_price_10k=ob_metrics["impact_price_10k"],
            impact_price_50k=ob_metrics["impact_price_50k"],
            last10_trades_buy_ratio=trade_metrics["last10_trades_buy_ratio"],
            last10_trades_avg_size=trade_metrics["last10_trades_avg_size"],
            vol_1m_z=vol_1m_z,
            vol_5m_z=vol_5m_z,
            session_flag=session,
            funding_recent=funding_rate,
            funding_pred=funding_rate,  # Simplified: use recent as prediction
            realized_skew_5m=realized_skew,
            realized_kurt_5m=realized_kurt,
            position_flag=position_flag,
            time_in_position_clipped=time_clipped,
            ts=orderbook.ts if orderbook else int(datetime.now().timestamp() * 1000),
        )
        
        return features
    
    def _default_features(self, position_flag: int = 0, time_in_position_s: float = 0.0) -> Features:
        """Return safe default features."""
        return Features(
            ret_15s=0.0, ret_1m=0.0, ret_5m=0.0, ret_15m=0.0,
            realized_vol_5m=0.0, realized_vol_15m=0.0,
            atr14_1m=0.0, atr14_5m=0.0,
            sma_ratio_5_20=0.0, sma_ratio_20_60=0.0,
            macd=0.0, macd_signal=0.0, macd_hist=0.0,
            rsi14=0.5,
            spread_in_ticks=0.0,
            depth_bid5=0.0, depth_ask5=0.0, depth_bid10=0.0, depth_ask10=0.0,
            order_imbalance=0.0,
            impact_price_10k=0.0, impact_price_50k=0.0,
            last10_trades_buy_ratio=0.5, last10_trades_avg_size=0.0,
            vol_1m_z=0.0, vol_5m_z=0.0,
            session_flag=0,
            funding_recent=0.0, funding_pred=0.0,
            realized_skew_5m=0.0, realized_kurt_5m=0.0,
            position_flag=position_flag,
            time_in_position_clipped=min(time_in_position_s / 3600.0, 1.0),
            ts=int(datetime.now().timestamp() * 1000),
        )
