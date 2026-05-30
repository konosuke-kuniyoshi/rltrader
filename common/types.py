"""
共通型定義 (RLTrader)

このファイルはデータモデルと型を定義します。主に以下を含みます:
- 取引や板スナップショットを表す `Trade`, `OrderBookSnapshot` 等の `dataclass`
- 環境・特徴量で利用する `Features` の構造
- 注文や実行結果を表す型と列挙型 (`OrderSide`, `OrderType`, `OrderStatus` など)

これらの型はシステム全体で共有されるため、外部サービスや DB とのインターフェースに
使われます。
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum
import numpy as np


class OrderSide(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class PositionFlag(Enum):
    """Position state."""
    FLAT = 0
    LONG = 1
    SHORT = -1


@dataclass
class Trade:
    """Trade data."""
    exchange: str
    symbol: str
    ts: int  # milliseconds
    price: float
    size: float
    side: str  # "BUY" or "SELL"
    buyer_maker: Optional[bool] = None
    trade_id: Optional[int] = None


@dataclass
class OrderBookSnapshot:
    """Order book snapshot."""
    exchange: str
    symbol: str
    ts: int  # milliseconds
    best_bid: float
    best_ask: float
    bids: List[tuple]  # [(price, quantity), ...]
    asks: List[tuple]  # [(price, quantity), ...]


@dataclass
class OHLCV:
    """OHLCV data."""
    exchange: str
    symbol: str
    ts: int  # milliseconds
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Features:
    """Market features (36-dim feature vector)."""
    # Price/Trend/Volatility
    ret_15s: float
    ret_1m: float
    ret_5m: float
    ret_15m: float
    realized_vol_5m: float
    realized_vol_15m: float
    atr14_1m: float
    atr14_5m: float
    sma_ratio_5_20: float
    sma_ratio_20_60: float
    macd: float
    macd_signal: float
    macd_hist: float
    rsi14: float
    
    # Book/Liquidity
    spread_in_ticks: float
    depth_bid5: float
    depth_ask5: float
    depth_bid10: float
    depth_ask10: float
    order_imbalance: float
    impact_price_10k: float
    impact_price_50k: float
    last10_trades_buy_ratio: float
    last10_trades_avg_size: float
    
    # Volume/Regime
    vol_1m_z: float
    vol_5m_z: float
    session_flag: int  # 0=Asia, 1=Europe, 2=US
    funding_recent: float
    funding_pred: float
    realized_skew_5m: float
    realized_kurt_5m: float
    
    # Position Info
    position_flag: int  # -1/0/1
    time_in_position_clipped: float
    
    ts: int  # timestamp (ms)
    
    def to_array(self) -> np.ndarray:
        """Convert to feature array (36-dim)."""
        return np.array([
            self.ret_15s, self.ret_1m, self.ret_5m, self.ret_15m,
            self.realized_vol_5m, self.realized_vol_15m,
            self.atr14_1m, self.atr14_5m,
            self.sma_ratio_5_20, self.sma_ratio_20_60,
            self.macd, self.macd_signal, self.macd_hist, self.rsi14,
            self.spread_in_ticks,
            self.depth_bid5, self.depth_ask5, self.depth_bid10, self.depth_ask10,
            self.order_imbalance,
            self.impact_price_10k, self.impact_price_50k,
            self.last10_trades_buy_ratio, self.last10_trades_avg_size,
            self.vol_1m_z, self.vol_5m_z,
            float(self.session_flag),
            self.funding_recent, self.funding_pred,
            self.realized_skew_5m, self.realized_kurt_5m,
            float(self.position_flag),
            self.time_in_position_clipped,
        ], dtype=np.float32)


@dataclass
class Order:
    """Order data."""
    exchange: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    client_id: Optional[str] = None
    reduce_only: bool = False
    post_only: bool = False


@dataclass
class ExecutionResult:
    """Execution result."""
    order_id: Optional[str]
    status: OrderStatus
    filled_qty: float
    avg_price: float
    commission: float
    error: Optional[str] = None
