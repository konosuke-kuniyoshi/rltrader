"""
暗号資産トレーディング用 Gym 環境

このモジュールは学習・評価用の OpenAI Gym 互換環境 `CryptoTradingEnv` を提供します。

概要:
- 観測: 36 次元の特徴ベクトル (`Features`) を返します。
- 行動: `Discrete(3)` -> 0=hold, 1=long, 2=short
- 内部データ取得: Redis のキー `state:{symbol}` から最新の状態を読み取ります。

使用方法:
- トレーニング中は `env.reset()` / `env.step(action)` を利用してください。
- 環境設定は `common/config.yaml` の `redis` / `risk` / `fees` セクションで調整できます。
"""

import gym
from gym import spaces
import numpy as np
import json
import redis
from typing import Optional, Tuple, Dict
from datetime import datetime, timedelta

from ..common import get_logger


class CryptoTradingEnv(gym.Env):
    """
    Gym environment for crypto trading with risk constraints.
    
    Observation: 36-dim feature vector from market state
    Action: Discrete(3) = {0: hold, 1: long, 2: short}
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        initial_capital: float = 10000.0,
        max_steps: int = 10000,
        config: Optional[Dict] = None,
    ):
        """初期化: 環境パラメータと外部接続 (Redis) を設定します。

        - `config` でリスク/手数料/Redis 接続設定を渡せます。
        """
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.max_steps = max_steps
        
        # Configuration
        self.config = config or {}
        self.taker_fee = self.config.get("fees", {}).get("taker", 0.0004)
        self.maker_fee = self.config.get("fees", {}).get("maker", 0.0002)
        self.slippage_rate = self.config.get("slippage", {}).get("rate", 0.0003)
        self.funding_per_8h = self.config.get("funding", {}).get("per_8h", 0.0001)
        
        # Risk constraints
        self.per_trade_loss_cap = self.config.get("risk", {}).get("per_trade_loss_cap", 0.02)
        self.daily_dd_stop = self.config.get("risk", {}).get("daily_dd_stop", 0.03)
        self.max_leverage = self.config.get("risk", {}).get("max_leverage", 3.0)
        self.max_notional_by_equity = self.config.get("risk", {}).get("max_notional_by_equity", 1.0)
        
        self.logger = get_logger(f"CryptoTradingEnv-{symbol}")
        
        # Redis for market data
        self.redis_client = redis.Redis(
            host=self.config.get("redis", {}).get("host", "localhost"),
            port=self.config.get("redis", {}).get("port", 6379),
            db=self.config.get("redis", {}).get("db", 0),
            decode_responses=True,
        )
        
        # Gym spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # hold, long, short
        
        # State
        self.equity = initial_capital
        self.position_size = 0.0  # quantity
        self.position_entry_price = 0.0
        self.position_entry_time = 0
        self.current_price = 0.0
        self.step_count = 0
        self.episode_start_equity = initial_capital
        self.day_start_equity = initial_capital
        self.pnl_unrealized = 0.0
        self.pnl_realized = 0.0
        self.trades_count = 0
        
        # Trading history for logging
        self.trades_log = []
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        self.equity = self.initial_capital
        self.position_size = 0.0
        self.position_entry_price = 0.0
        self.position_entry_time = 0
        self.step_count = 0
        self.episode_start_equity = self.initial_capital
        self.day_start_equity = self.initial_capital
        self.pnl_unrealized = 0.0
        self.pnl_realized = 0.0
        self.trades_count = 0
        self.trades_log = []
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get current market observation from Redis."""
        try:
            redis_key = f"state:{self.symbol}"
            data = self.redis_client.get(redis_key)
            
            if data:
                state_data = json.loads(data)
                features = np.array(state_data.get("features", np.zeros(36)), dtype=np.float32)
                self.current_price = state_data.get("price", self.current_price)
                return features
            else:
                self.logger.warning("No market data in Redis, using zeros")
                return np.zeros(36, dtype=np.float32)
        except Exception as e:
            self.logger.error(f"Failed to get observation: {e}")
            return np.zeros(36, dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step of the environment."""
        self.step_count += 1
        done = False
        reward = 0.0
        info = {"step": self.step_count}
        
        # Get current market state
        obs = self._get_observation()
        prev_price = self.current_price
        
        # Action: 0=hold, 1=long, 2=short
        position_change = False
        
        if action == 1:  # Go long
            if self.position_size <= 0:
                position_change = True
                reward -= self._execute_trade(action, obs)
        elif action == 2:  # Go short
            if self.position_size >= 0:
                position_change = True
                reward -= self._execute_trade(action, obs)
        # action == 0: hold, no change
        
        # Update unrealized PnL
        if self.position_size != 0:
            self._update_unrealized_pnl()
        
        # Apply funding cost
            funding_cost = self._apply_funding_cost()
            reward -= funding_cost
        
        # Check risk constraints
        violates_risk = self._check_risk_constraints()
        if violates_risk:
            self.logger.warning(f"Risk constraint violated at step {self.step_count}")
            done = True
            reward -= 0.1
        
        # Check max steps
        if self.step_count >= self.max_steps:
            done = True
        
        # Penalty for excess trading
        if position_change:
            reward -= 0.001  # Small trading penalty
        
        info.update({
            "equity": self.equity,
            "position": self.position_size,
            "pnl_realized": self.pnl_realized,
            "pnl_unrealized": self.pnl_unrealized,
            "trades": self.trades_count,
        })
        
        return obs, reward, done, info
    
    def _execute_trade(self, action: int, obs: np.ndarray) -> float:
        """Execute trade and return cost (fee + slippage)."""
        # Determine position size via risk rule
        atr = obs[6]  # atr14_1m
        sl_distance = max(atr, 0.0001)
        
        # Size calculation: min(equity*loss_cap/sl_distance, equity*notional, max_leverage*equity)
        size_by_loss_cap = (self.equity * self.per_trade_loss_cap) / sl_distance
        size_by_notional = self.equity * self.max_notional_by_equity / self.current_price
        size_by_leverage = self.max_leverage * self.equity / self.current_price
        
        new_size = min(size_by_loss_cap, size_by_notional, size_by_leverage)
        
        # Close old position
        close_cost = 0.0
        if self.position_size != 0:
            close_cost = self._close_position()
        
        # Open new position
        if action == 1:  # Long
            self.position_size = new_size
            self.position_entry_price = self.current_price * (1 + self.slippage_rate)
        else:  # Short
            self.position_size = -new_size
            self.position_entry_price = self.current_price * (1 - self.slippage_rate)
        
        self.position_entry_time = self.step_count
        
        # Calculate costs
        notional = abs(self.position_size) * self.current_price
        fee_cost = notional * self.taker_fee
        slippage_cost = notional * self.slippage_rate
        
        total_cost = close_cost + fee_cost + slippage_cost
        self.equity -= total_cost
        self.trades_count += 1
        
        self.trades_log.append({
            "step": self.step_count,
            "action": action,
            "price": self.current_price,
            "size": self.position_size,
            "cost": total_cost,
        })
        
        return total_cost
    
    def _close_position(self) -> float:
        """Close current position and return cost."""
        if self.position_size == 0:
            return 0.0
        
        notional = abs(self.position_size) * self.current_price
        pnl = self._calculate_pnl_on_close()
        
        fee_cost = notional * self.taker_fee
        slippage_cost = notional * self.slippage_rate
        
        total_cost = fee_cost + slippage_cost
        self.equity += pnl - total_cost
        self.pnl_realized += pnl - total_cost
        
        self.position_size = 0.0
        self.position_entry_price = 0.0
        
        return total_cost
    
    def _calculate_pnl_on_close(self) -> float:
        """Calculate PnL if we close now."""
        if self.position_size == 0:
            return 0.0
        
        if self.position_size > 0:
            # Long: profit if price rises
            exit_price = self.current_price * (1 - self.slippage_rate)
            pnl = self.position_size * (exit_price - self.position_entry_price)
        else:
            # Short: profit if price falls
            exit_price = self.current_price * (1 + self.slippage_rate)
            pnl = self.position_size * (exit_price - self.position_entry_price)
        
        return pnl
    
    def _update_unrealized_pnl(self):
        """Update unrealized PnL."""
        if self.position_size == 0:
            self.pnl_unrealized = 0.0
        elif self.position_size > 0:
            self.pnl_unrealized = self.position_size * (self.current_price - self.position_entry_price)
        else:
            self.pnl_unrealized = self.position_size * (self.current_price - self.position_entry_price)
    
    def _apply_funding_cost(self) -> float:
        """Apply funding rate cost."""
        if self.position_size == 0:
            return 0.0
        
        notional = abs(self.position_size) * self.current_price
        time_held_8h = (self.step_count - self.position_entry_time) / (8 * 3600)  # Rough estimate
        funding_cost = notional * self.funding_per_8h * time_held_8h * (1 if self.position_size > 0 else -1)
        
        self.equity -= abs(funding_cost)
        return abs(funding_cost)
    
    def _check_risk_constraints(self) -> bool:
        """Check if any risk constraint is violated. Returns True if violated."""
        # Check daily drawdown
        dd = 1 - (self.equity / self.day_start_equity)
        if dd > self.daily_dd_stop:
            self.logger.warning(f"Daily DD exceeded: {dd:.2%}")
            return True
        
        # Check leverage
        if self.position_size != 0:
            notional = abs(self.position_size) * self.current_price
            leverage = notional / self.equity
            if leverage > self.max_leverage:
                self.logger.warning(f"Leverage exceeded: {leverage:.2f}x")
                return True
        
        return False
    
    def render(self, mode: str = 'human'):
        """Render environment state."""
        print(f"Step: {self.step_count}, Equity: ${self.equity:.2f}, "
              f"Position: {self.position_size:.4f}, Price: ${self.current_price:.2f}, "
              f"PnL: ${self.pnl_realized + self.pnl_unrealized:.2f}")
