"""
エグゼキューションレイヤー (Binance USDT-M)

このモジュールは CCXT を用いて Binance USDT-M (Perpetual) に発注・取消を行う
`ExecutionEngine` を実装します。テストネット向けの設定にも対応しています。

必要なシークレット:
- `common/config.yaml` の `binance.api_key` と `binance.api_secret` を設定してください。
- テストネットを使う場合は `binance.testnet` フラグを有効にします。

注意事項:
- 実際の口座で実行する場合は十分に注意し、まずはテストネットで動作確認してください。
"""

import ccxt
import asyncio
import json
import redis
from typing import Optional, Dict, List
from datetime import datetime
from pathlib import Path
import yaml

from ..common import get_logger, OrderSide, OrderType, OrderStatus, ExecutionResult


class ExecutionEngine:
    """CCXT-based execution for Binance USDT-M Perpetual."""
    
    def __init__(self, config_path: str = "common/config.yaml"):
        """初期化: 設定を読み込み、CCXT の Exchange インスタンスと Redis を初期化します。

        - `config_path` から Binance の API キー等を読み込みます。
        """
        self.config = self._load_config(config_path)
        self.logger = get_logger("ExecutionEngine")
        
        self.symbol = self.config.get("symbol", "BTCUSDT")
        self.exchange_name = self.config.get("exchange", "binanceusdm")
        
        # Initialize CCXT exchange (testnet)
        exchange_config = {
            "enableRateLimit": True,
            "apiKey": self.config["binance"]["api_key"],
            "secret": self.config["binance"]["api_secret"],
            "urls": {
                "api": {
                    "fapi": "https://testnet.binancefuture.com/fapi",
                    "fapiData": "https://testnet.binancefuture.com/fapi",
                }
            } if self.config["binance"].get("testnet") else {},
        }
        
        self.exchange = ccxt.binanceusdm(exchange_config)
        
        # Redis for signal input
        self.redis_client = redis.Redis(
            host=self.config["redis"]["host"],
            port=self.config["redis"]["port"],
            db=self.config["redis"]["db"],
            decode_responses=True,
        )
        
        # Risk tracking
        self.daily_start_equity = None
        self.daily_pnl = 0.0
        self.open_orders: Dict[str, Dict] = {}  # client_id -> order info
        self.position_notional = 0.0
        self.position_side = 0  # -1: short, 0: flat, 1: long
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML."""
        path = Path(config_path)
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_position(self) -> Dict:
        """Get current position from exchange."""
        try:
            positions = self.exchange.fapiPrivateGetPositionsrisk({"symbol": self.symbol})
            
            for pos in positions:
                if pos["symbol"] == self.symbol:
                    return {
                        "quantity": float(pos.get("positionAmt", 0)),
                        "entry_price": float(pos.get("entryPrice", 0)),
                        "mark_price": float(pos.get("markPrice", 0)),
                        "leverage": float(pos.get("leverage", 1)),
                        "notional": float(pos.get("notional", 0)),
                        "pnl_unrealized": float(pos.get("unrealizedProfit", 0)),
                    }
            
            return {
                "quantity": 0.0,
                "entry_price": 0.0,
                "mark_price": 0.0,
                "leverage": 1.0,
                "notional": 0.0,
                "pnl_unrealized": 0.0,
            }
        except Exception as e:
            self.logger.error(f"Failed to get position: {e}")
            return {}
    
    def get_account_balance(self) -> float:
        """Get account equity."""
        try:
            account = self.exchange.fapiPrivateGetAccount()
            return float(account["totalWalletBalance"])
        except Exception as e:
            self.logger.error(f"Failed to get account balance: {e}")
            return 0.0
    
    def place_order(
        self,
        side: str,  # "BUY" or "SELL"
        order_type: str,  # "MARKET" or "LIMIT"
        quantity: float,
        price: Optional[float] = None,
        client_id: Optional[str] = None,
        reduce_only: bool = False,
        post_only: bool = False,
    ) -> ExecutionResult:
        """Place an order on the exchange."""
        
        try:
            params = {
                "symbol": self.symbol,
                "side": side,
                "type": order_type,
                "quantity": quantity,
            }
            
            if price is not None:
                params["price"] = price
            
            if client_id:
                params["clientOrderId"] = client_id
            
            if reduce_only:
                params["reduceOnly"] = True
            
            if post_only:
                params["timeInForce"] = "PO"
            
            order = self.exchange.create_order(
                symbol=self.symbol,
                type=order_type.lower(),
                side=side.lower(),
                amount=quantity,
                price=price,
                params=params,
            )
            
            self.logger.info(f"Order placed: {side} {quantity} @ {price} (ID: {order['id']})")
            
            # Track order
            if client_id:
                self.open_orders[client_id] = {
                    "order_id": order["id"],
                    "status": "PENDING",
                }
            
            return ExecutionResult(
                order_id=order["id"],
                status=OrderStatus.PENDING,
                filled_qty=float(order.get("filled", 0)),
                avg_price=float(order.get("average", 0)),
                commission=float(order.get("cost", 0)) * self.config["fees"]["taker"],
            )
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            return ExecutionResult(
                order_id=None,
                status=OrderStatus.REJECTED,
                filled_qty=0.0,
                avg_price=0.0,
                commission=0.0,
                error=str(e),
            )
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            self.exchange.cancel_order(order_id, self.symbol)
            self.logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order: {e}")
            return False
    
    def cancel_all_orders(self) -> bool:
        """Cancel all open orders."""
        try:
            orders = self.exchange.fetch_open_orders(self.symbol)
            for order in orders:
                self.cancel_order(order["id"])
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel all orders: {e}")
            return False
    
    def close_position(self) -> bool:
        """Close current position with market order."""
        try:
            position = self.get_position()
            if position["quantity"] == 0:
                self.logger.info("No position to close")
                return True
            
            side = "SELL" if position["quantity"] > 0 else "BUY"
            qty = abs(position["quantity"])
            
            result = self.place_order(
                side=side,
                order_type="MARKET",
                quantity=qty,
                reduce_only=True,
            )
            
            return result.status in (OrderStatus.FILLED, OrderStatus.PENDING)
        except Exception as e:
            self.logger.error(f"Failed to close position: {e}")
            return False
    
    def _check_risk_constraints(self, position_notional: float, equity: float) -> bool:
        """Check if proposed trade violates risk constraints."""
        
        # Check leverage
        leverage = position_notional / equity if equity > 0 else 0
        max_leverage = self.config["risk"]["max_leverage"]
        if leverage > max_leverage:
            self.logger.warning(f"Leverage constraint violated: {leverage:.2f}x > {max_leverage}x")
            return False
        
        # Check notional ratio
        notional_ratio = position_notional / equity if equity > 0 else 0
        max_ratio = self.config["risk"]["max_notional_by_equity"]
        if notional_ratio > max_ratio:
            self.logger.warning(f"Notional ratio constraint violated: {notional_ratio:.2%} > {max_ratio:.2%}")
            return False
        
        # Check daily DD
        if self.daily_start_equity is None:
            self.daily_start_equity = equity
        
        dd = 1 - (equity / self.daily_start_equity)
        daily_dd_stop = self.config["risk"]["daily_dd_stop"]
        if dd > daily_dd_stop:
            self.logger.warning(f"Daily drawdown exceeded: {dd:.2%} > {daily_dd_stop:.2%}")
            return False
        
        return True
    
    async def listen_for_signals(self):
        """Listen for trading signals from Redis."""
        
        signal_queue = f"signals:{self.symbol}"
        
        self.logger.info(f"Listening for signals on {signal_queue}...")
        
        while True:
            try:
                # Get signal from Redis queue
                signal_data = self.redis_client.lpop(signal_queue)
                
                if signal_data:
                    signal = json.loads(signal_data)
                    await self._execute_signal(signal)
                
                await asyncio.sleep(0.1)
            
            except Exception as e:
                self.logger.error(f"Signal processing error: {e}")
                await asyncio.sleep(1)
    
    async def _execute_signal(self, signal: Dict):
        """Execute trading signal."""
        
        action = signal.get("action")  # 0: hold, 1: long, 2: short
        size = signal.get("size", 0.0)
        
        equity = self.get_account_balance()
        position = self.get_position()
        
        # Check risk
        if not self._check_risk_constraints(abs(size) * signal.get("price", 1.0), equity):
            self.logger.warning("Risk constraint violated, skipping trade")
            return
        
        # Determine position change
        current_pos = position.get("quantity", 0)
        
        if action == 1:  # Long
            if current_pos < size:
                # Enter long or add to long
                qty_to_trade = size - current_pos
                self.place_order("BUY", "MARKET", qty_to_trade)
        
        elif action == 2:  # Short
            if current_pos > -size:
                # Enter short or add to short
                qty_to_trade = -size - current_pos
                self.place_order("SELL", "MARKET", abs(qty_to_trade))
        
        else:  # Hold (action == 0)
            # Close position if any
            if current_pos != 0:
                self.close_position()
    
    def run_sync(self):
        """Run execution synchronously (for non-async environments)."""
        
        self.logger.info("Starting execution engine...")
        
        try:
            while True:
                # Check for signals
                signal_queue = f"signals:{self.symbol}"
                signal_data = self.redis_client.lpop(signal_queue)
                
                if signal_data:
                    signal = json.loads(signal_data)
                    asyncio.run(self._execute_signal(signal))
                
                # Update position tracking
                position = self.get_position()
                self.position_notional = position.get("notional", 0.0)
                self.position_side = (1 if position.get("quantity", 0) > 0 else
                                      -1 if position.get("quantity", 0) < 0 else 0)
                
                asyncio.sleep(0.5)
        
        except KeyboardInterrupt:
            self.logger.info("Execution stopped by user")
            self.cancel_all_orders()
        
        except Exception as e:
            self.logger.error(f"Execution error: {e}")
            self.cancel_all_orders()
