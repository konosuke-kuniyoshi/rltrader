"""
Binance WebSocket クライアント (USDT-M Futures)

このモジュールは Binance の WebSocket ストリーム (aggTrade, depth) を購読し、
受信したイベントを内部でパースしてコールバック (`on_trade`, `on_orderbook`) を通じて
外部に提供します。

使い方:
- `BinanceWSClient(symbol)` を生成し、`on_trade` / `on_orderbook` にコールバックを設定します。
- `await client.run()` でトレードと板の購読を同時実行します。

注意:
- 公開データのため API キーは不要ですが、REST スナップショット取得ではレート制限に注意してください。
"""

import json
import asyncio
import aiohttp
from typing import Optional, Callable, Dict, List, Tuple
from datetime import datetime
import logging

from ..common import Trade, OrderBookSnapshot, get_logger


class BinanceWSClient:
    """Binance USDT-M Futures WebSocket client."""
    
    BASE_WS_URL = "wss://fstream.binance.com/ws"
    BASE_REST_URL = "https://fapi.binance.com"
    
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.symbol_lower = symbol.lower()
        self.logger = get_logger(f"BinanceWSClient-{symbol}")
        
        self.ws_connection = None
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Order book state
        self.last_update_id = 0
        self.bid_ask_buffer = {}  # pending updates before sync
        self.is_synchronized = False
        
        # Callbacks
        self.on_trade: Optional[Callable] = None
        self.on_orderbook: Optional[Callable] = None
    
    async def initialize(self):
        """
        HTTP/WebSocket セッションを初期化します。

        内部で aiohttp の `ClientSession` を生成し、REST 呼び出しや WS 接続に利用します。
        非同期処理の前に一度だけ呼び出してください（内部は冪等です）。
        """
        self.session = aiohttp.ClientSession()
    
    async def close(self):
        """
        セッションおよび WebSocket 接続をクローズします。

        プログラム終了時や再接続前などに呼び出して、コネクション資源を解放します。
        """
        if self.ws_connection:
            await self.ws_connection.close()
        if self.session:
            await self.session.close()
    
    async def subscribe_trades(self):
        """
        aggTrade ストリームを購読して取引イベントを受信します。

        - ストリームから受け取ったメッセージは `_handle_trade` に渡されます。
        - ネットワークエラーや切断時は再接続を試みます（簡易的なバックオフあり）。
        """
        if not self.session:
            await self.initialize()
        
        stream_name = f"{self.symbol_lower}@aggTrade"
        ws_url = f"{self.BASE_WS_URL}/{stream_name}"
        
        self.logger.info(f"Subscribing to trades: {stream_name}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url, heartbeat=30) as ws:
                    self.ws_connection = ws
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await self._handle_trade(json.loads(msg.data))
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            self.logger.warning("Trade stream closed, reconnecting...")
                            await asyncio.sleep(1)
                            break
        except Exception as e:
            self.logger.error(f"Trade stream error: {e}")
            await asyncio.sleep(5)
    
    async def subscribe_depth(self, depth: int = 20, update_speed_ms: int = 100):
        """
        板差分ストリーム (`depth@{ms}ms`) を購読します。

        動作:
        1. まず REST API で初期スナップショットを取得して同期（`_fetch_depth_snapshot`）。
        2. 差分イベントを受信し `_handle_depth_update` で反映します。

        引数:
        - `depth`: REST で取得する板の深さ（デフォルト 20）
        - `update_speed_ms`: 更新間隔（ms）
        """
        if not self.session:
            await self.initialize()
        
        # Initial snapshot
        await self._fetch_depth_snapshot(depth)
        
        # Subscribe to diff stream
        stream_name = f"{self.symbol_lower}@depth@{update_speed_ms}ms"
        ws_url = f"{self.BASE_WS_URL}/{stream_name}"
        
        self.logger.info(f"Subscribing to depth: {stream_name}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url, heartbeat=30) as ws:
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await self._handle_depth_update(json.loads(msg.data))
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            self.logger.warning("Depth stream closed, reconnecting...")
                            await asyncio.sleep(1)
                            break
        except Exception as e:
            self.logger.error(f"Depth stream error: {e}")
            await asyncio.sleep(5)
    
    async def _fetch_depth_snapshot(self, depth: int = 20):
        """
        REST API を呼んで板の初期スナップショットを取得し、内部状態を同期します。

        - 成功すると `self.last_update_id` と `self.bid_ask_buffer` を更新します。
        - タイムアウトや HTTP エラーはログに出力されます。
        """
        url = f"{self.BASE_REST_URL}/fapi/v1/depth"
        params = {"symbol": self.symbol, "limit": depth}
        
        try:
            async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.last_update_id = data["lastUpdateId"]
                    self.is_synchronized = True
                    
                    bids = [(float(p), float(q)) for p, q in data["bids"]]
                    asks = [(float(p), float(q)) for p, q in data["asks"]]
                    
                    self.bid_ask_buffer = {"bids": bids, "asks": asks}
                    self.logger.info(f"Depth snapshot synced: lastUpdateId={self.last_update_id}")
                else:
                    self.logger.error(f"Depth snapshot failed: {resp.status}")
        except Exception as e:
            self.logger.error(f"Depth snapshot error: {e}")
    
    async def _handle_trade(self, msg: Dict):
        """
        aggTrade メッセージをパースして `Trade` オブジェクトを生成し、
        設定されている `self.on_trade` コールバックに渡します。

        - `msg` は Binance の aggTrade ペイロードを想定しています。
        - 例外発生時はログ出力のみ行い処理を継続します。
        """
        try:
            trade = Trade(
                exchange="binanceusdm",
                symbol=self.symbol,
                ts=int(msg["T"]),  # milliseconds
                price=float(msg["p"]),
                size=float(msg["q"]),
                side="BUY" if msg["m"] == False else "SELL",
                buyer_maker=msg.get("m", False),
                trade_id=int(msg["a"]),
            )
            
            if self.on_trade:
                await self.on_trade(trade)
        except Exception as e:
            self.logger.error(f"Trade processing error: {e}, msg={msg}")
    
    async def _handle_depth_update(self, msg: Dict):
        """
        depth 差分イベントを適用して内部の板状態を更新し、
        一定の条件を満たしたら `OrderBookSnapshot` を生成して `self.on_orderbook` に渡します。

        処理内容:
        - 更新 ID の連続性を確認し、ギャップがあれば再同期を試みます。
        - 価格レベルの追加/削除を反映して簡易的な top-N の板を構築します。
        - 例外はログ出力して無視します（耐障害設計の一部）。
        """
        try:
            # Check update_id continuity
            U = msg["U"]  # first update ID in event
            u = msg["u"]  # final update ID in event
            
            if not self.is_synchronized:
                # Buffer updates until synchronized
                self.bid_ask_buffer["pending"] = msg
                if u >= self.last_update_id:
                    # Can now process
                    await self._fetch_depth_snapshot()
                return
            
            # Check for gap (shouldn't happen on fast channel, but just in case)
            if U <= self.last_update_id < u:
                self.logger.warning(f"Gap detected: last={self.last_update_id}, U={U}, u={u}. Resynchronizing...")
                self.is_synchronized = False
                await self._fetch_depth_snapshot()
                return
            
            # Apply updates
            if U > self.last_update_id + 1:
                self.logger.warning(f"Possible gap: last={self.last_update_id}, U={U}")
            
            # Update order book
            for price_str, qty_str in msg["b"]:  # bids
                price = float(price_str)
                qty = float(qty_str)
                if qty == 0:
                    self.bid_ask_buffer["bids"] = [
                        (p, q) for p, q in self.bid_ask_buffer.get("bids", []) if p != price
                    ]
                else:
                    self.bid_ask_buffer["bids"].append((price, qty))
            
            for price_str, qty_str in msg["a"]:  # asks
                price = float(price_str)
                qty = float(qty_str)
                if qty == 0:
                    self.bid_ask_buffer["asks"] = [
                        (p, q) for p, q in self.bid_ask_buffer.get("asks", []) if p != price
                    ]
                else:
                    self.bid_ask_buffer["asks"].append((price, qty))
            
            self.last_update_id = u
            
            # Emit snapshot
            bids = sorted(self.bid_ask_buffer.get("bids", []), key=lambda x: x[0], reverse=True)[:10]
            asks = sorted(self.bid_ask_buffer.get("asks", []), key=lambda x: x[0])[:10]
            
            if bids and asks:
                ob = OrderBookSnapshot(
                    exchange="binanceusdm",
                    symbol=self.symbol,
                    ts=int(msg["E"]),
                    best_bid=bids[0][0],
                    best_ask=asks[0][0],
                    bids=bids,
                    asks=asks,
                )
                
                if self.on_orderbook:
                    await self.on_orderbook(ob)
        except Exception as e:
            self.logger.error(f"Depth update error: {e}, msg={msg}")
    
    async def run(self):
        """トレードストリームと板差分ストリームを並行して実行します。

        - 内部で `initialize()` を呼び、`subscribe_trades` と `subscribe_depth` を
          `asyncio.gather` で並列実行します。
        - 例外が発生してもクリーンアップ (`close`) を行います。
        """
        if not self.session:
            await self.initialize()

        try:
            await asyncio.gather(
                self.subscribe_trades(),
                self.subscribe_depth(),
                return_exceptions=True
            )
        finally:
            await self.close()
