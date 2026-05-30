"""
Collector サービス

このファイルは Binance の WebSocket (取引・板情報) からデータを取得し、
TimescaleDB / PostgreSQL に保存するためのメインロジックを提供します。

主な用途:
- `Collector` クラスは WebSocket クライアントを初期化し、受信した Trade と
    OrderBookSnapshot を DB に挿入します。

使い方の例:
- 開発: `python -m collector.collector`
- Docker 環境では `docker-compose up collector` で起動します。

設定:
- `common/config.yaml` の `database` セクションで DB 接続情報を設定してください。
"""

import asyncio
import yaml
from pathlib import Path
from datetime import datetime
import logging

from ..storage import DBConnection
from ..common import Trade, OrderBookSnapshot, get_logger
from .ws_client import BinanceWSClient


class Collector:
    """Main collector service."""
    
    def __init__(self, config_path: str = "common/config.yaml"):
        """インスタンス初期化.

        - `config_path` から設定を読み込み、DB/WS クライアントを初期化します。
        """
        self.config = self._load_config(config_path)
        self.logger = get_logger("Collector")
        
        self.symbol = self.config.get("symbol", "BTCUSDT")
        self.exchange = self.config.get("exchange", "binanceusdm")
        
        # Initialize DB
        DBConnection.initialize(
            host=self.config["database"]["host"],
            port=self.config["database"]["port"],
            user=self.config["database"]["user"],
            password=self.config["database"]["password"],
            dbname=self.config["database"]["dbname"],
        )
        
        # Initialize WS client
        self.ws_client = BinanceWSClient(self.symbol)
        self.ws_client.on_trade = self._on_trade
        self.ws_client.on_orderbook = self._on_orderbook
        
        # Stats
        self.trades_count = 0
        self.orderbook_count = 0
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML."""
        path = Path(config_path)
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    async def _on_trade(self, trade: Trade):
        """Handle new trade."""
        try:
            query = """
            INSERT INTO trades (exchange, symbol, ts, price, size, side, buyer_maker, trade_id)
            VALUES (%s, %s, to_timestamp(%s / 1000.0), %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """
            params = (
                trade.exchange,
                trade.symbol,
                trade.ts,
                str(trade.price),
                str(trade.size),
                trade.side,
                trade.buyer_maker,
                trade.trade_id,
            )
            
            DBConnection.execute_update(query, params)
            self.trades_count += 1
            
            if self.trades_count % 1000 == 0:
                self.logger.info(f"Trades inserted: {self.trades_count}")
        except Exception as e:
            self.logger.error(f"Trade insert error: {e}")
    
    async def _on_orderbook(self, ob: OrderBookSnapshot):
        """Handle orderbook snapshot."""
        try:
            bids_json = str(ob.bids).replace("'", "\"")
            asks_json = str(ob.asks).replace("'", "\"")
            
            query = """
            INSERT INTO orderbook_snapshot (exchange, symbol, ts, best_bid, best_ask, bids, asks)
            VALUES (%s, %s, to_timestamp(%s / 1000.0), %s, %s, %s::jsonb, %s::jsonb)
            """
            params = (
                ob.exchange,
                ob.symbol,
                ob.ts,
                str(ob.best_bid),
                str(ob.best_ask),
                bids_json,
                asks_json,
            )
            
            DBConnection.execute_update(query, params)
            self.orderbook_count += 1
            
            if self.orderbook_count % 100 == 0:
                self.logger.info(f"Orderbook snapshots inserted: {self.orderbook_count}")
        except Exception as e:
            self.logger.error(f"Orderbook insert error: {e}")
    
    async def run(self):
        """Run collector."""
        self.logger.info(f"Starting collector for {self.symbol}...")
        try:
            await self.ws_client.run()
        except KeyboardInterrupt:
            self.logger.info("Collector stopped by user")
        except Exception as e:
            self.logger.error(f"Collector error: {e}")
        finally:
            await self.ws_client.close()
            DBConnection.close_pool()
            self.logger.info(f"Collector stopped. Trades: {self.trades_count}, OB: {self.orderbook_count}")


async def main():
    """Main entry point."""
    collector = Collector()
    await collector.run()


if __name__ == "__main__":
    asyncio.run(main())
