"""
Streamer サービス (特徴量計算と Redis への公開)

このモジュールは DB から市場データを取得して特徴量を計算し、Redis の
キー `state:{symbol}` に状態（特徴量配列・価格等）を定期的に公開します。

主な用途:
- 学習用および実行時のポリシー入力として最新の特徴量を配布します。

起動例:
- `python -m streamer.streamer` または Docker コンテナでの稼働。
"""

import asyncio
import json
import yaml
import redis
from pathlib import Path
from datetime import datetime, timedelta
import logging

from ..storage import DBConnection
from ..common import get_logger
from .feature_extractor import FeatureExtractor


class Streamer:
    """Feature computation and Redis streaming service."""
    
    def __init__(self, config_path: str = "common/config.yaml"):
        """初期化: 設定読み込み、DB/Redis、FeatureExtractor を初期化します。"""
        self.config = self._load_config(config_path)
        self.logger = get_logger("Streamer")
        
        self.symbol = self.config.get("symbol", "BTCUSDT")
        self.exchange = self.config.get("exchange", "binanceusdm")
        self.refresh_sec = self.config.get("state", {}).get("refresh_sec", 1.0)
        self.lookback_min = self.config.get("state", {}).get("lookback_min", 30)
        
        # Initialize DB
        DBConnection.initialize(
            host=self.config["database"]["host"],
            port=self.config["database"]["port"],
            user=self.config["database"]["user"],
            password=self.config["database"]["password"],
            dbname=self.config["database"]["dbname"],
        )
        
        # Initialize Redis
        self.redis_client = redis.Redis(
            host=self.config["redis"]["host"],
            port=self.config["redis"]["port"],
            db=self.config["redis"]["db"],
            decode_responses=True,
        )
        
        self.feature_extractor = FeatureExtractor(lookback_min=self.lookback_min)
        
        # Stats
        self.features_published = 0
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML."""
        path = Path(config_path)
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _fetch_ohlcv(self, minutes: int = 60) -> list:
        """Fetch recent OHLCV data from DB."""
        query = """
        SELECT open, high, low, close, volume, ts
        FROM ohlcv_1m
        WHERE symbol = %s AND exchange = %s
            AND ts > now() - interval '%s minutes'
        ORDER BY ts ASC
        """
        rows = DBConnection.execute_query(query, (self.symbol, self.exchange, minutes))
        
        from ..common import OHLCV
        result = []
        for row in rows:
            result.append(OHLCV(
                exchange=self.exchange,
                symbol=self.symbol,
                ts=int(row["ts"].timestamp() * 1000),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            ))
        return result
    
    def _fetch_recent_trades(self, minutes: int = 60) -> list:
        """Fetch recent trades from DB."""
        query = """
        SELECT price, size, side, ts
        FROM trades
        WHERE symbol = %s AND exchange = %s
            AND ts > now() - interval '%s minutes'
        ORDER BY ts ASC
        """
        rows = DBConnection.execute_query(query, (self.symbol, self.exchange, minutes))
        
        from ..common import Trade
        result = []
        for row in rows:
            result.append(Trade(
                exchange=self.exchange,
                symbol=self.symbol,
                ts=int(row["ts"].timestamp() * 1000),
                price=float(row["price"]),
                size=float(row["size"]),
                side=row["side"],
            ))
        return result
    
    def _fetch_latest_orderbook(self) -> 'OrderBookSnapshot':
        """Fetch latest orderbook snapshot from DB."""
        query = """
        SELECT best_bid, best_ask, bids, asks, ts
        FROM orderbook_snapshot
        WHERE symbol = %s AND exchange = %s
        ORDER BY ts DESC
        LIMIT 1
        """
        rows = DBConnection.execute_query(query, (self.symbol, self.exchange))
        
        if not rows:
            return None
        
        from ..common import OrderBookSnapshot
        row = rows[0]
        return OrderBookSnapshot(
            exchange=self.exchange,
            symbol=self.symbol,
            ts=int(row["ts"].timestamp() * 1000),
            best_bid=float(row["best_bid"]),
            best_ask=float(row["best_ask"]),
            bids=json.loads(row["bids"]),
            asks=json.loads(row["asks"]),
        )
    
    def _fetch_latest_funding_rate(self) -> float:
        """Fetch latest funding rate from DB."""
        query = """
        SELECT rate
        FROM funding_rate
        WHERE symbol = %s AND exchange = %s
        ORDER BY ts DESC
        LIMIT 1
        """
        rows = DBConnection.execute_query(query, (self.symbol, self.exchange))
        return float(rows[0]["rate"]) if rows else 0.0
    
    async def compute_and_publish(self):
        """Compute features and publish to Redis."""
        try:
            # Fetch data
            ohlcv = self._fetch_ohlcv(minutes=self.lookback_min)
            trades = self._fetch_recent_trades(minutes=self.lookback_min)
            orderbook = self._fetch_latest_orderbook()
            funding_rate = self._fetch_latest_funding_rate()
            
            if not ohlcv or not orderbook:
                self.logger.warning("Insufficient data for feature extraction")
                return
            
            # Extract features
            features = self.feature_extractor.extract_features(
                ohlcv=ohlcv,
                recent_trades=trades,
                orderbook=orderbook,
                position_flag=0,  # TODO: Get from position tracker
                time_in_position_s=0.0,
                funding_rate=funding_rate,
            )
            
            # Publish to Redis
            redis_key = f"state:{self.symbol}"
            state_data = {
                "features": features.to_array().tolist(),
                "ts": features.ts,
                "symbol": self.symbol,
                "price": float(orderbook.best_ask + orderbook.best_bid) / 2,
            }
            
            self.redis_client.set(redis_key, json.dumps(state_data), ex=60)
            self.features_published += 1
            
            if self.features_published % 60 == 0:
                self.logger.info(f"Features published: {self.features_published}")
        
        except Exception as e:
            self.logger.error(f"Feature extraction error: {e}")
    
    async def run(self):
        """Main streamer loop."""
        self.logger.info(f"Starting streamer for {self.symbol}...")
        
        try:
            while True:
                await self.compute_and_publish()
                await asyncio.sleep(self.refresh_sec)
        except KeyboardInterrupt:
            self.logger.info("Streamer stopped by user")
        except Exception as e:
            self.logger.error(f"Streamer error: {e}")
        finally:
            DBConnection.close_pool()
            self.redis_client.close()
            self.logger.info(f"Streamer stopped. Features published: {self.features_published}")


async def main():
    """Main entry point."""
    streamer = Streamer()
    await streamer.run()


if __name__ == "__main__":
    asyncio.run(main())
