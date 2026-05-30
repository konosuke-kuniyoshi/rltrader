"""Main runner for policy execution on testnet."""

import asyncio
import json
import yaml
import redis
import numpy as np
from pathlib import Path
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from ..common import get_logger
from .execution import ExecutionEngine


class PolicyRunner:
    """Run trained policy on testnet with live execution."""
    
    def __init__(self, model_path: str, config_path: str = "common/config.yaml"):
        """初期化: モデルと正規化情報、実行エンジン、Redis クライアントを準備します。

        - `model_path`: 学習済み PPO モデルのパス
        - `config_path`: 設定ファイルのパス
        """
        self.config = self._load_config(config_path)
        self.logger = get_logger("PolicyRunner")
        
        self.symbol = self.config.get("symbol", "BTCUSDT")
        
        # Load model and normalization stats
        self.model = PPO.load(model_path)
        
        norm_path = str(Path(model_path).parent / f"vec_normalize_{self.symbol}.pkl")
        try:
            self.vec_normalize = VecNormalize.load(norm_path, None)
        except:
            self.vec_normalize = None
            self.logger.warning(f"Could not load normalization stats from {norm_path}")
        
        # Initialize execution
        self.execution_engine = ExecutionEngine(config_path=config_path)
        
        # Redis for state/signals
        self.redis_client = redis.Redis(
            host=self.config["redis"]["host"],
            port=self.config["redis"]["port"],
            db=self.config["redis"]["db"],
            decode_responses=True,
        )
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML."""
        path = Path(config_path)
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    async def run(self, max_iterations: Optional[int] = None):
        """Run policy with live trading."""
        
        self.logger.info(f"Starting policy runner for {self.symbol}...")
        
        iteration = 0
        
        try:
            while max_iterations is None or iteration < max_iterations:
                
                # Get market state from Redis
                redis_key = f"state:{self.symbol}"
                state_data = self.redis_client.get(redis_key)
                
                if not state_data:
                    self.logger.warning("No market state available")
                    await asyncio.sleep(1)
                    continue
                
                state_obj = json.loads(state_data)
                obs = np.array(state_obj["features"], dtype=np.float32)
                price = state_obj.get("price", 0.0)
                
                # Apply normalization if available
                if self.vec_normalize:
                    obs = self.vec_normalize.normalize_obs(obs.reshape(1, -1))[0]
                
                # Get policy action
                try:
                    action, _ = self.model.predict(obs, deterministic=True)
                except Exception as e:
                    self.logger.error(f"Policy prediction error: {e}")
                    await asyncio.sleep(1)
                    continue
                
                # Get position info for size calculation
                position = self.execution_engine.get_position()
                equity = self.execution_engine.get_account_balance()
                
                # Size determination based on risk rule
                atr = obs[6] if len(obs) > 6 else 0.01
                sl_distance = max(atr * price, 1.0)
                
                size_by_loss_cap = (equity * self.config["risk"]["per_trade_loss_cap"]) / sl_distance
                size_by_notional = equity * self.config["risk"]["max_notional_by_equity"] / price
                size_by_leverage = self.config["risk"]["max_leverage"] * equity / price
                
                new_size = min(size_by_loss_cap, size_by_notional, size_by_leverage)
                
                # Create signal
                signal = {
                    "action": int(action),
                    "size": float(new_size),
                    "price": price,
                    "ts": state_obj.get("ts", 0),
                }
                
                # Put signal into queue for execution
                signal_queue = f"signals:{self.symbol}"
                self.redis_client.rpush(signal_queue, json.dumps(signal))
                
                self.logger.info(
                    f"Action: {action}, Size: {new_size:.4f}, Price: ${price:.2f}, "
                    f"Equity: ${equity:.2f}"
                )
                
                iteration += 1
                await asyncio.sleep(self.config.get("state", {}).get("refresh_sec", 1.0))
        
        except KeyboardInterrupt:
            self.logger.info("Policy runner stopped by user")
        except Exception as e:
            self.logger.error(f"Policy runner error: {e}")
        finally:
            # Cleanup: close position and cancel orders
            self.execution_engine.close_position()
            self.execution_engine.cancel_all_orders()
            self.logger.info("Cleanup completed")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run trained policy on testnet")
    parser.add_argument("model_path", type=str, help="Path to trained PPO model")
    parser.add_argument("--config", type=str, default="common/config.yaml", help="Config file path")
    parser.add_argument("--max-iter", type=int, default=None, help="Max iterations")
    
    args = parser.parse_args()
    
    runner = PolicyRunner(model_path=args.model_path, config_path=args.config)
    await runner.run(max_iterations=args.max_iter)


if __name__ == "__main__":
    asyncio.run(main())
