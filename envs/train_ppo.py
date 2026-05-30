"""
PPO 学習スクリプト (暗号資産トレーディング)

このファイルは `stable-baselines3` を使って PPO エージェントを学習するための
トレーナークラス `CryptoTrainer` を提供します。

使い方 (CLI):
- 学習: `python -m envs.train_ppo`
- 引数で設定ファイルや学習ステップ数を上書きできます。

設定:
- `common/config.yaml` の `training` セクションでハイパーパラメータを指定してください。
"""

import gym
import numpy as np
import yaml
import json
from pathlib import Path
from typing import Optional, Dict
import logging

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.env_util import make_vec_env

from ..common import get_logger
from .crypto_env import CryptoTradingEnv


class CryptoTrainer:
    """Trainer for PPO policy on crypto trading environment."""
    
    def __init__(self, config_path: str = "common/config.yaml"):
        """初期化: 設定読み込みと出力ディレクトリの準備を行います。"""
        self.config = self._load_config(config_path)
        self.logger = get_logger("CryptoTrainer")
        
        self.symbol = self.config.get("symbol", "BTCUSDT")
        self.output_dir = Path("models")
        self.output_dir.mkdir(exist_ok=True)
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML."""
        path = Path(config_path)
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_env(self, num_envs: int = 8) -> gym.Env:
        """Create vectorized environment with normalization."""
        
        def env_fn():
            env = CryptoTradingEnv(
                symbol=self.symbol,
                initial_capital=10000.0,
                max_steps=10000,
                config=self.config,
            )
            return env
        
        # Create vectorized env
        env = make_vec_env(env_fn, n_envs=num_envs, seed=self.config["training"]["seed"])
        
        # Apply observation/reward normalization
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        
        return env
    
    def _create_eval_env(self) -> gym.Env:
        """Create evaluation environment."""
        env = CryptoTradingEnv(
            symbol=self.symbol,
            initial_capital=10000.0,
            max_steps=10000,
            config=self.config,
        )
        return env
    
    def train(self, total_timesteps: Optional[int] = None, use_wandb: bool = False):
        """Train PPO agent."""
        
        total_timesteps = total_timesteps or self.config["training"]["total_timesteps"]
        
        self.logger.info(f"Starting training for {total_timesteps} timesteps...")
        
        # Create environments
        env = self._create_env(num_envs=self.config["training"].get("n_envs", 8))
        eval_env = self._create_eval_env()
        
        # Create callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.output_dir),
            log_path=str(self.log_dir),
            eval_freq=max(total_timesteps // 20, 5000),
            deterministic=True,
            render=False,
        )
        
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=self.config["training"].get("early_stopping_patience", 5),
            min_evals=3,
            verbose=1,
        )
        
        # Create policy
        policy_kwargs = {
            "net_arch": [256, 256],
            "activation_fn": None,  # Use ReLU by default
        }
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.config["training"]["learning_rate"],
            n_steps=self.config["training"]["n_steps"],
            batch_size=self.config["training"]["batch_size"],
            n_epochs=self.config["training"]["n_epochs"],
            gamma=self.config["training"]["gamma"],
            gae_lambda=self.config["training"]["gae_lambda"],
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(self.log_dir),
            seed=self.config["training"]["seed"],
        )
        
        try:
            # Train
            model.learn(
                total_timesteps=total_timesteps,
                callback=[eval_callback],
                log_interval=100,
                progress_bar=True,
            )
            
            # Save final model
            model_path = self.output_dir / f"ppo_final_{self.symbol}"
            model.save(str(model_path))
            self.logger.info(f"Model saved to {model_path}")
            
            # Save normalization stats
            norm_path = self.output_dir / f"vec_normalize_{self.symbol}.pkl"
            env.save(str(norm_path))
            self.logger.info(f"Normalization stats saved to {norm_path}")
        
        except KeyboardInterrupt:
            self.logger.info("Training stopped by user")
        except Exception as e:
            self.logger.error(f"Training error: {e}")
        
        finally:
            env.close()
            eval_env.close()
    
    def evaluate(self, model_path: str, num_episodes: int = 10):
        """Evaluate trained model."""
        
        self.logger.info(f"Evaluating model from {model_path}...")
        
        # Load model
        model = PPO.load(model_path)
        
        # Create evaluation environment
        env = self._create_eval_env()
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0.0
            step = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                step += 1
            
            episode_rewards.append(episode_reward)
            self.logger.info(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Steps={step}")
        
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        self.logger.info(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
        
        env.close()
        
        return {
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "rewards": episode_rewards,
        }


def main():
    """Main entry point for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PPO agent for crypto trading")
    parser.add_argument("--config", type=str, default="common/config.yaml", help="Config file path")
    parser.add_argument("--timesteps", type=int, default=None, help="Total timesteps for training")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate (requires --model-path)")
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained model for evaluation")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases for logging")
    
    args = parser.parse_args()
    
    trainer = CryptoTrainer(config_path=args.config)
    
    if args.eval_only:
        if not args.model_path:
            print("Error: --model-path required for evaluation")
            return
        trainer.evaluate(args.model_path)
    else:
        trainer.train(total_timesteps=args.timesteps, use_wandb=args.wandb)


if __name__ == "__main__":
    main()
