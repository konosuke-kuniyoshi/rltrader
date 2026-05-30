"""
ロギングユーティリティ

このモジュールはプロジェクト全体で利用するロガーを生成します。
標準出力とファイルへの出力を設定し、名前付きロガーを返します。

使い方:
- `get_logger("MyComponent")` を呼んでロガーを取得してください。
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """Create a logger instance."""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_path / f"{name}.log")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger
