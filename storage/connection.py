"""
データベース接続ユーティリティ

`DBConnection` は psycopg2 のコネクションプールを管理し、クエリ実行の
ヘルパーを提供します。設定は環境変数または `initialize` 引数で指定できます。

環境変数のデフォルト:
- `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`

使い方:
- 起動時に `DBConnection.initialize(...)` を呼んでから `execute_query` / `execute_update` を利用します。
"""

import os
from contextlib import contextmanager
from typing import Generator, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool


class DBConnection:
    """Database connection manager."""
    
    _pool: Optional[SimpleConnectionPool] = None
    
    @classmethod
    def initialize(
        cls,
        host: str = os.getenv("DB_HOST", "localhost"),
        port: int = int(os.getenv("DB_PORT", "5432")),
        user: str = os.getenv("DB_USER", "postgres"),
        password: str = os.getenv("DB_PASSWORD", "postgres"),
        dbname: str = os.getenv("DB_NAME", "rltrader"),
        minconn: int = 1,
        maxconn: int = 10,
    ):
        """Initialize connection pool."""
        if cls._pool is not None:
            return
        
        cls._pool = SimpleConnectionPool(
            minconn,
            maxconn,
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=dbname,
            connect_timeout=10,
        )
    
    @classmethod
    @contextmanager
    def get_connection(cls) -> Generator:
        """Get a connection from the pool."""
        if cls._pool is None:
            cls.initialize()
        
        conn = cls._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cls._pool.putconn(conn)
    
    @classmethod
    def execute_query(cls, query: str, params: tuple = ()) -> list:
        """Execute SELECT query."""
        with cls.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                return cur.fetchall()
    
    @classmethod
    def execute_update(cls, query: str, params: tuple = ()) -> int:
        """Execute INSERT/UPDATE/DELETE query."""
        with cls.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.rowcount
    
    @classmethod
    def close_pool(cls):
        """Close all connections in pool."""
        if cls._pool:
            cls._pool.closeall()
            cls._pool = None
