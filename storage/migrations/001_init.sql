-- TimescaleDB schema for RLTrader
-- Initialize with: psql -U postgres -h localhost -d rltrader -f storage/migrations/001_init.sql

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Metadata table for symbols
CREATE TABLE IF NOT EXISTS meta_symbols (
    exchange TEXT NOT NULL PRIMARY KEY,
    symbol TEXT NOT NULL,
    tick_size NUMERIC NOT NULL,
    min_qty NUMERIC NOT NULL,
    maker_fee NUMERIC NOT NULL,
    taker_fee NUMERIC NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Trades table (hypertable with daily partitioning)
CREATE TABLE IF NOT EXISTS trades (
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    price NUMERIC NOT NULL,
    size NUMERIC NOT NULL,
    side TEXT NOT NULL,
    buyer_maker BOOLEAN,
    trade_id BIGINT,
    PRIMARY KEY (exchange, symbol, ts, price, size)
);

-- Convert trades to hypertable (daily chunks)
SELECT create_hypertable('trades', 'ts', if_not_exists => TRUE, chunk_time_interval => '1 day'::interval);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_ts ON trades (symbol, ts DESC);

-- Order book snapshots (hypertable with hourly partitioning)
CREATE TABLE IF NOT EXISTS orderbook_snapshot (
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    best_bid NUMERIC NOT NULL,
    best_ask NUMERIC NOT NULL,
    bids JSONB NOT NULL,
    asks JSONB NOT NULL,
    PRIMARY KEY (exchange, symbol, ts)
);

-- Convert to hypertable (hourly chunks)
SELECT create_hypertable('orderbook_snapshot', 'ts', if_not_exists => TRUE, chunk_time_interval => '1 hour'::interval);
CREATE INDEX IF NOT EXISTS idx_orderbook_symbol_ts ON orderbook_snapshot (symbol, ts DESC);

-- OHLCV 1-minute bars (hypertable with weekly partitioning)
CREATE TABLE IF NOT EXISTS ohlcv_1m (
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (exchange, symbol, ts)
);

-- Convert to hypertable (weekly chunks)
SELECT create_hypertable('ohlcv_1m', 'ts', if_not_exists => TRUE, chunk_time_interval => '7 days'::interval);
CREATE INDEX IF NOT EXISTS idx_ohlcv_1m_symbol_ts ON ohlcv_1m (symbol, ts DESC);

-- Funding rates (hypertable with 30-day partitioning)
CREATE TABLE IF NOT EXISTS funding_rate (
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    rate NUMERIC NOT NULL,
    PRIMARY KEY (exchange, symbol, ts)
);

-- Convert to hypertable (30-day chunks)
SELECT create_hypertable('funding_rate', 'ts', if_not_exists => TRUE, chunk_time_interval => '30 days'::interval);
CREATE INDEX IF NOT EXISTS idx_funding_rate_symbol_ts ON funding_rate (symbol, ts DESC);

-- Continuous aggregates for faster queries
CREATE MATERIALIZED VIEW IF NOT EXISTS trades_5m_agg
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('5 minutes', ts) as bucket,
    MIN(price) as low,
    MAX(price) as high,
    FIRST(price, ts) as open,
    LAST(price, ts) as close,
    SUM(size) as volume
FROM trades
WHERE exchange = 'binanceusdm'
GROUP BY symbol, bucket
WITH NO DATA;

-- Refresh policy for continuous aggregate
SELECT add_continuous_agg_policy('trades_5m_agg', start_offset => '1 day'::interval, if_not_exists => true);

-- Retention policy (keep 90 days of high-frequency data, 2 years for aggregates)
SELECT add_retention_policy('trades', INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_retention_policy('orderbook_snapshot', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_retention_policy('ohlcv_1m', INTERVAL '2 years', if_not_exists => TRUE);
SELECT add_retention_policy('funding_rate', INTERVAL '1 year', if_not_exists => TRUE);

-- Insert default symbol metadata for BTCUSDT
INSERT INTO meta_symbols (exchange, symbol, tick_size, min_qty, maker_fee, taker_fee)
VALUES ('binanceusdm', 'BTCUSDT', 0.01, 0.001, 0.0002, 0.0004)
ON CONFLICT (exchange) DO NOTHING;
