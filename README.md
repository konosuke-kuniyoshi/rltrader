# RLTrader: RL-Based Crypto Automated Trading System

Binance USDT-M Perpetual（BTCUSDT）を対象に、強化学習（PPO）で売買方針を学習・実行する最小構成の暗号資産自動売買システムです。

## プロジェクト概要

### アーキテクチャ

```
┌──────────────────────────────────────────────────────────────┐
│ Binance USDT-M Futures (Testnet)                             │
└──────────────────────┬───────────────────────────────────────┘
                       │ WebSocket (trades, depth, funding)
                       ▼
         ┌─────────────────────────────┐
         │    Collector (WS Client)    │
         │  - Sync & Gap Detection     │
         │  - 6h Continuous           │
         └──────────┬──────────────────┘
                    │
                    ▼
         ┌─────────────────────────────┐
         │  TimescaleDB (Time-Series)  │
         │  - trades, orderbook_snapshot
         │  - ohlcv_1m, funding_rate   │
         │  - 90d retention            │
         └──────────┬──────────────────┘
                    │
                    ▼
         ┌─────────────────────────────┐
         │ Streamer (Feature Extraction│
         │  - 36-dim Feature Vector    │
         │  - 1s Refresh Cycle         │
         └──────────┬──────────────────┘
                    │
                    ▼
         ┌─────────────────────────────┐
         │   Redis (State Distribution)│
         │  state:BTCUSDT (JSON)       │
         │  signals:BTCUSDT (Queue)    │
         └──────────┬──────────────────┘
                   / \
                  /   \
                 ▼     ▼
   ┌──────────────────┐  ┌──────────────────┐
   │ Gym Environment  │  │ Policy Runner    │
   │ (8 parallel)     │  │ (Testnet Executor│
   │ - PPO Training   │  │ - CCXT Integration
   │ - 1M timesteps   │  │ - Risk Guards    │
   └──────────────────┘  └──────────────────┘
```

### コンポーネント

| コンポーネント | 役割 | 技術 |
|---|---|---|
| **Collector** | Binance WSでtrades/depth購読→DB保存 | aiohttp, psycopg2 |
| **Streamer** | DB→特徴量36次元→Redis配信 | pandas, numpy |
| **Gym Env** | 強化学習環境 | gymnasium, gym, Redis |
| **Trainer** | PPO学習 | stable-baselines3 |
| **Executor** | テストネット発注 | ccxt, Redis |
| **Infra** | DB/キャッシュ/監視 | TimescaleDB, Redis, Prometheus, Grafana |

---

## セットアップ

### 前提条件

- Docker & Docker Compose v3.9+
- Binance Futures テストネットアカウント（APIキー取得済み）
- 8GB RAM, 2+ CPU cores（GPU推奨：学習時）

### インストール手順

1. **リポジトリクローン＆環境変数設定**
```bash
cd /path/to/trade-on/rltrader

# .env 作成
cp ops/.env.example .env

# 秘密情報を編集
# DB_PASSWORD, BINANCE_API_KEY, BINANCE_API_SECRET を設定
nano .env
```

2. **DB初期化**
```bash
docker-compose up -d timescaledb

# スキーマ作成待機（数秒）
sleep 5

# マイグレーション実行確認
docker-compose exec timescaledb psql -U postgres -d rltrader -f /docker-entrypoint-initdb.d/001_init.sql
```

3. **全サービス起動**
```bash
docker-compose up -d

# ヘルスチェック
docker-compose ps
```

---

## 使用方法

### 1. データ収集（6時間連続）

Collector がBinanceから自動的にデータ取得を開始します。

```bash
# ログ確認
docker-compose logs -f collector

# DB内のデータ確認
docker-compose exec timescaledb psql -U postgres -d rltrader -c \
  "SELECT COUNT(*) as trade_count FROM trades WHERE symbol='BTCUSDT';"
```

### 1a. 便利スクリプト: Collector を 6 時間実行して自動停止

リポジトリ内の `ops/collector_run.sh` (Unix) または `ops/collector_run.ps1` (Windows PowerShell) を使うと
Collector を起動して6時間後に自動的に停止できます。

Unix/macOS (bash):
```bash
./ops/collector_run.sh
```

Windows PowerShell:
```powershell
.\
ops\collector_run.ps1
```

**期待値（6時間）:**
- Trades: 10万～100万件
- Orderbook snapshots: 36,000件（10秒ごと）

### 2. 特徴量ストリーミング

Streamer がDB→Redis で1秒ごとに特徴量を配信します。

```bash
# ストリーマーログ
docker-compose logs -f streamer

# Redis でFeatures確認
docker-compose exec redis redis-cli GET "state:BTCUSDT" | jq .
```

### 3. PPO学習（バックテスト環境）

```bash
# 学習スタート（8並列環境, 1M timesteps, ~30分GPU）
docker-compose exec trainer \
  python -m rltrader.envs.train_ppo \
    --config common/config.yaml \
    --timesteps 1000000

# または Docker 内で実行
docker-compose run --rm trainer \
  python -m rltrader.envs.train_ppo \
    --config common/config.yaml \
    --timesteps 1000000

# モデル出力
ls models/
# → best_model.zip, ppo_final_BTCUSDT.zip
```

### 4. モデル評価

```bash
docker-compose run --rm trainer \
  python -m rltrader.envs.train_ppo \
    --config common/config.yaml \
    --eval-only \
    --model-path models/best_model.zip
```

### 5. テストネット実行

```bash
# Executor サービス起動
docker-compose up -d executor

# ポリシー実行ログ
docker-compose logs -f executor

# Redisシグナルキュー確認
docker-compose exec redis redis-cli LLEN "signals:BTCUSDT"
```

---

## 設定パラメータ

[common/config.yaml](common/config.yaml) で以下を調整可能：

### 取引設定
```yaml
exchange: binanceusdm
symbol: BTCUSDT
fees:
  maker: 0.0002    # 0.02%
  taker: 0.0004    # 0.04%
slippage:
  rate: 0.0003     # 0.03% (片道)
funding:
  per_8h: 0.0001   # 0.01% / 8h
```

### リスク制約
```yaml
risk:
  per_trade_loss_cap: 0.02      # 1トレード最大損失 = 資産の2%
  daily_dd_stop: 0.03           # 日次DD上限 = 3%
  max_leverage: 3.0             # 最大レバレッジ
  max_notional_by_equity: 1.0   # ポジション額 ≤ 資産額
```

### 学習設定
```yaml
training:
  total_timesteps: 1000000
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
```

---

## 特徴量セット（36次元）

| グループ | 特徴量 | 計数 |
|---------|--------|------|
| **価格/トレンド** | ret_15s/1m/5m/15m, sma_ratio_5_20, sma_ratio_20_60 | 6 |
| **ボラティリティ** | realized_vol_5m, realized_vol_15m, atr14_1m, atr14_5m | 4 |
| **テクニカル** | macd, macd_signal, macd_hist, rsi14 | 4 |
| **板/流動性** | spread_in_ticks, depth_bid/ask5/10, order_imbalance | 6 |
| **インパクト** | impact_price_10k, impact_price_50k, last10_trades_buy_ratio, last10_trades_avg_size | 4 |
| **ボリューム/レジーム** | vol_1m_z, vol_5m_z, session_flag, funding_recent, funding_pred | 5 |
| **スキューネス/カートシス** | realized_skew_5m, realized_kurt_5m | 2 |
| **ポジション情報** | position_flag, time_in_position_clipped | 2 |

**正規化規約:** 価格水準は不使用。差分・リターン・ATR・出来高基準で無次元化。

---

## 報酬関数

```
R_t = ΔPnL(t) - fee(t) - slippage(t) - funding(t) - λ_trade × I_trade(t)

ΔPnL = unrealized PnL 変化
fee = taker_fee × notional（成行）
slippage = slippage_rate × notional
funding = funding_rate × notional × time_in_position × position_sign
I_trade = トレード発生フラグ（λ ≈ 0.001）
```

**リスク制約:**
- **per_trade_loss_cap:** SLを ATR から推定 → size = min(equity×2%/sl_dist, equity×max_notional, leverage×equity)
- **daily_dd_stop:** 日初equity × 0.97 割ったらエピソード強制終了
- **max_leverage, max_notional_by_equity:** リアルタイム検証

---

## 監視・デバッグ

### Grafana ダッシュボード

http://localhost:3000 (admin / admin)

- Prometheus datasource 自動接続
- Trade PnL, Equity Curve
- Position Size, Leverage
- Fee累計, Funding Cost

### Prometheus メトリクス

http://localhost:9090

- timescaledb 稼働状況
- redis メモリ使用量
- docker containers リソース

### ログファイル

```bash
# 各サービスログ
ls logs/
# → Collector-*.log, Streamer-*.log, ExecutionEngine-*.log

# Docker コンテナログ
docker-compose logs [service_name]
```

### DB クエリ例

```bash
# 最新100件のトレード
docker-compose exec timescaledb psql -U postgres -d rltrader -c \
  "SELECT ts, price, size, side FROM trades WHERE symbol='BTCUSDT' \
   ORDER BY ts DESC LIMIT 100;"

# 過去1時間の OHLCV
docker-compose exec timescaledb psql -U postgres -d rltrader -c \
  "SELECT ts, open, high, low, close, volume FROM ohlcv_1m \
   WHERE symbol='BTCUSDT' AND ts > now() - interval '1 hour' \
   ORDER BY ts ASC;"

# 資金調達レート推移
docker-compose exec timescaledb psql -U postgres -d rltrader -c \
  "SELECT ts, rate FROM funding_rate WHERE symbol='BTCUSDT' \
   ORDER BY ts DESC LIMIT 10;"
```

---

## テスト/検証チェックリスト

- [ ] **データ収集:** 連続6hで欠損なし、重複なし、整合性OK
  - Collector ログで "Trades inserted: X000" を確認
  - `SELECT COUNT(DISTINCT trade_id) FROM trades` で重複チェック
  
- [ ] **特徴量計算:** Redis `state:BTCUSDT` に36次元が1秒ごと配信
  - `redis-cli GET "state:BTCUSDT" | jq '.features | length'` = 36
  - `redis-cli MONITOR` でキー更新確認

- [ ] **学習エピソード:** Gym環境が正常に動作、報酬が増加傾向
  - trainer ログで `Ep 100: mean reward = X` を確認
  - `models/` に best_model.zip が生成

- [ ] **執行層:** テストネット発注が正常、リスク制約が機能
  - executor ログで `Action: 1, Size: 0.XXXX, Price: $XXXXX` を確認
  - リスク違反時 `Risk constraint violated` メッセージを確認

---

## トラブルシューティング

### Q: "No market data in Redis" が頻出

→ Streamer が DB からデータを読めていません：
- Collector が正常動作しているか確認
- TimescaleDB 接続確認：`docker-compose exec timescaledb psql -U postgres -d rltrader -c "SELECT COUNT(*) FROM trades;"`
- Streamer ログで SQL エラー確認

### Q: "Risk constraint violated" で即座に終了

→ Gym 環境でリスクチェックが厳しすぎます：
- `config.yaml` の `per_trade_loss_cap` を 0.05 に緩和（テスト用）
- initial_capital を増やす

### Q: executor が testnet に接続できない

→ API キー/シークレット確認：
- `.env` ファイルが正しく設定されているか確認
- Binance Futures テストネット URL が正しいか：https://testnet.binancefuture.com

### Q: Docker メモリ不足

→ docker-compose.yml で memory limit を設定：
```yaml
services:
  trainer:
    deploy:
      resources:
        limits:
          memory: 4G
```

---

## ファイル構成

```
rltrader/
├── common/
│   ├── config.yaml          # 統一設定ファイル
│   ├── types.py             # 型定義
│   ├── logger.py            # ロギングユーティリティ
│   └── __init__.py
├── storage/
│   ├── migrations/
│   │   └── 001_init.sql     # TimescaleDB スキーマ
│   ├── connection.py        # DB接続管理
│   └── __init__.py
├── collector/
│   ├── ws_client.py         # Binance WS クライアント
│   ├── collector.py         # メインコレクタースクリプト
│   └── __init__.py
├── streamer/
│   ├── feature_extractor.py # 特徴量計算エンジン
│   ├── streamer.py          # メインストリーマースクリプト
│   └── __init__.py
├── envs/
│   ├── crypto_env.py        # Gym 環境
│   ├── train_ppo.py         # PPO 学習スクリプト
│   └── __init__.py
├── execution/
│   ├── execution.py         # CCXT 実行エンジン
│   ├── runner.py            # ポリシー実行スクリプト
│   └── __init__.py
├── ops/
│   ├── docker-compose.yml   # 全サービスオーケストレーション
│   ├── Dockerfile.collector # Collector イメージ
│   ├── Dockerfile.streamer  # Streamer イメージ
│   ├── Dockerfile.trainer   # Trainer イメージ（GPU対応）
│   ├── Dockerfile.executor  # Executor イメージ
│   ├── .env.example         # 環境変数テンプレート
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── grafana/
│       └── provisioning/
│           └── provisioning.yaml
├── requirements.txt         # Python 依存パッケージ
├── requirements-trainer.txt # 学習用追加パッケージ
└── README.md                # このファイル
```

---

## Next Steps

### Phase 2: 本番化

- [ ] 取引所実ネット接続（CCXT で本番 API endpoint）
- [ ] 複数シンボル対応（ETHUSDT, BNBUSDT 等）
- [ ] リアルタイム価格アラート（Telegram 通知）
- [ ] バックテスト最適化（vectorbt 統合）
- [ ] モデルアンサンブル（複数ポリシーの投票）
- [ ] 動的レバレッジ調整

### Phase 3: 高度な戦略

- [ ] マルチタイムフレーム学習
- [ ] クロスチェーン裁定
- [ ] オプション戦略統合
- [ ] マイクロ構造モデリング

---

## ライセンス

MIT License

---

## 参考文献

- [Binance USDT-M Futures API](https://binance-docs.github.io/apidocs/futures/en/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [TimescaleDB Hypertables](https://docs.timescale.com/timescaledb/latest/)
- [CCXT Documentation](https://docs.ccxt.com/)

---

**最終更新:** 2026-05-31  
**保守者:** RL Trader Team
