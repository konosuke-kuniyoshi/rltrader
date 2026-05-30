# Current Volume Snapshot

Binance API キーの発行待ちの間に、現時点での出来高情報や進捗を記録するためのファイルです。

## 現在の状況
- Binance アカウント: 審査中
- API キー: 未取得
- 作成日: 2026-05-31

## 現在の出来高
- 取引ペア: BTCUSDT
- 取引所: Binance (テストネット / 本番)
- 現在の出来高: `TODO`  
  ※ 実データがある場合はここに記入してください。

## メモ
- API キー発行後に以下を実行して更新してください:
  1. `ops/.env` に `BINANCE_API_KEY` / `BINANCE_API_SECRET` を設定
  2. `docker-compose up -d --no-deps --build collector`
  3. `docker-compose logs -f collector` でデータ収集を確認
  4. 収集済み出来高をこのファイルに追記

## GitHub への保存
```bash
git add CURRENT_VOLUME.md
git commit -m "Add current volume snapshot placeholder"
git push origin main
```

---

このファイルは、Binance API キーが取得できるまでの間の進捗記録用です。