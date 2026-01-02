# Data retrieval presets

`DataRetrievalConfig` YAMLs for common markets and cadences.

- `yahoo_daily.yaml`: EOD U.S. equities with cache enabled.
- `alpha_intraday.yaml`: 1h intraday equities with Alpha Vantage style throttling.
- `crypto_binance.yaml`: Binance spot candles at 1h frequency for BTC/ETH.

Each file sets tickers, dates, frequency, cache behaviour, and any API key placeholders expected by `DataRetrieval`.
