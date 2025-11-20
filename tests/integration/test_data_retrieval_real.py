"""Integration test for the DataRetrieval cache-first workflow."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd

from backtester.data.data_retrieval import (
    DataRetrieval,
    DataRetrievalConfig,
    clear_data_retrieval_cache,
)


def test_data_retrieval_reuses_cache_before_downloading(monkeypatch: Any) -> None:
    """Ensure DataRetrieval loads cached data once and serves all subsequent requests from memory."""
    clear_data_retrieval_cache()
    config = DataRetrievalConfig(
        data_source="yahoo",
        tickers=["SPY"],
        fields=["close", "open"],
        start_date="2024-01-01",
        finish_date="2024-01-10",
    )
    sample_data = pd.DataFrame(
        {
            "Close": [500.0, 501.0, 502.0],
            "Open": [499.0, 500.0, 501.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    monkeypatch.setenv("FRED_API_KEY", "demo-key")

    with (
        patch("backtester.data.data_retrieval.Market") as mock_market_cls,
        patch("backtester.data.data_retrieval.MarketDataGenerator"),
        patch("backtester.data.data_retrieval.DataQuality"),
    ):
        mock_market = MagicMock()
        mock_market.fetch_market.return_value = sample_data
        mock_market_cls.return_value = mock_market

        retrieval = DataRetrieval(config)
        assert retrieval.config.fred_api_key == "demo-key"

        first = retrieval.get_data()
        pd.testing.assert_frame_equal(first, sample_data)

        second = retrieval.get_data()
        assert mock_market.fetch_market.call_count == 1
        pd.testing.assert_frame_equal(second, sample_data)

        retrieval_again = DataRetrieval(config)
        third = retrieval_again.get_data()
        assert mock_market.fetch_market.call_count == 1
        pd.testing.assert_frame_equal(third, sample_data)

    clear_data_retrieval_cache()
