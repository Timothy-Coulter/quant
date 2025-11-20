"""Test module for FDPY data retrieval functionality.

This script demonstrates and tests the FDPY data retrieval functionality
using the backtester framework for fetching financial data.
"""

import os
import time

import pandas as pd
from dotenv import load_dotenv
from findatapy.market import Market, MarketDataGenerator, MarketDataRequest
from findatapy.util import DataConstants, LoggerManager

# Load variables from .env file
load_dotenv()

market = Market(market_data_generator=MarketDataGenerator())

# Get you FRED API key from https://fred.stlouisfed.org/docs/api/api_key.html
fred_api_key = os.getenv("FRED_API_KEY")
alpha_vantage_api_key = os.getenv("ALPHAVANTAGE_API_KEY")

fields = ["open", "close", "high", "low"]

url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
df = pd.read_csv(url)
tickers = sorted(df['Symbol'].tolist())
print(len(tickers))
print(tickers)

# # Download equities data from Alpha Vantage
# md_request = MarketDataRequest(
#     start_date="decade",  # start date
#     data_source="yahoo",  # use Bloomberg as data source
#     freq="daily",
#     gran_freq="daily",
#     tickers=["Apple", "Microsoft", "Citigroup", "S&P500-ETF"],  # ticker (findatapy)
#     fields=fields,  # which fields to download
#     vendor_tickers=["aapl", "msft", "c", "spy"],  # ticker (Alpha Vantage)
#     vendor_fields=[f.title() for f in fields])  # which Bloomberg fields to download)

# df = market.fetch_market(md_request)
# print(df.tail(n=10))
# print(df.head())
# print(df.columns)

# Option 1: Use the DataHub CSV


tickers = [t for t in tickers if t not in ["WBA", "BRK.B", "BF.B"]]

DataConstants.market_thread_technique = "thread"

logger = LoggerManager().getLogger(__name__)

md_request = MarketDataRequest(
    start_date="decade",  # start date
    data_source="yahoo",  # use Bloomberg as data source
    freq="daily",
    gran_freq="daily",
    tickers=tickers,  # ticker (findatapy)
    fields=fields,  # which fields to download
    vendor_tickers=tickers,  # ticker (Alpha Vantage)
    vendor_fields=[f.title() for f in fields],
)  # which Bloomberg fields to download)

t1 = time.time()

df = market.fetch_market(md_request)

t2 = time.time()

logger.info("Loaded data from yahoo directly, now try reading from Redis in-memory cache")
md_request.cache_algo = "cache_algo_return"  # change flag to cache algo
# so won"t attempt to download via web

t3 = time.time()

df = market.fetch_market(md_request)

t4 = time.time()

print(df)

logger.info("Read from Redis cache.. that was a lot quicker!")

print(t2 - t1)
print(t4 - t3)
