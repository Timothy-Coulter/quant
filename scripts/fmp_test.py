"""Test module for FMP data retrieval functionality.

This script demonstrates and tests the FMP (Financial Modeling Prep) data
retrieval functionality using the fmp-python wrapper library.
"""

import os

from fmp_python.fmp import FMP

api_key = os.getenv("FMP_API_KEY")
fmp = FMP(api_key=api_key, output_format='pandas')

# Example: get historical price for one ticker
# (Note: you’ll need to check the correct endpoint and parameters for 1-minute interval)
ticker = 'AAPL'
# Example endpoint (check docs for correct one):
df_price = fmp.get_historical_price(symbol=ticker)

# Suppose df_price is a Pandas DataFrame with datetime index and columns such as open/close/volume
df_price = df_price.set_index('date')  # adjust according to wrapper output
# Then you can slice / resample / align to your sequence length
