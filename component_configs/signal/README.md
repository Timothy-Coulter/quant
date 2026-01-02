# Signal templates

Standardised signal payloads for downstream validation or fixtures. These map to the `SignalGenerator` schema (type/action/confidence/timestamp/metadata).

- `bullish_breakout.yaml`: BUY signal with high confidence and breakout metadata.
- `risk_off.yaml`: HOLD/exit recommendation tagged for de-risking.

Useful for stubbing signal buses and schema validation in smoke tests.
