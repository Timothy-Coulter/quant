"""Data retrieval classes for loading market data with cache-first logic."""

import os
import time
from threading import Event, Lock
from typing import Any

import pandas as pd
from findatapy.market import Market, MarketDataGenerator, MarketDataRequest
from findatapy.timeseries import DataQuality
from findatapy.util import LoggerManager

from backtester.core.config import DataRetrievalConfig
from backtester.utils.cache_utils import FrameCache

_DATA_CACHE = FrameCache()


def _build_cache_key(config: DataRetrievalConfig) -> tuple[Any, ...]:
    tickers = config.tickers
    if isinstance(tickers, str):
        tickers_tuple: tuple[Any, ...] = (tickers,)
    elif tickers is None:
        tickers_tuple = ()
    else:
        tickers_tuple = tuple(tickers)
    return (
        config.data_source,
        tickers_tuple,
        str(config.start_date),
        str(config.finish_date),
        str(config.freq),
        tuple(config.fields),
    )


class DataRetrieval:
    """Data retrieval class that implements cache-first logic.

    This class attempts to load data from cache first, and if that fails,
    it falls back to downloading from the data source.
    """

    def __init__(self, config: DataRetrievalConfig | None = None) -> None:
        """Initialize DataRetrieval with configuration.

        Parameters
        ----------
        config : DataRetrievalConfig
            Configuration object containing all data retrieval parameters
        """
        self.config = (config or self.default_config()).model_copy(deep=True)
        self.market = Market(market_data_generator=MarketDataGenerator())
        self.data_quality = DataQuality()
        self.logger = LoggerManager().getLogger(__name__)
        self._cache_lock = Lock()
        self._inflight_requests: dict[tuple[Any, ...], Event] = {}

        # Load API keys from environment if not provided in config
        self._populate_api_keys(self.config)

    def _populate_api_keys(self, config: DataRetrievalConfig) -> None:
        """Load API keys from environment variables if not provided in config."""
        if config.fred_api_key is None:
            config.fred_api_key = os.getenv("FRED_API_KEY")
        if config.alpha_vantage_api_key is None:
            config.alpha_vantage_api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        if config.eikon_api_key is None:
            config.eikon_api_key = os.getenv("EIKON_API_KEY")

    def _resolve_config(self, override: DataRetrievalConfig | None) -> DataRetrievalConfig:
        """Return a copy of the base config merged with an optional override."""
        if override is not None:
            resolved = override.model_copy(deep=True)
        else:
            resolved = self.config.model_copy(deep=True)
        self._populate_api_keys(resolved)
        return resolved

    @classmethod
    def default_config(cls) -> DataRetrievalConfig:
        """Return the default configuration used when none is provided."""
        return DataRetrievalConfig()

    def _create_market_data_request(
        self, config: DataRetrievalConfig, cache_algo: str = "internet_load_return"
    ) -> MarketDataRequest:
        """Create a MarketDataRequest from the configuration.

        Parameters
        ----------
        cache_algo : str, optional
            Cache algorithm to use, by default "internet_load_return"

        Returns:
        -------
        MarketDataRequest
            Configured market data request
        """
        return MarketDataRequest(
            data_source=config.data_source,
            start_date=config.start_date,
            finish_date=config.finish_date,
            tickers=config.tickers,
            category=config.category,
            freq_mult=config.freq_mult,
            freq=config.freq,
            gran_freq=config.gran_freq,
            cut=config.cut,
            fields=config.fields,
            cache_algo=cache_algo,
            vendor_tickers=config.vendor_tickers,
            vendor_fields=config.vendor_fields,
            environment=config.environment,
            trade_side=config.trade_side,
            resample=config.resample,
            resample_how=config.resample_how,
            split_request_chunks=config.split_request_chunks,
            list_threads=config.list_threads,
            fred_api_key=config.fred_api_key,
            alpha_vantage_api_key=config.alpha_vantage_api_key,
            eikon_api_key=config.eikon_api_key,
            push_to_cache=config.push_to_cache,
            overrides=config.overrides,
        )

    def load_from_cache(
        self, target_config: DataRetrievalConfig | None = None
    ) -> pd.DataFrame | None:
        """Attempt to load data from cache.

        Returns:
        -------
        Optional[pd.DataFrame]
            Cached data if available, None otherwise
        """
        config = self._resolve_config(target_config)
        return self._load_from_cache_for_config(config)

    def _load_from_cache_for_config(self, config: DataRetrievalConfig) -> pd.DataFrame | None:
        try:
            self.logger.info("Attempting to load data from cache...")
            md_request = self._create_market_data_request(config, cache_algo="cache_algo_return")

            start_time = time.time()
            df = self.market.fetch_market(md_request)
            end_time = time.time()

            if df is not None and not df.empty:
                self.logger.info(
                    f"Successfully loaded {len(df)} rows from cache in {end_time - start_time:.2f} seconds"
                )
                return df
            else:
                self.logger.warning("Cache returned empty or None data")
                return None

        except Exception as e:
            self.logger.warning(f"Failed to load from cache: {str(e)}")
            return None

    def download_data(self, target_config: DataRetrievalConfig | None = None) -> pd.DataFrame:
        """Download data from the data source.

        Returns:
        -------
        pd.DataFrame
            Downloaded market data
        """
        config = self._resolve_config(target_config)
        return self._download_data_for_config(config)

    def _download_data_for_config(self, config: DataRetrievalConfig) -> pd.DataFrame:
        try:
            self.logger.info("Downloading data from data source...")
            md_request = self._create_market_data_request(config, cache_algo="internet_load_return")

            start_time = time.time()
            df = self.market.fetch_market(md_request)
            end_time = time.time()

            if df is not None and not df.empty:
                self.logger.info(
                    f"Successfully downloaded {len(df)} rows in {end_time - start_time:.2f} seconds"
                )
                return df
            else:
                raise ValueError("Download returned empty or None data")

        except Exception as e:
            self.logger.error(f"Failed to download data: {str(e)}")
            raise

    def get_data(self, config_override: DataRetrievalConfig | None = None) -> pd.DataFrame:
        """Get data with cache-first logic.

        This method first attempts to load from cache. If that fails,
        it falls back to downloading from the data source.

        Returns:
        -------
        pd.DataFrame
            Market data from cache or download
        """
        # First try to load from cache
        effective_config = self._resolve_config(config_override)
        if effective_config.cache_max_entries is not None:
            _DATA_CACHE.configure(max_entries=effective_config.cache_max_entries)
        cache_key = _build_cache_key(effective_config)
        cached_frame = _DATA_CACHE.get(cache_key)
        if cached_frame is not None:
            self.logger.info("Returning data from in-memory cache for key %s", cache_key)
            return cached_frame

        wait_event: Event | None = None
        with self._cache_lock:
            cached_frame = _DATA_CACHE.get(cache_key)
            if cached_frame is not None:
                self.logger.info("Returning data from in-memory cache for key %s", cache_key)
                return cached_frame
            wait_event = self._inflight_requests.get(cache_key)
            if wait_event is None:
                wait_event = Event()
                self._inflight_requests[cache_key] = wait_event
                fetch_owner = True
            else:
                fetch_owner = False

        if not fetch_owner:
            wait_event.wait()
            cached_frame = _DATA_CACHE.get(cache_key)
            if cached_frame is None:
                raise RuntimeError("expected cached frame after inflight fetch")
            return cached_frame

        try:
            cached_data = self._load_from_cache_for_config(effective_config)
            if cached_data is None:
                self.logger.info("Cache load failed, attempting download from data source...")
                cached_data = self._download_data_for_config(effective_config)

            _DATA_CACHE.set(cache_key, cached_data, ttl=effective_config.cache_ttl_seconds)
            return cached_data
        finally:
            with self._cache_lock:
                event = self._inflight_requests.pop(cache_key, None)
                if event is not None:
                    event.set()

    def validate_data_quality(
        self,
        df: pd.DataFrame,
        check_duplicates: bool = True,
        check_missing: bool = True,
        check_outliers: bool = False,
    ) -> dict[str, Any]:
        """Validate data quality after loading.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        check_duplicates : bool, optional
            Whether to check for duplicated dates, by default True
        check_missing : bool, optional
            Whether to check for missing values, by default True
        check_outliers : bool, optional
            Whether to check for outliers, by default False

        Returns:
        -------
        Dict[str, Any]
            Dictionary containing validation results
        """
        if df is None or df.empty:
            return {"error": "No data to validate"}

        validation_results = self._initialize_validation_results(df)

        try:
            self._perform_validations(
                validation_results, df, check_duplicates, check_missing, check_outliers
            )
            self._calculate_completeness(validation_results, df)

            self.logger.info(
                f"Data quality validation completed. "
                f"Completeness: {validation_results['completeness']['percentage_complete']:.2f}%"
            )

        except Exception as e:
            self.logger.error(f"Error during data quality validation: {str(e)}")
            validation_results["error"] = str(e)

        return validation_results

    def _initialize_validation_results(self, df: pd.DataFrame) -> dict[str, Any]:
        """Initialize the validation results dictionary with basic info."""
        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "date_range": {
                "start": df.index.min() if hasattr(df.index, "min") else None,
                "end": df.index.max() if hasattr(df.index, "max") else None,
            },
        }

    def _perform_validations(
        self,
        validation_results: dict[str, Any],
        df: pd.DataFrame,
        check_duplicates: bool,
        check_missing: bool,
        check_outliers: bool,
    ) -> None:
        """Perform data quality validations based on flags."""
        if check_duplicates:
            self._check_duplicated_dates(df, validation_results)
        if check_missing:
            self._check_missing_values(df, validation_results)
        if check_outliers:
            self._check_outliers(df, validation_results)

    def _calculate_completeness(self, validation_results: dict[str, Any], df: pd.DataFrame) -> None:
        """Calculate data completeness percentage."""
        total_missing = validation_results.get("missing_values", {}).get("total_missing", 0)
        total_cells = len(df) * len(df.columns)
        percentage_complete = (1 - total_missing / total_cells) * 100 if total_cells > 0 else 100

        validation_results["completeness"] = {
            "percentage_complete": percentage_complete,
        }

    def _check_duplicated_dates(self, df: pd.DataFrame, validation_results: dict[str, Any]) -> None:
        """Check for duplicated dates in the dataframe."""
        self.logger.info("Checking for duplicated dates...")
        try:
            count, dups = self.data_quality.count_repeated_dates(df)
            validation_results["duplicated_dates"] = {
                "count": count if count is not None else 0,
                "duplicated_entries": dups if dups is not None else [],
            }
            if count and count > 0:
                self.logger.warning(f"Found {count} duplicated date entries")
            else:
                self.logger.info("No duplicated dates found")
        except Exception as e:
            self.logger.warning(f"Error checking for duplicated dates: {str(e)}")
            validation_results["duplicated_dates"] = {
                "count": 0,
                "duplicated_entries": [],
            }

    def _check_missing_values(self, df: pd.DataFrame, validation_results: dict[str, Any]) -> None:
        """Check for missing values in the dataframe."""
        self.logger.info("Checking for missing values...")
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        validation_results["missing_values"] = {
            "total_missing": int(total_missing),
            "by_column": missing_counts.to_dict(),
        }
        if total_missing > 0:
            self.logger.warning(f"Found {total_missing} missing values across all columns")
        else:
            self.logger.info("No missing values found")

    def _check_outliers(self, df: pd.DataFrame, validation_results: dict[str, Any]) -> None:
        """Check for outliers in numeric columns."""
        self.logger.info("Checking for outliers...")
        outlier_info = {}
        numeric_columns = df.select_dtypes(include=['number']).columns

        for col in numeric_columns:
            if col in df.columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_info[col] = {
                    "count": len(outliers),
                    "percentage": len(outliers) / len(df) * 100,
                    "bounds": {"lower": lower_bound, "upper": upper_bound},
                }

        validation_results["outliers"] = outlier_info

        total_outliers = sum(info["count"] for info in outlier_info.values())  # type: ignore[misc]
        if total_outliers > 0:
            self.logger.warning(f"Found {total_outliers} potential outliers")
        else:
            self.logger.info("No outliers detected")

    def get_data_with_validation(
        self,
        check_duplicates: bool = True,
        check_missing: bool = True,
        check_outliers: bool = False,
    ) -> dict[str, Any]:
        """Get data with cache-first logic and validate quality.

        This method retrieves data and performs quality validation.

        Parameters
        ----------
        check_duplicates : bool, optional
            Whether to check for duplicated dates, by default True
        check_missing : bool, optional
            Whether to check for missing values, by default True
        check_outliers : bool, optional
            Whether to check for outliers, by default False

        Returns:
        -------
        Dict[str, Any]
            Dictionary containing data and validation results
        """
        self.logger.info("Retrieving data with quality validation...")

        # Get the data
        df = self.get_data()

        if df is None or df.empty:
            return {
                "data": None,
                "validation_results": {"error": "No data retrieved"},
                "success": False,
            }

        # Validate data quality
        validation_results = self.validate_data_quality(
            df,
            check_duplicates=check_duplicates,
            check_missing=check_missing,
            check_outliers=check_outliers,
        )

        # Check if data quality is acceptable
        has_issues = False
        issue_messages = []

        if validation_results.get("duplicated_dates", {}).get("count", 0) > 0:
            has_issues = True
            issue_messages.append(
                f"Found {validation_results['duplicated_dates']['count']} duplicated dates"
            )

        if validation_results.get("missing_values", {}).get("total_missing", 0) > 0:
            has_issues = True
            issue_messages.append(
                f"Found {validation_results['missing_values']['total_missing']} missing values"
            )

        completeness = validation_results.get("completeness", {}).get("percentage_complete", 0)
        if completeness < 95:  # Less than 95% complete
            has_issues = True
            issue_messages.append(f"Data completeness is {completeness:.2f}% (below 95% threshold)")

        if has_issues:
            self.logger.warning(f"Data quality issues detected: {'; '.join(issue_messages)}")
            success = False
        else:
            self.logger.info("Data quality validation passed - no issues detected")
            success = True

        return {
            "data": df,
            "validation_results": validation_results,
            "success": success,
            "quality_issues": issue_messages if has_issues else [],
        }

    def update_config(self, **kwargs: Any) -> None:
        """Update configuration parameters.

        Parameters
        ----------
        **kwargs
            Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.debug(f"Updated config parameter {key} = {value}")
            else:
                self.logger.warning(f"Unknown config parameter: {key}")

    def __str__(self) -> str:
        """String representation of the DataRetrieval object."""
        return f"DataRetrieval(data_source={self.config.data_source}, tickers={self.config.tickers}, start_date={self.config.start_date})"

    def __repr__(self) -> str:
        """Detailed string representation of the DataRetrieval object."""
        return (
            f"DataRetrieval(config=DataRetrievalConfig("
            f"data_source='{self.config.data_source}', "
            f"tickers={self.config.tickers}, "
            f"fields={self.config.fields}, "
            f"start_date='{self.config.start_date}', "
            f"finish_date='{self.config.finish_date}'))"
        )


def clear_data_retrieval_cache() -> None:
    """Clear the shared in-memory DataRetrieval cache (primarily for tests)."""
    _DATA_CACHE.clear()
