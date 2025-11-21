"""Basic integration tests ensuring the package loads in the runtime environment."""


def test_backtester_package_is_importable() -> None:
    """Sanity check to keep the integration test suite non-empty."""
    import backtester  # noqa: F401
