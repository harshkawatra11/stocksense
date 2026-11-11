from __future__ import annotations


class StockSenseError(Exception):
    """Base class for all StockSense errors."""


class DataNotAvailable(StockSenseError):
    """Expected absence of data (e.g. a file not yet published)."""


class DataValidationError(StockSenseError):
    """Data was present but failed validation."""


class BrokerError(StockSenseError):
    """A broker adapter failed."""


class ConfigError(StockSenseError):
    """Configuration or settings are missing or malformed."""


class LookaheadError(StockSenseError):
    """A computation used information from the future."""


class LockboxError(StockSenseError):
    """The lockbox period was touched illegally."""
