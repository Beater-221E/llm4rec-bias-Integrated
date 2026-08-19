"""Typed exceptions used across the lab."""


class LabError(Exception):
    """Base error for llm4rec-bias-Integrated."""


class ConfigurationError(LabError):
    """Invalid or incomplete experiment configuration."""


class DatasetValidationError(LabError):
    """Dataset split / leakage / schema validation failed."""


class CheckpointError(LabError):
    """Missing, incompatible, or corrupt checkpoint."""


class InvalidGenerationError(LabError):
    """Model output failed the shared strict parser."""


class MissingArtifactError(LabError):
    """Expected run artifact (data, config, predictions) is missing."""
