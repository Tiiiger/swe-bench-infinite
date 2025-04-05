"""
Custom exceptions for SWE-Bench Infinite.
"""

class SWEBenchError(Exception):
    """Base exception class for all SWE-Bench errors."""
    pass

class GitError(SWEBenchError):
    """Exception raised for git-related errors."""
    pass

class AnthropicResponseError(SWEBenchError):
    """Exception raised when there's an issue with Anthropic API response."""
    pass

class DockerBuildError(SWEBenchError):
    """Exception raised when Docker build fails."""
    pass

class RequirementsError(SWEBenchError):
    """Exception raised when there's an issue with requirements processing."""
    pass 