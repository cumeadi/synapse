class SynapseError(Exception):
    """Base exception for all Synapse errors."""
    pass

class AuthenticationError(SynapseError):
    """Raised when an API key is invalid or missing."""
    pass

class NamespaceNotFoundError(SynapseError):
    """Raised when a specified namespace does not exist."""
    pass

class APIError(SynapseError):
    """Raised when the backend API returns an error response."""
    def __init__(self, message: str, status_code: int = None, details: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details
