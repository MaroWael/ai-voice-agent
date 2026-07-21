class VectorStoreError(Exception):
    """Base exception for all vector store operations."""

    pass


class VectorStoreConnectionError(VectorStoreError):
    """Raised when the vector store service is unreachable or encounters network failure."""

    pass


class CollectionError(VectorStoreError):
    """Raised when collection creation or verification fails."""

    pass


class BatchUploadError(VectorStoreError):
    """Raised when point upsert operations fail."""

    pass
