from typing import Generator, TypeVar

T = TypeVar("T")


def chunk_items(items: list[T], chunk_size: int) -> Generator[list[T], None, None]:
    """
    Yield successive chunks of size `chunk_size` from `items`.

    Args:
        items: List of elements to split.
        chunk_size: Maximum size of each chunk. Must be > 0.

    Yields:
        Sublists of length up to `chunk_size`.

    Raises:
        ValueError: If chunk_size is less than or equal to 0.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]
