import asyncio
import json
from pathlib import Path

from app.knowledge.models.raw_document import RawDocument


class JsonKnowledgeLoader:
    """
    Loads raw JSON knowledge files from disk.

    Responsibilities:
    - Read one file or all JSON files in a directory.
    - Parse into RawDocument via Pydantic validation.
    - Inject source_path after parsing.

    No business logic. No normalization. No chunking.
    """

    async def load_file(self, path: Path) -> RawDocument:
        """Load and parse a single JSON file into a RawDocument."""
        raw_text = await asyncio.to_thread(path.read_text, encoding="utf-8")
        data = json.loads(raw_text)
        # source_path is not in the JSON; inject it before validation.
        data["source_path"] = path
        return RawDocument.model_validate(data)

    async def load_directory(self, directory: Path) -> list[RawDocument]:
        """Load all *.json files in a directory concurrently."""
        paths = sorted(directory.glob("*.json"))
        tasks = [self.load_file(path) for path in paths]
        return list(await asyncio.gather(*tasks))
