import logging
import re
from app.config.settings import settings

logger = logging.getLogger(__name__)

# Sentence boundary regex (splits on ., !, ?, ؟, \n while preserving trailing punctuation)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?؟\n])\s+")

# Natural pause boundary regex (splits on commas, semicolons: , , ،, ;, ؛)
_PAUSE_SPLIT_RE = re.compile(r"(?<=[,،;؛])\s+")

# Conjunction split regex (splits before/after conjunctions like "and", "و", "or", "but")
_CONJUNCTION_SPLIT_RE = re.compile(r"\s+(?=(?:and|or|but|&|و|أو|ولكن)\s+)", re.IGNORECASE)

# Regexp to match atomic number+unit tokens (e.g., "250 Egyptian Pounds", "250 EGP", "250 جنيه مصري", "500 USD")
_ATOMIC_NUMBER_UNIT_RE = re.compile(
    r"\b\d+(?:,\d+)*(?:\.\d+)?(?:\s*(?:Egyptian\s+Pounds|EGP|USD|EUR|LE|L\.E\.|جنيه\s+مصري|جنيه|دولار|يورو|%|months|شهر|أشهر))?\b",
    re.IGNORECASE,
)


class SpeechChunker:
    """
    Splits speech-formatted text into manageable chunks suitable for TTS synthesis providers.

    Responsibilities:
    - Configurable maximum chunk length (defaults to settings.TTS_MAX_CHUNK_LENGTH).
    - Splitting hierarchy:
        1. Sentence boundaries (. ! ? ؟ \n)
        2. Natural pauses (, ، ; ؛)
        3. Conjunctions (and, و, etc.)
        4. Safe word boundary packing (preserves atomic units like "250 Egyptian Pounds")
    - Generic, domain-agnostic implementation.
    """

    def __init__(self, max_chunk_length: int | None = None) -> None:
        self.max_chunk_length = (
            max_chunk_length if max_chunk_length is not None else settings.TTS_MAX_CHUNK_LENGTH
        )

    def split(self, text: str) -> list[str]:
        """
        Split *text* into a list of strings where each string is <= self.max_chunk_length.
        """
        if not text or not text.strip():
            return []

        clean_text = text.strip()
        if len(clean_text) <= self.max_chunk_length:
            self._log_chunks(clean_text, [clean_text])
            return [clean_text]

        # Step 1: Split into sentences
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(clean_text) if s.strip()]
        chunks: list[str] = []
        current_acc: list[str] = []

        for sent in sentences:
            if len(sent) <= self.max_chunk_length:
                test_str = " ".join(current_acc + [sent]).strip()
                if len(test_str) <= self.max_chunk_length:
                    current_acc.append(sent)
                else:
                    if current_acc:
                        chunks.append(" ".join(current_acc).strip())
                    current_acc = [sent]
            else:
                if current_acc:
                    chunks.append(" ".join(current_acc).strip())
                    current_acc = []

                sub_chunks = self._split_long_sentence(sent)
                chunks.extend(sub_chunks)

        if current_acc:
            chunks.append(" ".join(current_acc).strip())

        final_chunks = [c.strip() for c in chunks if c and c.strip()]
        self._log_chunks(clean_text, final_chunks)
        return final_chunks

    def _split_long_sentence(self, sentence: str) -> list[str]:
        """Sub-splits a single sentence exceeding max_chunk_length."""
        # Step 2: Try splitting on natural pauses (, ، ; ؛)
        pauses = [p.strip() for p in _PAUSE_SPLIT_RE.split(sentence) if p.strip()]
        if len(pauses) > 1:
            packed = self._pack_segments(pauses)
            if all(len(c) <= self.max_chunk_length for c in packed):
                return packed

        # Step 3: Try splitting on conjunctions
        conj_segments = [c.strip() for c in _CONJUNCTION_SPLIT_RE.split(sentence) if c.strip()]
        if len(conj_segments) > 1:
            packed = self._pack_segments(conj_segments)
            if all(len(c) <= self.max_chunk_length for c in packed):
                return packed

        # Step 4: Fallback to safe word boundary packing
        return self._pack_safe_words(sentence)

    def _pack_segments(self, segments: list[str]) -> list[str]:
        """Packs text segments into chunks <= max_chunk_length."""
        chunks = []
        current: list[str] = []

        for seg in segments:
            test_str = " ".join(current + [seg]).strip()
            if len(test_str) <= self.max_chunk_length:
                current.append(seg)
            else:
                if current:
                    chunks.append(" ".join(current).strip())
                if len(seg) <= self.max_chunk_length:
                    current = [seg]
                else:
                    word_chunks = self._pack_safe_words(seg)
                    chunks.extend(word_chunks)
                    current = []
        if current:
            chunks.append(" ".join(current).strip())
        return chunks

    def _pack_safe_words(self, text: str) -> list[str]:
        """
        Splits text by words while preserving atomic multi-token units
        (e.g., '250 Egyptian Pounds' or '250 جنيه مصري').
        """
        tokens = self._tokenize_atomic_units(text)
        chunks = []
        current: list[str] = []

        for token in tokens:
            test_str = " ".join(current + [token]).strip()
            if len(test_str) <= self.max_chunk_length:
                current.append(token)
            else:
                if current:
                    chunks.append(" ".join(current).strip())
                current = [token]

        if current:
            chunks.append(" ".join(current).strip())

        return chunks

    @staticmethod
    def _tokenize_atomic_units(text: str) -> list[str]:
        """Tokenizes text into words and atomic number+unit phrases."""
        spans = []
        for match in _ATOMIC_NUMBER_UNIT_RE.finditer(text):
            spans.append((match.start(), match.end(), match.group(0)))

        if not spans:
            return text.split()

        tokens = []
        last_end = 0
        for start, end, match_str in spans:
            if start > last_end:
                prefix_words = text[last_end:start].split()
                tokens.extend(prefix_words)
            tokens.append(match_str)
            last_end = end

        if last_end < len(text):
            suffix_words = text[last_end:].split()
            tokens.extend(suffix_words)

        return [t for t in tokens if t.strip()]

    def _log_chunks(self, original_text: str, chunks: list[str]) -> None:
        logger.info(
            "Speech Chunking Summary:\n"
            "  Original length: %d chars\n"
            "  Chunks created:  %d",
            len(original_text),
            len(chunks),
        )
        for idx, chunk in enumerate(chunks, start=1):
            logger.debug("  Chunk %d (%d chars): %r", idx, len(chunk), chunk)
