class WordChunker:
    def __init__(
        self,
        chunk_size=100,
        overlap=0
    ):
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
        ):
            raise ValueError(
                "chunk_size must be an integer"
            )

        if (
            isinstance(overlap, bool)
            or not isinstance(overlap, int)
        ):
            raise ValueError(
                "overlap must be an integer"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

        if self.chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero"
            )

        if self.overlap < 0:
            raise ValueError(
                "overlap must not be negative"
            )

        if self.overlap >= self.chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size"
            )

    def chunk(
        self,
        text,
        document_id="doc"
    ):
        words = text.split()

        if not words:
            return []

        step = (
            self.chunk_size
            - self.overlap
        )

        chunks = []

        for start in range(
            0,
            len(words),
            step
        ):
            end = start + self.chunk_size

            chunk_words = words[
                start:end
            ]

            if not chunk_words:
                break

            chunk_index = len(chunks)

            chunks.append(
                {
                    "chunk_id": (
                        f"{document_id}-"
                        f"{chunk_index:03d}"
                    ),
                    "text": " ".join(
                        chunk_words
                    ),
                    "start_word": start,
                    "end_word": (
                        start
                        + len(chunk_words)
                    )
                }
            )

            if end >= len(words):
                break

        return chunks
