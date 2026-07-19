from .chunk_store import ChunkStore, JsonlChunkStore
from .parsers import OcrParser, Parser


def get_chunk_store() -> ChunkStore:
    return JsonlChunkStore()


def get_parser() -> Parser:
    return OcrParser(lang="tur", chunk_store=get_chunk_store())
