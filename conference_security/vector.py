import json
from sqlalchemy.types import UserDefinedType


class VectorEmbedding(UserDefinedType):
    """Stores native pgvector values in PostgreSQL; SQLite uses the JSON variant."""
    cache_ok = True

    def get_col_spec(self, **_: object) -> str:
        return "vector"

    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return None
            return "[" + ",".join(str(float(item)) for item in value) + "]"
        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None or isinstance(value, list):
                return value
            return json.loads(value)
        return process