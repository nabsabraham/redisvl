from typing import Dict, Any, Optional, List

from redisvl.schema import IndexSchema
from redisvl.schema.fields import BaseVectorField


class LlamaIndexSchema(IndexSchema):
    """RedisVL index schema for working with LlamaIndex."""

    # User should not be able to change these for the default LlamaIndex
    key_separator: str = "_"
    storage_type: str = "hash"
    id_field_name: str = "id"
    doc_id_field_name: str = "doc_id"
    text_field_name: str = "text"
    vector_field_name: str = "vector"
    default_vector_field_args: Dict[str, Any] = {
        "dims": 1536,
        "algorithm": "FLAT",
        "datatype": "FLOAT32",
        "distance_metric": "COSINE",
        "initial_cap": None,
        "block_size": None,
        "m": 16,
        "ef_construction": 200,
        "ef_runtime": 10,
        "epsilon": 0.8,
    }

    def __init__(
        self,
        name: str,
        prefix: str,
        vector_field_name: str,
        metadata_fields: List[str],
        **kwargs,
    ):
        # Construct the base base index schema
        super().__init__(name=name, prefix=prefix, **kwargs)

        self.vector_field_name = vector_field_name

        # Add llama index fields
        self.add_field("tag", name=self.id_field_name, sortable=False)
        self.add_field("tag", name=self.doc_id_field_name, sortable=False)
        self.add_field("text", name=self.text_field_name, weight=1.0)

        # Add user-specified metadata fields
        for metadata_field in metadata_fields:
            # TODO: allow addition of text fields as metadata?
            self.add_field("tag", name=metadata_field, sortable=False)

        class Config:
            # Ignore extra fields passed in kwargs
            ignore_extra = True
