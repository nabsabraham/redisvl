from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml

from redisvl.schema import (
    SchemaModel,
    FieldsModel,
    IndexModel,
    TextFieldSchema,
    NumericFieldSchema,
    TagFieldSchema,
    BaseField,
    HNSWVectorField,
    FlatVectorField,
    ExtraField,
    read_schema
)

class LCFieldsModel(FieldsModel):
    content_key: str = "content"
    content_vector_key: str = "content_vector"

    def add_content_field(self) -> None:
        if self.text is None:
            self.text = []
        for field in self.text:
            if field.name == self.content_key:
                return
        self.text.append(TextFieldSchema(name=self.content_key))

    def add_vector_field(self) -> None:
        # catch case where user inputted no vector field spec
        # in the index schema
        if self.vector is None:
            self.vector = []

        for field in self.vector:
            if field.name == self.content_vector_key:
                return
        # use the default
        self.vector.append(FlatVectorField(name=self.content_vector_key))

    def as_dict(self) -> Dict[str, List[Any]]:
        schemas: Dict[str, List[Any]] = {"text": [], "tag": [], "numeric": []}
        # iter over all class attributes
        for attr, attr_value in self.__dict__.items():
            # only non-empty lists
            if isinstance(attr_value, list) and len(attr_value) > 0:
                field_values: List[Dict[str, Any]] = []
                # iterate over all fields in each category (tag, text, etc)
                for val in attr_value:
                    value: Dict[str, Any] = {}
                    # iterate over values within each field to extract
                    # settings for that field (i.e. name, weight, etc)
                    for field, field_value in val.__dict__.items():
                        # make enums into strings
                        if isinstance(field_value, Enum):
                            value[field] = field_value.value
                        # don't write null values
                        elif field_value is not None:
                            value[field] = field_value
                    field_values.append(value)

                schemas[attr] = field_values

        schema: Dict[str, List[Any]] = {}
        # only write non-empty lists from defaults
        for k, v in schemas.items():
            if len(v) > 0:
                schema[k] = v
        return schema

    @property
    def content_vector(self) -> Union[FlatVectorField, HNSWVectorField]:
        if not self.vector:
            raise ValueError("No vector fields found")
        for field in self.vector:
            if field.name == self.content_vector_key:
                return field
        raise ValueError("No content_vector field found")

    @property
    def vector_dtype(self) -> np.dtype:
        # should only ever be called after pydantic has validated the schema
        return REDIS_VECTOR_DTYPE_MAP[self.content_vector.datatype]


    def set_content_vector_dim(self, dims) -> None:
        self.content_vector.dims = dims

    def get_fields(self) -> List["RedisField"]:
        redis_fields: List["RedisField"] = []
        if self.is_empty:
            return redis_fields

        for field_name in self.__fields__.keys():
            if field_name not in ["content_key", "content_vector_key", "extra"]:
                field_group = getattr(self, field_name)
                if field_group is not None:
                    for field in field_group:
                        redis_fields.append(field.as_field())
        return redis_fields

    @property
    def metadata_keys(self) -> List[str]:
        keys: List[str] = []
        if self.is_empty:
            return keys

        for field_name in self.__fields__.keys():
            field_group = getattr(self, field_name)
            if field_group is not None:
                for field in field_group:
                    # check if it's a metadata field. exclude vector and content key
                    if not isinstance(field, str) and field.name not in [
                        self.content_key,
                        self.content_vector_key,
                    ]:
                        keys.append(field.name)
        return keys

class LCSchemaModel(SchemaModel):
    fields: LCFieldsModel


def _get_schema_with_lc_defaults(
    schema: Optional[Union[Dict[str, str], str, os.PathLike]] = None,
    **kwargs,
    ) -> "LCSchemaModel":

    if schema:
        schema_dict = read_schema(schema)
        lc_schema = LCSchemaModel(**schema_dict)
    else:
        name = kwargs.get("index_name", uuid.uuid4().hex) # TODO unify this
        key_prefix = kwargs.get("key_prefix", f"doc:{index_name}")
        storage_type = kwargs.get("storage_type", "HASH")
        key_separator = kwargs.get("key_separator", ":")

        index_schema = IndexModel(
            name=name,
            prefix=key_prefix,
            key_separator=key_separator,
            storage_type=storage_type,
        )
        fields = LCFieldsModel()
        lc_schema = LCSchemaModel(index=index_schema, fields=fields)

    # ensure content field is present
    lc_schema.fields.add_content_field()

    # ensure vector field is present
    lc_schema.fields.add_vector_field(vector_field)
    return lc_schema