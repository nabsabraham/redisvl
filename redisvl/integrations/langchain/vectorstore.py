

from langchain.schema.vectorstore import VectorStore, VectorStoreRetriever


from __future__ import annotations

import logging
import os
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

import numpy as np
import yaml

from langchain._api import deprecated
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.docstore.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore, VectorStoreRetriever
from langchain.utilities.redis import (
    _array_to_buffer,
    _buffer_to_array,
    check_redis_module_exist,
    get_client,
)
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores.redis.constants import (
    REDIS_REQUIRED_MODULES,
    REDIS_TAG_SEPARATOR,
)
from langchain.vectorstores.utils import maximal_marginal_relevance

from redisvl.index import SearchIndex
from redisvl.query import RangeQuery, VectorQuery
from redisvl.schema import IndexModel, SchemaModel
from redisvl.integrations.langchain.schema import (
    _get_schema_with_langchain_defaults,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from redis.client import Redis as RedisType
    from redis.commands.search.query import Query

    from langchain.vectorstores.redis.filters import RedisFilterExpression
    from langchain.vectorstores.redis.schema import RedisModel


def _default_relevance_score(val: float) -> float:
    return 1 - val

def connect_from_env_or_kwargs(kwargs) -> SearchIndex:
    """Connect to a Redis index from kwargs or environment variables.

    Args:
        kwargs (Dict[str, Any]): Keyword arguments.

    Returns:
        SearchIndex: Redis search index.
    """
    redis_url = get_from_dict_or_env(kwargs, "redis_url", "REDIS_URL")
    try:
        # TODO use importlib to check if redis is installed
        import redis  # noqa: F401

    except ImportError as e:
        raise ImportError(
            "Could not import redis python package. "
            "Please install it with `pip install redis`."
        ) from e

    try:
        # We need to first remove redis_url from kwargs,
        # otherwise passing it to Redis will result in an error.
        if "redis_url" in kwargs:
            kwargs.pop("redis_url")
        index_name = get_from_dict_or_env(kwargs, "index_name", "REDIS_INDEX_NAME")
        index = SearchIndex(index_name).connect(url=redis_url, **kwargs)
    except ValueError as e:
        raise ValueError(f"Your redis connected error: {e}")
    return index


class Redis(VectorStore):

    def __init__(
        self,
        redis_url: str,
        embedding: Embeddings,
        index_name: Optional[str],
        schema: Optional[Union[Dict[str, str], str, os.PathLike]] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        **kwargs: Any,
    ):
        """Initialize with necessary components."""
        self._embeddings = embedding
        self.relevance_score_fn = relevance_score_fn
        self._schema = _get_schema_with_lc_defaults(schema, **kwargs)

        try:
            # TODO remove all kwargs not used for redis
            self._index = SearchIndex(self._schema, url=redis_url, **kwargs)
        except ValueError as e:
            raise ValueError(f"Redis failed to connect: {e}")


    @classmethod
    def from_texts_return_keys(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        index_name: Optional[str] = None,
        schema: Optional[Union[Dict[str, str], str, os.PathLike]] = None,
        **kwargs: Any,
    ) -> Tuple[Redis, List[str]]:

        redis_url = get_from_dict_or_env(kwargs, "redis_url", "REDIS_URL")

        if "redis_url" in kwargs:
            kwargs.pop("redis_url")

        # see if the user specified keys
        keys = None
        if "keys" in kwargs:
            keys = kwargs.pop("keys")

        # type check for metadata
        if metadatas:
            if isinstance(metadatas, list) and len(metadatas) != len(texts):  # type: ignore  # noqa: E501
                raise ValueError("Number of metadatas must match number of texts")
            if not (isinstance(metadatas, list) and isinstance(metadatas[0], dict)):
                raise ValueError("Metadatas must be a list of dicts")

        # Create instance
        instance = cls(
            redis_url,
            embedding,
            index_name,
            schema=schema
            **kwargs,
        )

        # TODO figure out what to do with indexed metadata

        # Add data to Redis
        keys = instance.add_texts(texts, metadatas, keys=keys)
        return instance, keys

    @classmethod
    def from_texts(
        cls: Type[Redis],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        index_name: Optional[str] = None,
        schema: Optional[Union[Dict[str, str], str, os.PathLike]] = None,
        **kwargs: Any,
    ) -> Redis:

        instance, _ = cls.from_texts_return_keys(
            texts,
            embedding,
            metadatas=metadatas,
            index_name=index_name,
            schema=schema,
            **kwargs,
        )
        return instance

    @classmethod
    def from_existing_index(
        cls,
        embedding: Embeddings,
        index_name: str,
        schema: Union[Dict[str, str], str, os.PathLike],
        **kwargs: Any,
    ) -> Redis:

        redis_url = get_from_dict_or_env(kwargs, "redis_url", "REDIS_URL")
        if "redis_url" in kwargs:
            kwargs.pop("redis_url")
        try:
            instance = cls(
                redis_url,
                index_name,
                embedding,
                schema=schema,
                **kwargs,
            )
            assert(instance._index.exists()), f"Index {index_name} does not exist"
        except Exception as e:
            raise ValueError(f"Redis failed to connect: {e}")


    @staticmethod
    def delete(
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> bool:
        """
        Delete a Redis entry.

        Args:
            ids: List of ids (keys in redis) to delete.
            redis_url: Redis connection url. This should be passed in the kwargs
                or set as an environment variable: REDIS_URL.

        Returns:
            bool: Whether or not the deletions were successful.

        Raises:
            ValueError: If the redis python package is not installed.
            ValueError: If the ids (keys in redis) are not provided
        """
        redis_url = get_from_dict_or_env(kwargs, "redis_url", "REDIS_URL")

        if ids is None:
            raise ValueError("'ids' (keys)() were not provided.")


        try:
            # We need to first remove redis_url from kwargs,
            # otherwise passing it to Redis will result in an error.
            if "redis_url" in kwargs:
                kwargs.pop("redis_url")
            client = get_client(redis_url=redis_url, **kwargs)
        except ValueError as e:
            raise ValueError(f"Your redis connected error: {e}")
        # Check if index exists
        try:
            client.delete(*ids)
            logger.info("Entries deleted")
            return True
        except:  # noqa: E722
            # ids does not exist
            return False

    @staticmethod
    def drop_index(
        index_name: str,
        delete_documents: bool,
        **kwargs: Any,
    ) -> bool:
        """
        Drop a Redis search index.

        Args:
            index_name (str): Name of the index to drop.
            delete_documents (bool): Whether to drop the associated documents.

        Returns:
            bool: Whether or not the drop was successful.
        """

        # Check if index exists
        try:
            client.ft(index_name).dropindex(delete_documents)
            logger.info("Drop index")
            return True
        except:  # noqa: E722
            # Index not exist
            return False

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        embeddings: Optional[List[List[float]]] = None,
        batch_size: int = 1000,
        clean_metadata: bool = True,
        **kwargs: Any,
    ) -> List[str]:
        """Add more texts to the vectorstore.

        Args:
            texts (Iterable[str]): Iterable of strings/text to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
                Defaults to None.
            embeddings (Optional[List[List[float]]], optional): Optional pre-generated
                embeddings. Defaults to None.
            keys (List[str]) or ids (List[str]): Identifiers of entries.
                Defaults to None.
            batch_size (int, optional): Batch size to use for writes. Defaults to 1000.

        Returns:
            List[str]: List of ids added to the vectorstore
        """

        # Get keys or ids from kwargs
        # Other vectorstores use ids
        keys_or_ids = kwargs.get("keys", kwargs.get("ids"))

        # type check for metadata
        if metadatas:
            if isinstance(metadatas, list) and len(metadatas) != len(texts):  # type: ignore  # noqa: E501
                raise ValueError("Number of metadatas must match number of texts")
            if not (isinstance(metadatas, list) and isinstance(metadatas[0], dict)):
                raise ValueError("Metadatas must be a list of dicts")

        # set the vector dimensions
        self._index.schema.set_content_vector_dims


        embeddings = embeddings or self._embeddings.embed_documents(list(texts))
        # Write data to redis
        records = []
        keys = []
        for i, text in enumerate(texts):
            # Use provided values by default or fallback
            key = keys_or_ids[i] if keys_or_ids else str(uuid.uuid4().hex)
            if not key.startswith(self.key_prefix + ":"):
                key = self.key_prefix + ":" + key
            metadata = metadatas[i] if metadatas else {}
            metadata = _prepare_metadata(metadata) if clean_metadata else metadata
            mapping={
                self._schema.content_key: text,
                self._schema.content_vector_key: _array_to_buffer(
                    embeddings[i], self._schema.vector_dtype
                ),
                **metadata,
            }
            records.append(mapping)

        # index created during load if it does not exist.
        self._index.upsert(records, keys=keys)
        return keys

    def as_retriever(self, **kwargs: Any) -> RedisVectorStoreRetriever:
        tags = kwargs.pop("tags", None) or []
        tags.extend(self._get_retriever_tags())
        return RedisVectorStoreRetriever(vectorstore=self, **kwargs, tags=tags)


    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[RedisFilterExpression] = None,
        return_metadata: bool = True,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with **vector distance**.

        The "scores" returned from this function are the raw vector
        distances from the query vector. For similarity scores, use
        ``similarity_search_with_relevance_scores``.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.
            filter (RedisFilterExpression, optional): Optional metadata filter.
                Defaults to None.
            return_metadata (bool, optional): Whether to return metadata.
                Defaults to True.

        Returns:
            List[Tuple[Document, float]]: A list of documents that are
                most similar to the query with the distance for each document.
        """

        query_embedding = self._embeddings.embed_query(query)
        redis_query = self._prepare_query(
            embedding,
            k=k,
            filter=filter,
            distance_threshold=distance_threshold,
            with_distance=False,
        )

        results = self._index.query(redis_query)

        # Prepare document results
        docs_with_scores: List[Tuple[Document, float]] = []
        for result in results.docs:
            metadata = {"id": result.id}
            metadata.update(self._collect_metadata(result))

            doc = Document(page_content=result.content, metadata=metadata)
            distance = self._calculate_fp_distance(result.distance)
            docs_with_scores.append((doc, distance))

        return docs_with_scores

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[RedisFilterExpression] = None,
        distance_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.
            filter (RedisFilterExpression, optional): Optional metadata filter.
                Defaults to None.
            distance_threshold (Optional[float], optional): Maximum vector distance
                between selected documents and the query vector. Defaults to None.

        Returns:
            List[Document]: A list of documents that are most similar to the query
                text.
        """
        query_embedding = self._embeddings.embed_query(query)
        return self.similarity_search_by_vector(
            query_embedding,
            k=k,
            filter=filter,
            distance_threshold=distance_threshold,
            **kwargs,
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[RedisFilterExpression] = None,
        distance_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search between a query vector and the indexed vectors.

        Args:
            embedding (List[float]): The query vector for which to find similar
                documents.
            k (int): The number of documents to return. Default is 4.
            filter (RedisFilterExpression, optional): Optional metadata filter.
                Defaults to None.
            distance_threshold (Optional[float], optional): Maximum vector distance
                between selected documents and the query vector. Defaults to None.

        Returns:
            List[Document]: A list of documents that are most similar to the query
                text.
        """

        redis_query = self._prepare_query(
            embedding,
            k=k,
            filter=filter,
            distance_threshold=distance_threshold,
            with_metadata=return_metadata,
            with_distance=False,
        )

        results = self._index.query(redis_query)

        # Prepare document results
        docs = []
        for result in results.docs:
            metadata = {}
            metadata = {"id": result.id}
            metadata.update(self._collect_metadata(result))

            content_key = self._schema.content_key
            docs.append(
                Document(page_content=getattr(result, content_key), metadata=metadata)
            )
        return docs

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[RedisFilterExpression] = None,
        return_metadata: bool = True,
        distance_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
            among selected documents.

        Args:
            query (str): Text to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult (float): Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            filter (RedisFilterExpression, optional): Optional metadata filter.
                Defaults to None.
            return_metadata (bool, optional): Whether to return metadata.
                Defaults to True.
            distance_threshold (Optional[float], optional): Maximum vector distance
                between selected documents and the query vector. Defaults to None.

        Returns:
            List[Document]: A list of Documents selected by maximal marginal relevance.
        """
        # Embed the query
        query_embedding = self._embeddings.embed_query(query)

        # Fetch the initial documents
        prefetch_docs = self.similarity_search_by_vector(
            query_embedding,
            k=fetch_k,
            filter=filter,
            return_metadata=return_metadata,
            distance_threshold=distance_threshold,
            **kwargs,
        )
        prefetch_ids = [doc.metadata["id"] for doc in prefetch_docs]

        # Get the embeddings for the fetched documents
        prefetch_embeddings = [
            _buffer_to_array(
                cast(
                    bytes,
                    self.client.hget(prefetch_id, self._schema.content_vector_key),
                ),
                dtype=self._schema.vector_dtype,
            )
            for prefetch_id in prefetch_ids
        ]

        # Select documents using maximal marginal relevance
        selected_indices = maximal_marginal_relevance(
            np.array(query_embedding), prefetch_embeddings, lambda_mult=lambda_mult, k=k
        )
        selected_docs = [prefetch_docs[i] for i in selected_indices]

        return selected_docs

    def _collect_metadata(self, result: "Document") -> Dict[str, Any]:
        """Collect metadata from Redis.

        Method ensures that there isn't a mismatch between the metadata
        and the index schema passed to this class by the user or generated
        by this class.

        Args:
            result (Document): redis.commands.search.Document object returned
                from Redis.

        Returns:
            Dict[str, Any]: Collected metadata.
        """
        # new metadata dict as modified by this method
        meta = {}
        for key in self._schema.metadata_keys:
            try:
                meta[key] = getattr(result, key)
            except AttributeError:
                # warning about attribute missing
                logger.warning(
                    f"Metadata key {key} not found in metadata. "
                    + "Setting to None. \n"
                    + "Metadata fields defined for this instance: "
                    + f"{self._schema.metadata_keys}"
                )
                meta[key] = None
        return meta

    def _prepare_query(
        self,
        query_embedding: List[float],
        k: int = 4,
        filter: Optional[RedisFilterExpression] = None,
        distance_threshold: Optional[float] = None,
        with_metadata: bool = True,
        with_distance: bool = False,
    ) -> Tuple["Query", Dict[str, Any]]:

        # prepare return fields including score
        return_fields = [self._schema.content_key]
        if with_metadata:
            return_fields.extend(self._schema.metadata_keys)

        if distance_threshold:
            return RangeQuery(
                query_embedding,
                vector_field_name=self._schema.content_vector_key,
                return_fields=return_fields,
                num_results=k,
                filter_expression=filter,
                dtype=self._schema.vector_dtype,
                distance_threshold=distance_threshold,
                return_score=with_distance,
                )
        return VectorQuery(
            query_embedding,
            vector_field_name=self._schema.content_vector_key,
            return_fields=return_fields,
            num_results=k,
            filter_expression=filter,
            dtype=self._schema.vector_dtype,
            return_score=with_distance,
        )



    def _calculate_fp_distance(self, distance: str) -> float:
        """Calculate the distance based on the vector datatype

        Two datatypes supported:
        - FLOAT32
        - FLOAT64

        if it's FLOAT32, we need to round the distance to 4 decimal places
        otherwise, round to 7 decimal places.
        """
        if self._schema.content_vector.datatype == "FLOAT32":
            return round(float(distance), 4)
        return round(float(distance), 7)

    def _check_deprecated_kwargs(self, kwargs: Mapping[str, Any]) -> None:
        """Check for deprecated kwargs."""

        deprecated_kwargs = {
            "redis_host": "redis_url",
            "redis_port": "redis_url",
            "redis_password": "redis_url",
            "content_key": "index_schema",
            "vector_key": "vector_schema",
            "distance_metric": "vector_schema",
        }
        for key, value in kwargs.items():
            if key in deprecated_kwargs:
                raise ValueError(
                    f"Keyword argument '{key}' is deprecated. "
                    f"Please use '{deprecated_kwargs[key]}' instead."
                )

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        if self.relevance_score_fn:
            return self.relevance_score_fn

        metric_map = {
            "COSINE": self._cosine_relevance_score_fn,
            "IP": self._max_inner_product_relevance_score_fn,
            "L2": self._euclidean_relevance_score_fn,
        }
        try:
            return metric_map[self._schema.content_vector.distance_metric]
        except KeyError:
            return _default_relevance_score


def _generate_field_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a schema for the search index in Redis based on the input metadata.

    Given a dictionary of metadata, this function categorizes each metadata
        field into one of the three categories:
    - text: The field contains textual data.
    - numeric: The field contains numeric data (either integer or float).
    - tag: The field contains list of tags (strings).

    Args
        data (Dict[str, Any]): A dictionary where keys are metadata field names
            and values are the metadata values.

    Returns:
        Dict[str, Any]: A dictionary with three keys "text", "numeric", and "tag".
            Each key maps to a list of fields that belong to that category.

    Raises:
        ValueError: If a metadata field cannot be categorized into any of
            the three known types.
    """
    result: Dict[str, Any] = {
        "text": [],
        "numeric": [],
        "tag": [],
    }

    for key, value in data.items():
        # Numeric fields
        try:
            int(value)
            result["numeric"].append({"name": key})
            continue
        except (ValueError, TypeError):
            pass

        # None values are not indexed as of now
        if value is None:
            continue

        # if it's a list of strings, we assume it's a tag
        if isinstance(value, (list, tuple)):
            if not value or isinstance(value[0], str):
                result["tag"].append({"name": key})
            else:
                name = type(value[0]).__name__
                raise ValueError(
                    f"List/tuple values should contain strings: '{key}': {name}"
                )
            continue

        # Check if value is string before processing further
        if isinstance(value, str):
            result["text"].append({"name": key})
            continue

        # Unable to classify the field value
        name = type(value).__name__
        raise ValueError(
            "Could not generate Redis index field type mapping "
            + f"for metadata: '{key}': {name}"
        )

    return result


def _prepare_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare metadata for indexing in Redis by sanitizing its values.

    - String, integer, and float values remain unchanged.
    - None or empty values are replaced with empty strings.
    - Lists/tuples of strings are joined into a single string with a comma separator.

    Args:
        metadata (Dict[str, Any]): A dictionary where keys are metadata
            field names and values are the metadata values.

    Returns:
        Dict[str, Any]: A sanitized dictionary ready for indexing in Redis.

    Raises:
        ValueError: If any metadata value is not one of the known
            types (string, int, float, or list of strings).
    """

    def raise_error(key: str, value: Any) -> None:
        raise ValueError(
            f"Metadata value for key '{key}' must be a string, int, "
            + f"float, or list of strings. Got {type(value).__name__}"
        )

    clean_meta: Dict[str, Union[str, float, int]] = {}
    for key, value in metadata.items():
        if not value:
            clean_meta[key] = ""
            continue

        # No transformation needed
        if isinstance(value, (str, int, float)):
            clean_meta[key] = value

        # if it's a list/tuple of strings, we join it
        elif isinstance(value, (list, tuple)):
            if not value or isinstance(value[0], str):
                clean_meta[key] = REDIS_TAG_SEPARATOR.join(value)
            else:
                raise_error(key, value)
        else:
            raise_error(key, value)
    return clean_meta


class RedisVectorStoreRetriever(VectorStoreRetriever):
    """Retriever for Redis VectorStore."""

    vectorstore: Redis
    """Redis VectorStore."""
    search_type: str = "similarity"
    """Type of search to perform. Can be either
    'similarity',
    'similarity_distance_threshold',
    'similarity_score_threshold'
    """

    search_kwargs: Dict[str, Any] = {
        "k": 4,
        "score_threshold": 0.9,
        # set to None to avoid distance used in score_threshold search
        "distance_threshold": None,
    }
    """Default search kwargs."""

    allowed_search_types = [
        "similarity",
        "similarity_distance_threshold",
        "similarity_score_threshold",
        "mmr",
    ]
    """Allowed search types."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(query, **self.search_kwargs)
        elif self.search_type == "similarity_distance_threshold":
            if self.search_kwargs["distance_threshold"] is None:
                raise ValueError(
                    "distance_threshold must be provided for "
                    + "similarity_distance_threshold retriever"
                )
            docs = self.vectorstore.similarity_search(query, **self.search_kwargs)

        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query, **self.search_kwargs
                )
            )
            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to vectorstore."""
        return self.vectorstore.add_documents(documents, **kwargs)

    async def aadd_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        """Add documents to vectorstore."""
        return await self.vectorstore.aadd_documents(documents, **kwargs)