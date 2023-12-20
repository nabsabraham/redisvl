"""Llama Index Redis Vector store index.

An index that that is built on top of an existing vector store.
"""
import logging

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from llama_index.bridge.pydantic import PrivateAttr
from llama_index.schema import (
    BaseNode,
    MetadataMode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import metadata_dict_to_node, node_to_metadata_dict

from redis import Redis

from redisvl.utils.utils import array_to_buffer
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery, CountQuery, FilterQuery
from redisvl.query.filter import Tag, FilterExpression
from redisvl.integrations.llama_index.schema import LlamaIndexSchema


_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from redis.client import Redis as RedisType
    from redis.commands.search.field import VectorField


class RedisVectorStore(BasePydanticVectorStore):
    stores_text = True
    stores_node = True
    flat_metadata = False

    index: SearchIndex
    schema: LlamaIndexSchema

    _vector_field_args: Dict[str, Any] = PrivateAttr()
    _overwrite: bool = PrivateAttr()
    _return_fields: List[str] = PrivateAttr()

    def __init__(
        self,
        index_name: str,
        client: Redis,
        index_prefix: str = "llama_index",
        prefix_ending: str = "/vector",
        vector_field_args: Dict[str, Any] = {},
        metadata_fields: List[str] = [],
        # redis_url: str = "redis://localhost:6379",
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize RedisVectorStore.

        For index arguments that can be passed to RediSearch, see
        https://redis.io/docs/stack/search/reference/vectors/

        The index arguments will depend on the index type chosen. There
        are two available index types
            - FLAT: a flat index that uses brute force search
            - HNSW: a hierarchical navigable small world graph index

        Args:
            index_name (str): Name of the index.
            client (Redis): The redis client instance.
            index_prefix (str): Prefix for the index. Defaults to "llama_index".
                The actual prefix used by Redis will be
                "{index_prefix}{prefix_ending}".
            prefix_ending (str): Prefix ending for the index. Be careful when
                changing this: https://github.com/jerryjliu/llama_index/pull/6665.
                Defaults to "/vector".
            vector_field_args (Dict[str, Any]): Arguments for the vector field.
                Defaults to None.
            metadata_fields (List[str]): List of metadata fields to store in the
                index (only supports TAG fields).
            overwrite (bool): Whether to overwrite the index if it already exists.
                Defaults to False.
            kwargs (Any): Additional arguments to pass to the redis client.

        Raises:
            ValueError: If redis-py is not installed
            ValueError: If RediSearch is not installed

        Examples:
            >>> from llama_index.vector_stores.redis import RedisVectorStore
            >>> # Create a RedisVectorStore
            >>> vector_store = RedisVectorStore(
            >>>     index_name="my_index",
            >>>     index_prefix="llama_index",
            >>>     vector_field_args={"name": "embedding", "algorithm": "HNSW", "m": 16, "ef_construction": 200, "distance_metric": "cosine"},
            >>>     redis_url="redis://localhost:6379/",
            >>>     overwrite=True)
        """

        if "redis_url" in kwargs:
            client = Redis.from_url(kwargs.pop("redis_url"))
            _logger.warning(
                "Deprecation warning: 'redis_url' is deprecated, in the future please provide a Redis client instance"
            )

        if "index_args" in kwargs:
            vector_field_args = kwargs.pop("index_args")
            _logger.warning(
                "Deprecation warning: 'index_args' is deprecated, in the future please use 'vector_field_args' instead."
            )

        self._vector_field_args = vector_field_args or {}
        self._overwrite = overwrite

        # Create LlamaIndexSchema and Index
        vector_field_name = self._get_vector_field_name()
        self.schema = LlamaIndexSchema(
            name=index_name,
            prefix=index_prefix+prefix_ending,
            vector_field_name=vector_field_name,
            metadata_fields=metadata_fields
        )
        self.index = SearchIndex(schema=self.schema)
        self.index.set_client(client)
        self._return_fields = [
            "id", "doc_id", "text", vector_field_name, "vector_score", "_node_content",
        ]
        super().__init__()

    def _get_vector_field_name(self) -> str:
        """Get the vector field name."""
        # handles backwards compatibility
        vector_field = self._vector_field_args.pop("vector_field", None)
        vector_key = self._vector_field_args.pop("vector_key", None)
        name = self._vector_field_args.get("name", "vector")
        return name or vector_field or vector_key

    @property
    def client(self) -> Redis:
        """Return the redis client instance."""
        return self.index.client

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """Add nodes to the index.

        Args:
            nodes (List[BaseNode]): List of nodes with embeddings

        Returns:
            List[str]: List of ids of the documents added to the index.

        Raises:
            ValueError: If the index already exists and overwrite is False.
        """
        # Check to see if empty document list was passed
        if len(nodes) == 0:
            return []

        if not self.index.exists() or self._overwrite == True:
            # Update vector dims in schema
            self._vector_field_args["dims"] = len(nodes[0].get_embedding())
            # Make sure to add the vector field
            if self.schema.vector_field_name not in self.schema.field_names:
                self.schema.add_field(
                    "vector", name=self.schema.vector_field_name, **self._vector_field_args
                )

        # Create the index using user-defined overwrite policy
        self.index.create(overwrite=self._overwrite)

        def preprocess_node(node: BaseNode) -> Dict[str, Any]:
            obj = {
                "id": node.node_id,
                "doc_id": node.ref_doc_id,
                "text": node.get_content(metadata_mode=MetadataMode.NONE),
                self.schema.vector_field_name: array_to_buffer(node.get_embedding()),
            }
            additional_metadata = node_to_metadata_dict(
                node, remove_text=True, flat_metadata=self.flat_metadata
            )
            return {**obj, **additional_metadata}

        # Load data to the index
        raw_ids = self.index.load(
            nodes, key_field="id", preprocess=preprocess_node
        )
        ids = [
            id.strip(self.index.prefix + self.index.key_separator) for id in raw_ids
        ]

        _logger.info(f"Added {len(ids)} documents to index {self.index._name}")
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        # Find docs in the index that match the document_id
        doc_filter = Tag("doc_id") == ref_doc_id
        total = self.index.query(CountQuery(doc_filter))
        results = self.index.query(FilterQuery(
            return_fields=["id"],
            filter_expression=doc_filter,
            num_results=total
        ))
        if len(results) == 0:
            # don't raise an error but warn the user that doc wasn't found
            # could be a result of eviction policy
            _logger.warning(
                f"Document with doc_id {ref_doc_id} not found "
                f"in index {self.index.name}"
            )
            return

        # clean up keys
        with self.index.client.pipeline(transaction=False) as pipe:
            for doc in results:
                pipe.delete(doc["id"])
            pipe.execute()

        _logger.info(
            f"Deleted {len(results)} documents from index {self.index.name}"
        )

    def delete_index(self) -> None:
        """Delete the index and all documents."""
        _logger.info(f"Deleting index {self.index._name}")
        self.index.delete(drop=True)

    @staticmethod
    def _create_redis_filter(self, metadata_filters: MetadataFilters) -> FilterExpression:
        """_summary_

        Args:
            metadata_filters (MetadataFilters): _description_

        Returns:
            FilterExpression: _description_
        """
        # Currently only supports TAG matches
        # Index must be created with the metadata field in the index schema
        # otherwise this will raise an error
        filter_expression = FilterExpression("*")
        for filter in metadata_filters.filters:
            if filter.key not in self.schema.field_names:
                raise ValueError(f"{filter.key} field was not indexed as part of the schema.")
            filter_expression = filter_expression & (Tag(filter.key) == filter.value)
        return filter_expression

    def _create_redis_query(self, query: VectorStoreQuery) -> VectorQuery:
        """Creates a RedisQuery from a VectorStoreQuery."""
        filters = self._create_redis_filter(query.filters)
        redis_query = VectorQuery(
            vector=query.query_embedding,
            vector_field_name=self.schema.vector_field_name,
            return_fields=self._return_fields,
            num_results=query.similarity_top_k
        )
        redis_query.set_filter(filters)
        return redis_query

    def _extract_node_and_score(self, doc, redis_query: VectorQuery):
        """Extracts a node and its score from a document."""
        try:
            node = metadata_dict_to_node({"_node_content": doc["_node_content"]})
            node.text = doc["text"]
        except Exception:
            # Handle legacy metadata format
            node = TextNode(
                text=doc["text"],
                id_=doc["id"],
                embedding=None,
                relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=doc["doc_id"])}
            )
        score = 1 - float(doc[redis_query.DISTANCE_ID])
        return node, score

    def _process_query_results(self, results, redis_query: VectorQuery) -> VectorStoreQueryResult:
        """Processes query results and returns a VectorStoreQueryResult."""
        ids, nodes, scores = [], [], []
        for doc in results:
            node, score = self._extract_node_and_score(doc, redis_query)
            ids.append(doc["id"])
            nodes.append(node)
            scores.append(score)
        _logger.info(f"Found {len(nodes)} results for query with id {ids}")
        return VectorStoreQueryResult(nodes=nodes, ids=ids, similarities=scores)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query the index.

        Args:
            query (VectorStoreQuery): Query object.

        Returns:
            VectorStoreQueryResult: Query result.

        Raises:
            ValueError: If query.query_embedding is None or no documents are found.
            RedisError: If there is an error querying the index.
        """
        from redis.exceptions import RedisError
        from redis.exceptions import TimeoutError as RedisTimeoutError

        if not query.query_embedding:
            raise ValueError("Query embedding is required for querying.")

        redis_query = self._create_redis_query(query)
        _logger.info(f"Querying index {self.index.name} with filters {redis_query.filters}")

        try:
            results = self.index.query(redis_query)
        except (RedisTimeoutError, RedisError) as e:
            _logger.error(f"Error querying {self.index.name}: {e}")
            raise

        if not results:
            raise ValueError(
                f"No docs found on index '{self.index.name}' with prefix '{self.index.prefix}' "
                "and filters '{redis_query.filters}'. * Check the index prefix and metadata fields."
            )

        return self._process_query_results(results, redis_query)
