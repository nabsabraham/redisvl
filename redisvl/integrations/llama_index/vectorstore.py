"""Llama Index Redis Vector store index.

An index that that is built on top of an existing vector store.
"""
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import fsspec

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

from redisvl.utils.utils import array_to_buffer
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag, FilterExpression
from redisvl.integrations.llama_index.schema import LlamaIndexSchema


_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from redis.client import Redis as RedisType
    from redis.commands.search.field import VectorField


# currently only supports exact tag match - {} denotes a tag
# must create the index with the correct metadata field before using a field as a
#   filter, or it will return no results
def create_filter_expression(metadata_filters: MetadataFilters) -> FilterExpression:
    filter_expression = FilterExpression("*")
    for filter in metadata_filters.filters:
        filter_expression = filter_expression & (Tag(filter.key) == filter.value)
    return filter_expression


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
        index_prefix: str = "llama_index",
        prefix_ending: str = "/vector",
        vector_field_args: Optional[Dict[str, Any]] = None,
        metadata_fields: Optional[List[str]] = None,
        redis_url: str = "redis://localhost:6379",
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
            redis_url (str): URL for the redis instance.
                Defaults to "redis://localhost:6379".
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
            >>>     index_args={"algorithm": "HNSW", "m": 16, "ef_construction": 200,
                "distance_metric": "cosine"},
            >>>     redis_url="redis://localhost:6379/",
            >>>     overwrite=True)
        """
        try:
            import redis
            # TODO: is this necessary at this point??
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )

        # Create LlamaIndexSchema and Index
        # TODO: favor / deprecate one of these...
        vector_field = str(self._index_args.get("vector_field"))
        vector_key = str(self._index_args.get("vector_key"))
        self.schema = LlamaIndexSchema(
            name=index_name,
            prefix=index_prefix+prefix_ending,
            vector_field_name=vector_field or vector_key or "vector",
            metadata_fields=metadata_fields
        )
        self.index = SearchIndex(schema=self.schema, redis_url=redis_url)

        if "index_args" in kwargs:
            vector_field_args = kwargs.pop("index_args")
            ## TODO: raise deprecation warning

        self._vector_field_args = vector_field_args or {}
        self._overwrite = overwrite
        self._return_fields = [
            "id",
            "doc_id",
            "text",
            self.schema.vector_field_name,
            "vector_score",
            "_node_content",
        ]

        super().__init__()

    @property
    def client(self) -> "RedisType":
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
        # check to see if empty document list was passed
        if len(nodes) == 0:
            return []

        # Check vector dims in schema?
        self._vector_field_args["dims"] = len(nodes[0].get_embedding())

        # create the index (using the overwrite policy defined by user)
        self.index.create(overwrite=self._overwrite)

        # Make sure vector field is added to the index schema
        self.schema.add_field("vector", **self._vector_field_args)

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

        # TODO: need a better way to do this here...
        ids = [node.node_id for node in nodes]

        self.index.load(
            nodes,
            key_field="id",
            preprocess=preprocess_node
        )

        _logger.info(f"Added {len(ids)} documents to index {self.index._name}")
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        # find all documents that match a doc_id
        doc_filter = Tag("doc_id") == ref_doc_id
        results = self.index.search(str(doc_filter))
        # TODO does this actually return all results here?? or only partial?
        if len(results.docs) == 0:
            # don't raise an error but warn the user that doc wasn't found
            # could be a result of eviction policy
            _logger.warning(
                f"Document with doc_id {ref_doc_id} not found "
                f"in index {self.index._name}"
            )
            return

        # clean up keys
        with self.index.client.pipeline(transaction=False) as pipe:
            for doc in results.docs:
                pipe.delete(doc.id)
            pipe.execute()

        _logger.info(
            f"Deleted {len(results.docs)} documents from index {self.index.name}"
        )

    def delete_index(self) -> None:
        """Delete the index and all documents."""
        _logger.info(f"Deleting index {self.index._name}")
        self.index.delete(drop=True)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query the index.

        Args:
            query (VectorStoreQuery): query object

        Returns:
            VectorStoreQueryResult: query result

        Raises:
            ValueError: If query.query_embedding is None.
            redis.exceptions.RedisError: If there is an error querying the index.
            redis.exceptions.TimeoutError: If there is a timeout querying the index.
            ValueError: If no documents are found when querying the index.
        """
        from redis.exceptions import RedisError
        from redis.exceptions import TimeoutError as RedisTimeoutError

        if not query.query_embedding:
            raise ValueError("Query embedding is required for querying.")

        redis_query = VectorQuery(
            vector=query.query_embedding,
            vector_field_name=self.schema.vector_field_name,
            return_fields=self._return_fields,
            num_results=query.similarity_top_k
        )
        filters = create_filter_expression(query.filters)
        redis_query.set_filter(filters)

        _logger.info(f"Querying index {self.index.name} with filters {filters}")

        try:
            results = self.index.query(redis_query)
        except RedisTimeoutError as e:
            _logger.error(f"Query timed out on {self.index.name}: {e}")
            raise
        except RedisError as e:
            _logger.error(f"Error querying {self.index.name}: {e}")
            raise

        if len(results) == 0:
            raise ValueError(
                f"No docs found on index '{self.index.name}' with "
                f"prefix '{self.index.prefix}' and filters '{filters}'. "
                "* Did you originally create the index with a different prefix? "
                "* Did you index your metadata fields when you created the index?"
            )

        ids = []
        nodes = []
        scores = []
        for doc in results:
            try:
                node = metadata_dict_to_node({"_node_content": doc["_node_content"]})
                node.text = doc["text"]
            except Exception:
                # TODO: Legacy support for old metadata format
                node = TextNode(
                    text=doc["text"],
                    id_=doc["id"],
                    embedding=None,
                    relationships={
                        NodeRelationship.SOURCE: RelatedNodeInfo(node_id=doc["doc_id"])
                    },
                )
            ids.append(doc["id"])
            nodes.append(node)
            scores.append(1 - float(doc[redis_query.DISTANCE_ID]))

        _logger.info(f"Found {len(nodes)} results for query with id {ids}")
        return VectorStoreQueryResult(nodes=nodes, ids=ids, similarities=scores)

    def _create_vector_field(self) -> "VectorField":
        """Create a RediSearch VectorField.

        Args:
            name (str): The name of the field.
            algorithm (str): The algorithm used to index the vector.
            dims (int): The dimensionality of the vector.
            datatype (str): The type of the vector. default: FLOAT32
            distance_metric (str): The distance metric used to compare vectors.
            initial_cap (int): The initial capacity of the index.
            block_size (int): The block size of the index.
            m (int): The number of outgoing edges in the HNSW graph.
            ef_construction (int): Number of maximum allowed potential outgoing edges
                            candidates for each node in the graph,
                            during the graph building.
            ef_runtime (int): The umber of maximum top candidates to hold during the
                KNN search

        Returns:
            A RediSearch VectorField.
        """
        DEFAULTS = {
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
        
        algorithm = self._vector_field_args.get("algorithm", "flat")
        
        if algorithm.upper() == "HNSW":
            return {
                "name": self.schema.vector_field_name,
                **DEFAULTS,
                **self._vector_field_args
            }
                "type": "hnsw",
                "dims": dims,
                "distance": distance_metric.upper(),
                "m": m,
                "ef_construction": ef_construction,
                "ef_runtime": ef_runtime,
                "epsilon": epsilon,
                "initial_cap": initial_cap
            }
        else:
            return {
                "name": name,
                "type": "flat",
                "dims": dims,
                "distance": distance_metric.upper(),
                "initial_cap": initial_cap,
                "block_size": block_size,
            }
