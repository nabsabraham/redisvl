

# %%
import os
import sys
import logging
import textwrap

import warnings

warnings.filterwarnings("ignore")

# stop huggingface warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Uncomment to see debug logs
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.vector_stores import RedisVectorStore
from IPython.display import Markdown, display


os.environ["OPENAI_API_KEY"] = "sk-<your key here>"

# %% [markdown]
# Download Data

# %%
!mkdir -p 'data/paul_graham/'
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

# %% [markdown]
# ### Read in a dataset
# Here we will use a set of Paul Graham essays to provide the text to turn into embeddings, store in a ``RedisVectorStore`` and query to find context for our LLM QnA loop.

# %%
# load documents
documents = SimpleDirectoryReader("./data/paul_graham").load_data()
print(
    "Document ID:",
    documents[0].doc_id,
    "Document Hash:",
    documents[0].doc_hash,
)

# %% [markdown]
# You can process your files individually using [SimpleDirectoryReader](/examples/data_connectors/simple_directory_reader.ipynb):

# %%
loader = SimpleDirectoryReader("./data/paul_graham")
documents = loader.load_data()
for file in loader.input_files:
    print(file)


from llama_index.storage.storage_context import StorageContext

vector_store = RedisVectorStore(
    index_name="pg_essays",
    index_prefix="llama",
    redis_url="redis://localhost:6379",  # Default
    overwrite=True,
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# %% [markdown]
# ### Handle duplicated index
# 
# Regardless of whether overwrite=True is used in RedisVectorStore(), the process of generating the index and storing data in Redis still takes time. Currently, it is necessary to implement your own logic to manage duplicate indexes. One possible approach is to set a flag in Redis to indicate the readiness of the index. If the flag is set, you can bypass the index generation step and directly load the index from Redis.

# # %%
# import redis
# r = redis.Redis()
# index_name = "pg_essays"
# r.set(f"added:{index_name}", "true")

# # Later in code
# if r.get(f"added:{index_name}"):
#     # Skip to deploy your index, restore it. Please see "Restore index from Redis" section below. 

# %% [markdown]
# ### Query the data
# Now that we have our document stored in the index, we can ask questions against the index. The index will use the data stored in itself as the knowledge base for ChatGPT. The default setting for as_query_engine() utilizes OpenAI embeddings and ChatGPT as the language model. Therefore, an OpenAI key is required unless you opt for a customized or local language model.

# %%
query_engine = index.as_query_engine()
response = query_engine.query("What did the author learn?")
print(textwrap.fill(str(response), 100))

# %%
response = query_engine.query("What was a hard moment for the author?")
print(textwrap.fill(str(response), 100))

# %% [markdown]
# ### Saving and Loading
# 
# Redis allows the user to perform backups in the background or synchronously. With Llamaindex, the ``RedisVectorStore.persist()`` function can be used to trigger such a backup.

# %%
!docker exec -it redis-vecdb ls /data

# %%
# RedisVectorStore's persist method doesn't use the persist_path argument
vector_store.persist(persist_path="")

# %%
!docker exec -it redis-vecdb ls /data

# %% [markdown]
# ### Restore index from Redis

# %%
vector_store = RedisVectorStore(
    index_name="pg_essays",
    index_prefix="llama",
    redis_url="redis://localhost:6379",
    overwrite=True,
)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# %% [markdown]
# Now you can reuse your index as discussed above.

# %%
pgQuery = index.as_query_engine()
pgQuery.query("What is the meaning of life?")
# or
pgRetriever = index.as_retriever()
pgRetriever.retrieve("What is the meaning of life?")

# %% [markdown]
# Learn more about [query_engine](/module_guides/deploying/query_engine/root.md)  and [retrievers](/module_guides/querying/retriever/root.md).

# %% [markdown]
# ### Deleting documents or index completely
# 
# Sometimes it may be useful to delete documents or the entire index. This can be done using the `delete` and `delete_index` methods.

# %%
document_id = documents[0].doc_id
document_id

# %%
redis_client = vector_store.client
print("Number of documents", len(redis_client.keys()))

# %%
vector_store.delete(document_id)

# %%
print("Number of documents", len(redis_client.keys()))

# %%
# now lets delete the index entirely (happens in the background, may take a second)
# this will delete all the documents and the index
vector_store.delete_index()

# %%
print("Number of documents", len(redis_client.keys()))

# %% [markdown]
# ### Working with Metadata
# 
# RedisVectorStore supports adding metadata and then using it in your queries (for example, to limit the scope of documents retrieved). However, there are a couple of important caveats:
# 1. Currently, only [Tag fields](https://redis.io/docs/stack/search/reference/tags/) are supported, and only with exact match.
# 2. You must declare the metadata when creating the index (usually when initializing RedisVectorStore). If you do not do this, your queries will come back empty. There is no way to modify an existing index after it had already been created (this is a Redis limitation).
# 
# Here's how to work with Metadata:
# 
# 
# ### When **creating** the index
# 
# Make sure to declare the metadata when you **first** create the index:

# %%
vector_store = RedisVectorStore(
    index_name="pg_essays_with_metadata",
    index_prefix="llama",
    redis_url="redis://localhost:6379",
    overwrite=True,
    metadata_fields=["user_id", "favorite_color"],
)

# %% [markdown]
# Note: the field names `text`, `doc_id`, `id` and the name of your vector field (`vector` by default) should **not** be used as metadata field names, as they are are reserved.

# %% [markdown]
# ### When adding a document
# 
# Add your metadata under the `metadata` key. You can add metadata to documents you load in just by looping over them:

# %%
# load your documents normally, then add your metadata
documents = SimpleDirectoryReader("../data/paul_graham").load_data()

for document in documents:
    document.metadata = {"user_id": "12345", "favorite_color": "blue"}

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# load documents
print(
    "Document ID:",
    documents[0].doc_id,
    "Document Hash:",
    documents[0].doc_hash,
    "Metadata:",
    documents[0].metadata,
)

# %% [markdown]
# ### When querying the index
# 
# To filter by your metadata fields, include one or more of your metadata keys, like so:

# %%
from llama_index.vector_stores.types import MetadataFilters, ExactMatchFilter

query_engine = index.as_query_engine(
    similarity_top_k=3,
    filters=MetadataFilters(
        filters=[
            ExactMatchFilter(key="user_id", value="12345"),
            ExactMatchFilter(key="favorite_color", value="blue"),
        ]
    ),
)

response = query_engine.query("What did the author learn?")
print(textwrap.fill(str(response), 100))

# %% [markdown]
# ## Troubleshooting
# 
# In case you run into issues retrieving your documents from the index, you might get a message similar to this.
# ```
# No docs found on index 'pg_essays' with prefix 'llama' and filters '(@user_id:{12345} & @favorite_color:{blue})'.
# * Did you originally create the index with a different prefix?
# * Did you index your metadata fields when you created the index?
# ```
# 
# If you get this error, there a couple of gotchas to be aware of when working with Redis:
# #### Prefix issues
# 
# If you first create your index with a specific `prefix` but later change that prefix in your code, your query will come back empty. Redis saves the prefix your originally created your index with and expects it to be consistent.
# 
# To see what prefix your index was created with, you can run `FT.INFO <name of your index>` in the Redis CLI and look under `index_definition` => `prefixes`.
# 
# #### Empty queries when using metadata
# 
# If you add metadata to the index *after* it has already been created and then try to query over that metadata, your queries will come back empty.
# 
# Redis indexes fields upon index creation only (similar to how it indexes the prefixes, above).
# 
# If you have an existing index and want to make sure it's dropped, you can run `FT.DROPINDEX <name of your index>` in the Redis CLI. Note that this will *not* drop your actual data.


