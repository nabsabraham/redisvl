import pytest

from redisvl.index import SearchIndex
from redisvl.query import FilterQuery
from redisvl.query.filter import Tag


@pytest.fixture
def filter_query():
    return FilterQuery(
        return_fields=None,
        filter_expression=Tag("credit_score") == "high",
    )


@pytest.fixture
def index(sample_data):
    fields_spec = [
        {"name": "credit_score", "type": "tag"},
        {"name": "user", "type": "tag"},
        {"name": "job", "type": "text"},
        {"name": "age", "type": "numeric"},
        {
            "name": "user_embedding",
            "type": "vector",
            "attrs": {
                "dims": 3,
                "distance_metric": "cosine",
                "algorithm": "flat",
                "datatype": "float32",
            },
        },
    ]

    json_schema = {
        "index": {
            "name": "user_index_json",
            "prefix": "users_json",
            "storage_type": "json",
        },
        "fields": fields_spec,
    }

    # construct a search index from the schema
    index = SearchIndex.from_dict(json_schema)

    # connect to local redis instance
    index.connect("redis://localhost:6379")

    # create the index (no data yet)
    index.create(overwrite=True)

    # Prepare and load the data
    index.load(sample_data)

    # run the test
    yield index

    # clean up
    index.delete(drop=True)


def test_process_results_unpacks_json_properly(index, filter_query):
    results = index.query(filter_query)
    assert len(results) == 4
