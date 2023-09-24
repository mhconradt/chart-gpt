import json

import pytest
import tiktoken
from pandas import DataFrame
from snowflake.connector import SnowflakeConnection
from tiktoken import Encoding

from chart_gpt import DatabaseCrawler
from chart_gpt import SQLIndex
from chart_gpt import SQLIndexData
from chart_gpt import SQLGenerator
from chart_gpt import get_connection


@pytest.fixture
def database_connection() -> SnowflakeConnection:
    return get_connection()


@pytest.fixture
def database_crawler(database_connection) -> DatabaseCrawler:
    return DatabaseCrawler(database_connection)


@pytest.fixture
def tpc_ds_index_data(database_crawler) -> SQLIndexData:
    if True:
        index_data = database_crawler.get_index_data()
        return index_data


@pytest.fixture
def index(tpc_ds_index_data):
    if True:
        return SQLIndex.from_data(tpc_ds_index_data)


@pytest.fixture
def tpc_ds_questions() -> list[str]:
    with open('data/TPCDS_SF10TCL-queries.json', 'r') as f:
        return json.load(f)


@pytest.fixture
def sql_generator(database_connection, index):
    return SQLGenerator(database_connection, index)


@pytest.fixture
def gpt4_encoding() -> Encoding:
    return tiktoken.encoding_for_model('gpt-4')


def test_get_context(tpc_ds_questions, index, sql_generator, gpt4_encoding):
    for question in tpc_ds_questions[:5]:
        context = sql_generator.get_context(question, n_tables=10, n_columns=10)
        tokens = gpt4_encoding.encode(context)
        n_tokens = len(tokens)
        assert n_tokens <= 8192


def test_crawler(database_crawler):
    descriptions = database_crawler.get_table_descriptions(5)
    assert isinstance(descriptions, DataFrame)
    assert descriptions.columns.names == ['table', 'column']
    samples = database_crawler.get_table_samples(n_rows=5, n_tables=5)
    assert isinstance(samples, DataFrame)
    assert samples.columns.names == ['table', 'column']


def test_index(database_crawler):
    index_data = database_crawler.get_index_data()
    text = SQLIndex.get_text_sample(index_data)
    assert text.index.names == ['table', 'column']
    assert text.columns == ['text']
