import json

import pytest
import tiktoken
from snowflake.connector import SnowflakeConnection
from tiktoken import Encoding

from chart_gpt import DatabaseCrawler
from chart_gpt import Index
from chart_gpt import IndexData
from chart_gpt import SQLGenerator
from chart_gpt import get_connection


@pytest.fixture
def database_connection() -> SnowflakeConnection:
    return get_connection()


@pytest.fixture
def database_crawler(database_connection) -> DatabaseCrawler:
    return DatabaseCrawler(database_connection)


@pytest.fixture
def tpc_ds_index_data(database_crawler) -> IndexData:
    if True:
        index_data = database_crawler.get_index_data()
        return index_data


@pytest.fixture
def tpc_ds_index(tpc_ds_index_data):
    if True:
        return Index.from_data(tpc_ds_index_data)


@pytest.fixture
def tpc_ds_questions() -> list[str]:
    with open('data/TPCDS_SF10TCL-queries.json', 'r') as f:
        return json.load(f)


@pytest.fixture
def sql_generator(database_connection, tpc_ds_index):
    return SQLGenerator(database_connection, tpc_ds_index)


@pytest.fixture
def gpt4_encoding() -> Encoding:
    return tiktoken.encoding_for_model('gpt-4')


def test_get_context(tpc_ds_questions, index, sql_generator, gpt4_encoding):
    for question in tpc_ds_questions:
        context = sql_generator.get_context(question)
        tokens = gpt4_encoding.encode(context)
        n_tokens = len(tokens)
        assert n_tokens <= 8192
