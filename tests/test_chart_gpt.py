import json

import pandas as pd
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
from chart_gpt import get_create_table
from chart_gpt import get_select_x
from chart_gpt import postprocess_generated_sql


@pytest.fixture
def database_connection() -> SnowflakeConnection:
    return get_connection()


@pytest.fixture
def database_crawler(database_connection) -> DatabaseCrawler:
    return DatabaseCrawler(database_connection)


@pytest.fixture
def tpc_ds_index_data(database_crawler) -> SQLIndexData:
    if False:
        fks = database_crawler.get_foreign_keys()
        return fks
    else:
        return SQLIndexData(
            descriptions=pd.read_parquet('tests/data/tpc_ds_table_descriptions.parquet'),
            samples=pd.read_parquet('tests/data/tpc_ds_table_samples.parquet'),
            foreign_keys=pd.read_parquet('tests/data/tpc_ds_foreign_keys.parquet'),
        )


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
        context = sql_generator.get_context(question, n_tables=10)
        tokens = gpt4_encoding.encode(context)
        n_tokens = len(tokens)
        assert n_tokens <= 8192


def test_generate(tpc_ds_questions, sql_generator):
    for question in tpc_ds_questions:
        generated = sql_generator.generate(question)


def test_crawler(database_crawler):
    descriptions = database_crawler.get_table_descriptions(5)
    assert isinstance(descriptions, DataFrame)
    assert descriptions.columns.names == ['table', 'column']
    samples = database_crawler.get_table_samples(n_rows=5, n_tables=5)
    assert isinstance(samples, DataFrame)
    assert samples.columns.names == ['table', 'column']


def test_index(tpc_ds_index_data):
    descriptions = pd.concat([tpc_ds_index_data.descriptions, tpc_ds_index_data.foreign_keys])
    ddls = descriptions.groupby(axis=1, level='table').apply(get_create_table)
    samples = tpc_ds_index_data.samples.groupby(axis=1, level='table').apply(get_select_x)
    index = SQLIndex.from_data(tpc_ds_index_data)
    # text = SQLIndex.get_text_sample(tpc_ds_index_data)
    # assert text.index.names == ['table', 'column']
    # assert text.columns == ['text']


def test_postprocess_sql():
    query = postprocess_generated_sql("```sql select * from foo; ``` Foo bar is the ...")
    assert query == "select * from foo;"
