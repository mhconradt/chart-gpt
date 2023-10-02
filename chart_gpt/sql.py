import logging
import sys
import time

import pandas as pd
from pandas import DataFrame
from pandas import DataFrame
from pandas import Series
from snowflake.connector import DictCursor
from snowflake.connector import SnowflakeConnection

from chart_gpt.utils import ChartGptModel
from chart_gpt.utils import extract_json
from chart_gpt.utils import generate_completion
from chart_gpt.utils import generate_completion

logger = logging.getLogger(__name__)

# These have really high token counts

LLM_PANDAS_DISPLAY_OPTIONS = (
    "display.max_columns", 100,
    "display.width", 1000,
    "display.max_colwidth", 16
)

SQL_GENERATION_MODEL = 'gpt-4'

DEFAULT_SAMPLE_LIMIT = 3

SQL_EMBEDDING_MODEL = 'text-embedding-ada-002'
SUMMARY_DELIMITER = '\n' + '-' * 128 + '\n'


def get_table_context(data: "SQLIndexData") -> Series:
    descriptions = pd.concat([data.descriptions, data.foreign_keys])
    ddl = descriptions.groupby(axis=1, level='table').apply(get_create_table)
    select_x = data.samples.groupby(axis=1, level='table').apply(get_select_x)
    context = ddl + '\n' + select_x
    return context


class DatabaseCrawler(ChartGptModel):
    connection: SnowflakeConnection

    def show_tables(self) -> list[str]:
        logger.info("Listing tables")
        cursor = self.connection.cursor(cursor_class=DictCursor)
        query = "show tables;"
        tables = cursor.execute(query).fetchall()
        table_names = [table['name'] for table in tables]  # type: ignore
        return table_names

    def describe_table(self, table: str) -> DataFrame:
        logger.debug("Getting description for table %s", table)
        cursor = self.connection.cursor(cursor_class=DictCursor)
        query = "describe table identifier(%(table_name)s);"
        parameters = {"table_name": table}
        columns = cursor.execute(query, parameters).fetchall()
        df = pd.DataFrame(columns).set_index('name').dropna(axis=1).transpose()
        return df

    def sample_table(self, table: str, limit: int = DEFAULT_SAMPLE_LIMIT) -> DataFrame:
        logger.debug("Sampling %d rows from table %s", limit, table)
        cursor = self.connection.cursor(cursor_class=DictCursor)
        query = "select * from identifier(%(table_name)s) sample (%(limit)s rows);"
        parameters = {"table_name": table, "limit": limit}
        columns = cursor.execute(query, parameters).fetchall()
        df = pd.DataFrame(columns)
        df = df.dropna(axis=1)
        return df

    def get_table_descriptions(self, tables: list[str]) -> DataFrame:
        return pd.concat(
            [self.describe_table(table) for table in tables],
            keys=tables,
            names=['table', 'column'],
            axis=1
        )

    def get_table_samples(self, tables: list[str], n_rows: int = DEFAULT_SAMPLE_LIMIT) -> DataFrame:
        return pd.concat(
            [self.sample_table(table, limit=n_rows) for table in tables],
            keys=tables,
            names=['table', 'column'],
            axis=1
        )

    def get_index_data(self) -> "SQLIndexData":
        t0 = time.perf_counter()
        logger.info("Indexing tables")
        tables = self.show_tables()
        samples = self.get_table_samples(tables)
        descriptions = self.get_table_descriptions(tables)
        foreign_keys = self.get_foreign_keys()
        logger.info("Indexing tables took %.3f seconds", time.perf_counter() - t0)
        return SQLIndexData(
            samples=samples,
            descriptions=descriptions,
            foreign_keys=foreign_keys,
        )

    def get_foreign_keys(self) -> DataFrame:
        query = "select current_database(), current_schema();"
        database, schema = self.connection.cursor().execute(
            query).fetchone()
        cursor = self.connection.cursor(cursor_class=DictCursor)
        query = f"show imported keys in schema {database}.{schema};"
        foreign_keys = DataFrame(cursor.execute(query).fetchall())
        deduplicated = foreign_keys.drop_duplicates(['fk_table_name', 'fk_column_name'])
        indexed = deduplicated.set_index(['fk_table_name', 'fk_column_name'])  # type: ignore
        renamed = indexed.rename_axis(['table', 'column'])
        selected = renamed[['pk_table_name', 'pk_column_name']]
        transposed = selected.transpose()
        return transposed

    def get_index(self) -> "SQLIndex":
        return SQLIndex.from_data(self.get_index_data())


def postprocess_generated_sql(answer: str) -> str:
    start, stop = '```sql', '```'
    start_index = answer.index(start) + len(start)
    stop_index = answer.index(stop, start_index)
    return answer[start_index:stop_index].strip()


class SQLIndexData(ChartGptModel):
    # row index: column_property
    # column index: (table name, column name)
    descriptions: DataFrame
    # row index: i
    # column index: (table name, column name)
    samples: DataFrame
    foreign_keys: DataFrame

    class Config:
        arbitrary_types_allowed = True

    @property
    def tables(self) -> list[str]:
        return list(self.samples.columns.levels[0])


class SQLIndex(ChartGptModel):
    data: SQLIndexData
    # (table) -> dim_0, dim_1, ..., dim_n
    context: Series

    @classmethod
    def from_data(cls, data: SQLIndexData) -> "SQLIndex":
        # [table, column] -> [text]
        table_context = get_table_context(data)
        return cls(data=data, context=table_context)

    def top_tables(self, query: str) -> list[str]:
        completion = generate_completion(f"""
        Tables: {self.data.tables}
        Question: {query}
        JSON list of tables to query in order to answer question:
        """, model='gpt-4')
        tables = extract_json(completion, start='[', stop=']')
        assert isinstance(tables, list)
        return tables

    def top_context(self, query: str) -> list[str]:
        return self.context.loc[self.top_tables(query)].tolist()


class SQLGenerator(ChartGptModel):
    connection: SnowflakeConnection
    index: SQLIndex

    def generate_valid_query(self, question: str) -> str:
        logger.info("Generating query for question: %s", question)
        errors = []
        n = 1
        while True:
            prompt = self.build_prompt(question, errors)
            statement = self.generate(prompt)
            try:
                self.validate(statement)
            except Exception as e:
                errors.append(f"Query: {statement}. Error: {e}")
                logging.warning("Invalid query (%d/3): %s. Error %s", n, statement, e,
                                stack_info=False)
                n += 1
                if n > 3:
                    raise
            else:
                logging.info("Generated valid query: %s")
                return statement

    def generate(self, prompt):
        answer = generate_completion(prompt)
        statement = postprocess_generated_sql(answer)
        return statement

    def build_prompt(self, question, errors):
        summary = self.get_context(question)
        system_prompt = f"""
                Write a valid Snowflake SQL query that answers the question/command: {question}
                Use the following tables: {summary}
                The query should be in markdown format: beginning with ```sql and ending with ```.
                {self.get_error_prompt(errors)}
            """
        return system_prompt

    def get_context(self, question: str) -> str:
        context = self.index.top_context(question)
        summary = SUMMARY_DELIMITER.join(context)
        return summary

    def validate(self, statement: str):
        self.connection.cursor().describe(statement)
        content = self.connection.cursor().execute(f"explain using text {statement}").fetchone()[0]
        if 'CartesianJoin' in content:
            raise ValueError(
                "Query must not use CartesianJoin. This is likely a bug, use union all instead.")

    def get_error_prompt(self, messages) -> str:
        if messages:
            return "Previous attempts:\n" + "\n".join(messages)
        return ""


def get_column_ddl(column: Series) -> str:
    # https://docs.snowflake.com/en/sql-reference/sql/create-table#syntax
    parts = [column.name[1], column['type']]  # type: ignore
    if isinstance(column['pk_table_name'], str):
        fk_table, fk_column = column['pk_table_name'], column['pk_column_name']
        parts.append(f"REFERENCES {fk_table}({fk_column})")
    if column['null?'] == 'N':
        parts.append('NOT NULL')
    if column['primary key'] == 'Y':
        parts.append('PRIMARY KEY')
    return " ".join(parts)


def get_create_table(description: DataFrame) -> str:
    columns = description.apply(get_column_ddl)
    all_columns = ",\n    ".join(columns)
    # Does whitespace matter here?
    return f"""CREATE TABLE {description.name} ( {all_columns} );"""


def get_select_x(sample: DataFrame, n_rows: int = 3) -> str:
    with pd.option_context(*LLM_PANDAS_DISPLAY_OPTIONS):
        parts = [
            "/*",
            f"{n_rows} example rows:",
            f"SELECT * FROM {sample.name} LIMIT {n_rows};",
            str(sample.droplevel(0, axis=1).rename_axis(None, axis=1).head(n_rows)),
            "*/"
        ]
        return "\n".join(parts)


def chat_summarize_data(result_set: DataFrame, question: str, query: str) -> str:
    with pd.option_context(*LLM_PANDAS_DISPLAY_OPTIONS):
        return generate_completion(f"""
            User's question: {question}.
            Generated SQL query: {query}
            SQL query result set: {result_set}
            Summarize how this dataset answers the user's question:
        """, model='gpt-3.5-turbo')
