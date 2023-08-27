from os import getenv

import openai
import pandas as pd
import requests
import tiktoken
from openai.embeddings_utils import cosine_similarity
from openai.embeddings_utils import get_embedding
from pandas import DataFrame
from snowflake.connector import DictCursor
from snowflake.connector import SnowflakeConnection
from snowflake.connector import connect

SQL_GENERATION_MODEL = 'gpt-4'

DEFAULT_SAMPLE_LIMIT = 10

DEFAULT_TABLE_LIMIT = 10

SYSTEM_PROMPT_MESSAGE = {
    "role": "system",
    "content": "You're a helpful assistant powering a BI tool. " \
               "Part of your work is generating structured output such as JSON or SQL."
}

EMBEDDING_MODEL = 'text-embedding-ada-002'
SUMMARY_DELIMITER = '\n' + '-' * 128 + '\n'


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# Planning: Determine the list of columns and the chart type


def get_snowflake_connection():
    return connect(
        account=getenv('SF_ACCOUNT'),
        user=getenv('SF_USER'),
        password=getenv('SF_PASSWORD'),
        role=getenv('SF_ROLE'),
        database=getenv('SF_DATABASE'),
        schema=getenv('SF_SCHEMA')
    )


class DataBoss:
    def __init__(self, connection: SnowflakeConnection):
        self.connection = connection

    def show_tables(self, limit: int = DEFAULT_TABLE_LIMIT) -> list[str]:
        cursor = self.connection.cursor(cursor_class=DictCursor)
        return [table['name'] for table in cursor.execute("show tables;").fetchall()[:limit]]

    def describe_table(self, table: str) -> str:
        cursor = self.connection.cursor(cursor_class=DictCursor)
        columns = cursor.execute("describe table identifier(%(table_name)s);",
                                 {"table_name": table}).fetchall()
        df = pd.DataFrame(columns)
        df = df[[c for c in df.columns if df[c].notna().any()]]
        return df.to_markdown()

    def sample_table(self, table: str, limit: int = DEFAULT_SAMPLE_LIMIT) -> str:
        cursor = self.connection.cursor(cursor_class=DictCursor)
        columns = cursor.execute(
            "select * from identifier(%(table_name)s) sample (%(limit)s rows);",
            {"table_name": table, "limit": limit}).fetchall()
        df = pd.DataFrame(columns)
        df = df[[c for c in df.columns if df[c].notna().any()]]
        return df.to_markdown()

    def get_table_descriptions(self, limit: int = DEFAULT_TABLE_LIMIT) -> dict[str, str]:
        return {table: self.describe_table(table) for table in self.show_tables(limit=limit)}

    def get_table_samples(
            self,
            row_limit: int = DEFAULT_SAMPLE_LIMIT,
            table_limit: int = DEFAULT_TABLE_LIMIT
    ) -> dict[str, str]:
        return {
            table: self.sample_table(table, limit=row_limit)
            for table in self.show_tables(limit=table_limit)
        }


def get_top_tables(index: DataFrame, query: str, n: int = 5) -> DataFrame:
    query_embed = get_embedding(query, engine=EMBEDDING_MODEL)
    return index.sort_values(
        by='embedding',
        key=lambda s: s.map(lambda e: cosine_similarity(query_embed, e)),
        ascending=False
    ).iloc[:n]


def get_sql_query(query: str, summary: str) -> str:
    return openai.ChatCompletion.create(
        model=SQL_GENERATION_MODEL,
        messages=[
            SYSTEM_PROMPT_MESSAGE,
            {
                "role": "system",
                "content": f"""
                    1. Write a valid SQL query that answers the question/command: {query}.
                    2. Use the following tables: {summary}.
                    3. Include absolutely nothing but the SQL query, especially not markdown syntax. It should start with WITH or SELECT and end with ;
                    4. Always include a LIMIT clause and include no more than but potentially less than 100 rows.
                    5. If the question/command can't be accurately answered by querying the database, return nothing at all.
                    6. If the values in a column are only meaningful to a computer and not to a domain expert, do not include it. For example, prefer using names vs. IDs.
                """
            }
        ]
    ).choices[0].message.content


def build_columns_index(table_columns):
    df = pd.DataFrame([
        {
            "table": table,
            "columns": columns,
            "summary": f"Table: {table}\nColumns: \n{columns}",
        }
        for table, columns in table_columns.items()
    ])
    return build_index(df, 'summary')


def build_sample_index(table_samples):
    df = pd.DataFrame([
        {
            "table": table,
            "sample": sample,
            "summary": f"Table: {table}\nColumns: \n{sample}",
        }
        for table, sample in table_samples.items()
    ])
    return build_index(df, 'summary')


def build_index(df, embed_col):
    response = openai.Embedding.create(
        input=list(df[embed_col]),
        engine=EMBEDDING_MODEL
    )
    df['embedding'] = [item.embedding for item in response.data]
    return df


def generate_sql(index, question) -> str:
    top_tables = get_top_tables(index, question)
    summary = SUMMARY_DELIMITER.join(top_tables['summary'])
    answer = get_sql_query(question, summary)
    return answer.removeprefix('```sql').removeprefix('```SQL').removesuffix('```')


def generate_valid_sql(conn, index, question) -> str:
    while True:
        statement = generate_sql(index, question)
        try:
            conn.cursor().describe(statement)
        except Exception:
            print(f"BAD: {statement}")
        else:
            print(f"GOOD: {statement}")
            return statement


def e2e_qa(conn, index, question):
    statement = generate_valid_sql(conn, index, question)
    print(statement)
    answer = conn.cursor().execute(statement).fetch_pandas_all()
    print(answer.to_markdown())


# Examples?


def main():
    conn = get_snowflake_connection()
    db = DataBoss(conn)
    table_samples = db.get_table_samples(table_limit=100)
    # Build an index of tables from embeddings of the table name and columns
    # index = build_columns_index(table_samples)
    index = build_sample_index(table_samples)
    question1 = "Show me the customers with the highest credit rating."
    e2e_qa(conn, index, question1)
    question2 = "Show me how many orders we've had in the last three months."
    e2e_qa(conn, index, question2)
    # Get tables and descriptions for query


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
