import glob
import json
import math
import random
from os import getenv
from typing import Container
from typing import Literal
from typing import Sequence

import altair as alt
import numpy as np
import openai
import pandas as pd
from openai.embeddings_utils import cosine_similarity
from openai.embeddings_utils import get_embedding
from pandas import DataFrame
from pandas import Series
from pydantic import BaseModel
from snowflake.connector import DictCursor
from snowflake.connector import SnowflakeConnection
from snowflake.connector import connect

# These have really high token counts
CHART_CONTEXT_EXCLUDE_EXAMPLES = [
    "layer_likert",  # 4532
    "isotype_bar_chart",  # 3924
    "interactive_dashboard_europe_pop",  # 3631
    "layer_line_window"  # 3520
]

CHART_DEFAULT_CONTEXT_ROW_LIMIT = 10

VEGA_LITE_CHART_PROMPT_FORMAT = """
Examples:
{examples}
Generate a Vega Lite chart that answers the user's question from the data.
User question: {question}
Generated SQL query: {query}
SQL query result (these will be automatically included in data.values):
{result}
Vega-Lite definition following schema at https://vega.github.io/schema/vega-lite/v5.json:
"""

DEFAULT_CONTEXT_COLUMN_LIMIT = 5

LLM_PANDAS_DISPLAY_OPTIONS = (
    "display.max_columns", 100,
    "display.width", 1000,
    "display.max_colwidth", 16
)

DEFAULT_CONTEXT_ROW_LIMIT = 5

DEFAULT_CONTEXT_TABLE_LIMIT = 5

TABLE_SUMMARY_FORMAT = """
Table: {table}
Sample: {sample}
"""

SQL_GENERATION_MODEL = 'gpt-4'

DEFAULT_SAMPLE_LIMIT = 10
DEFAULT_TABLE_LIMIT = 100

SYSTEM_PROMPT_MESSAGE = {
    "role": "system",
    "content": "You're a helpful assistant powering a BI tool. " \
               "Part of your work is generating structured output such as JSON or SQL."
}

SQL_EMBEDDING_MODEL = 'text-embedding-ada-002'
CHART_EMBEDDING_MODEL = 'text-embedding-ada-002'
SUMMARY_DELIMITER = '\n' + '-' * 128 + '\n'


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# Planning: Determine the list of columns and the chart type


def get_connection():
    return connect(
        account=getenv('SF_ACCOUNT'),
        user=getenv('SF_USER'),
        password=getenv('SF_PASSWORD'),
        role=getenv('SF_ROLE'),
        database=getenv('SF_DATABASE'),
        schema=getenv('SF_SCHEMA')
    )


class DatabaseCrawler:
    def __init__(self, connection: SnowflakeConnection):
        self.connection = connection

    def show_tables(self, limit: int = DEFAULT_TABLE_LIMIT) -> list[str]:
        cursor = self.connection.cursor(cursor_class=DictCursor)
        return [table['name'] for table in cursor.execute("show tables;").fetchall()[:limit]]

    def describe_table(self, table: str) -> DataFrame:
        cursor = self.connection.cursor(cursor_class=DictCursor)
        columns = cursor.execute("describe table identifier(%(table_name)s);",
                                 {"table_name": table}).fetchall()
        df = pd.DataFrame(columns)
        df = df[[c for c in df.columns if df[c].notna().any()]]
        return df

    def sample_table(self, table: str, limit: int = DEFAULT_SAMPLE_LIMIT) -> DataFrame:
        cursor = self.connection.cursor(cursor_class=DictCursor)
        columns = cursor.execute(
            "select * from identifier(%(table_name)s) sample (%(limit)s rows);",
            {"table_name": table, "limit": limit}).fetchall()
        df = pd.DataFrame(columns)
        df = df[[c for c in df.columns if df[c].notna().any()]]
        return df

    def get_table_descriptions(self, n: int = DEFAULT_TABLE_LIMIT) -> dict[str, DataFrame]:
        return {table: self.describe_table(table) for table in self.show_tables(limit=n)}

    def get_table_samples(
            self,
            n_rows: int = DEFAULT_SAMPLE_LIMIT,
            n_tables: int = DEFAULT_TABLE_LIMIT
    ) -> dict[str, DataFrame]:
        return {
            table: self.sample_table(table, limit=n_rows)
            for table in self.show_tables(limit=n_tables)
        }

    def get_index_data(self) -> "SQLIndexData":
        return SQLIndexData(
            samples=self.get_table_samples(),
            descriptions=self.get_table_descriptions(),
        )

    def get_index(self) -> "SQLIndex":
        return SQLIndex.from_data(self.get_index_data())


def generate_completion(system_prompt, model: str = SQL_GENERATION_MODEL):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt}
        ]
    ).choices[0].message.content
    return response


def postprocess_generated_sql(answer):
    return answer.removeprefix('```sql').removeprefix('```SQL').removesuffix('```')


def chat_summarize_data(df, question, query):
    with pd.option_context(*LLM_PANDAS_DISPLAY_OPTIONS):
        return generate_completion(f"""
            User's question: {question}.
            Generated SQL query: {query}
            SQL query result set: {df}
            Answer to the user's question:
        """, model='gpt-3.5-turbo')


class SQLIndexData(BaseModel):
    # Maps table name to output of `DESCRIBE TABLE`
    descriptions: dict[str, DataFrame]
    # Maps table name to a sample of that table.
    samples: dict[str, DataFrame]

    class Config:
        arbitrary_types_allowed = True

    def subset(self, tables: Container[str]) -> "SQLIndexData":
        return SQLIndexData(
            descriptions={table: description for table, description in self.descriptions.items()
                          if table in tables},
            samples={table: sample for table, sample in self.samples.items()
                     if table in tables},
        )


def pd_vss_lookup(index: DataFrame, query: np.array, n: int) -> Series:  # ?
    # Couldn't help myself but vectorize this
    index_norm = np.sqrt((index ** 2).sum(axis=1))
    query_norm = (query @ query) ** 0.5
    similarity = index @ query / (index_norm * query_norm)
    table_scores = similarity.sort_values(ascending=False) \
        .head(n)
    return table_scores


class SQLIndex(BaseModel):
    data: SQLIndexData
    embeddings: DataFrame

    class Config:
        arbitrary_types_allowed = True

    def subset(self, tables: Container[str]) -> "SQLIndex":
        return SQLIndex(
            data=self.data.subset(tables),
            embeddings=self.embeddings.loc[tables]
        )

    @classmethod
    def from_data(
            cls,
            data: SQLIndexData,
            strategy: Literal["sample", "semantic"] = "sample"
    ) -> "SQLIndex":
        # [table, column] -> [text]
        if strategy == "sample":
            df = cls.get_text_sample(data)
        elif strategy == "semantic":
            df = cls.get_text_semantic(data)
        response = openai.Embedding.create(input=list(df['text']), engine=SQL_EMBEDDING_MODEL)
        embeddings = df.assign(embedding=[item.embedding for item in response.data]) \
            .embedding \
            .apply(lambda s: pd.Series(s)) \
            .rename(lambda i: f'dim_{i}', axis=1)
        return cls(data=data, embeddings=embeddings)

    @classmethod
    def get_text_semantic(cls, data: SQLIndexData) -> DataFrame:
        openai.ChatCompletion.create()
        df = DataFrame([
            {'table': table, 'column': column, 'text': df[[column]].to_markdown(index=False)}
            for table, df in data.samples.items() for column in df.columns
        ]).set_index(['table', 'column'])
        return df

    @classmethod
    def get_text_sample(cls, data: SQLIndexData) -> DataFrame:
        df = DataFrame([
            {'table': table, 'column': column, 'text': df[[column]].to_markdown(index=False)}
            for table, df in data.samples.items() for column in df.columns
        ]).set_index(['table', 'column'])
        return df

    def top_tables(self, query: str, n: int = 5, mode: Literal['sum', 'mean'] = 'sum') -> DataFrame:
        query_embedding = np.array(get_embedding(query, engine=SQL_EMBEDDING_MODEL))
        table_embeddings = self.embeddings.groupby(level='table').agg(mode)
        table_scores = pd_vss_lookup(table_embeddings, query_embedding, n)
        return table_scores.index.tolist()

    def top_columns(self, query: str, n: int = 5) -> list[str]:
        query_embedding = np.array(get_embedding(query, engine=SQL_EMBEDDING_MODEL))
        column_scores = pd_vss_lookup(self.embeddings, query_embedding, n)
        return [column for table, column in column_scores.index.tolist()]


class SQLGenerator:
    def __init__(self, connection: SnowflakeConnection, index: SQLIndex):
        self.connection = connection
        self.index = index

    def generate_valid_query(self, question: str) -> str:
        errors = []
        n = 1
        while True:
            prompt = self.build_prompt(question, errors)
            print(prompt)
            statement = self.generate(prompt)
            try:
                self.validate(statement)
            except Exception as e:
                errors.append(f"Query {n}/3: {statement}. Error: {e}")
                print(f"BAD: {statement}")
                n += 1
                if n > 3:
                    raise
            else:
                print(f"GOOD: {statement}")
                return statement

    def generate(self, prompt):
        response = generate_completion(prompt)
        answer = response
        statement = postprocess_generated_sql(answer)
        return statement

    def build_prompt(self, question, errors):
        summary = self.get_context(question)
        system_prompt = f"""
                1. Write a valid SQL query that answers the question/command: {question}.
                2. Use the following tables: {summary}.
                3. Include absolutely nothing but the SQL query, especially not markdown syntax. It should start with WITH or SELECT and end with ;
                4. Always include a LIMIT clause and include no more than but potentially less than 100 rows.
                5. If the question/command can't be accurately answered by querying the database, return nothing at all.
                6. If the values in a column are only meaningful to a computer and not to a domain expert, do not include it. For example, prefer using names vs. IDs.
                {self.get_error_prompt(errors)}
            """
        return system_prompt

    def get_context(
            self,
            question: str,
            n_tables: int = DEFAULT_CONTEXT_TABLE_LIMIT,
            n_rows: int = DEFAULT_CONTEXT_ROW_LIMIT,
            n_columns: int = DEFAULT_CONTEXT_COLUMN_LIMIT
    ) -> str:
        top_tables = self.index.top_tables(question, n=n_tables)
        with pd.option_context(*LLM_PANDAS_DISPLAY_OPTIONS):
            table_samples = {}
            for table in top_tables:
                table_index = self.index.subset([table])
                columns = table_index.top_columns(question, n_columns)
                table_samples[table] = self.index.data.samples[table][columns].head(n_rows)
            summary = SUMMARY_DELIMITER.join([
                TABLE_SUMMARY_FORMAT.format(
                    table=table,
                    sample=sample
                )
                for table, sample in table_samples.items()
            ])
        return summary

    def validate(self, statement: str):
        self.connection.cursor().describe(statement)

    def get_error_prompt(self, messages) -> str:
        if messages:
            return "Previous attempts:\n" + "\n".join(messages)
        return ""


class ChartIndexData(BaseModel):
    ...

    @classmethod
    def load(cls):
        pass


class ChartIndex(BaseModel):
    # [chart_id] -> [specification]
    specifications: DataFrame
    # [chart_id] -> [dim_0, dim_1, ..., dim_n]
    embeddings: DataFrame

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def create(cls) -> "ChartIndex":
        vl_example_files = glob.glob('data/vega-lite/examples/*.vl.json')
        vl_example_names = [
            example.removeprefix('data/vega-lite/examples/').removesuffix('.vl.json')
            for example in vl_example_files
        ]
        # vl_example_specs = [json.load(open(example, 'r')) for example in vl_example_files]
        vl_example_specs_json = [open(example, 'r').read() for example in vl_example_files]
        response = openai.Embedding.create(
            input=vl_example_specs_json,
            engine=CHART_EMBEDDING_MODEL
        )
        embeddings = [item.embedding for item in response.data]
        embeddings_df = DataFrame(
            embeddings,
            index=vl_example_names
        ).rename(lambda i: f"dim_{i}", axis=1)
        specifications_df = DataFrame(
            vl_example_specs_json,
            index=vl_example_names,
            columns=['specification']
        ).drop(CHART_CONTEXT_EXCLUDE_EXAMPLES)
        return ChartIndex(embeddings=embeddings_df, specifications=specifications_df)

    def top_charts(self, question: str, data: DataFrame) -> Series:
        """
        Finds the most relevant chart specifications to the given question and data.
        :return: Series [chart_id] -> specification
        """
        embedding_query_string = json.dumps(
            {
                "title": question,
                "data": {
                    "values": data.head(CHART_DEFAULT_CONTEXT_ROW_LIMIT).to_dict(orient="records")
                }
            }
        )
        embedding_query = np.array(
            get_embedding(embedding_query_string, engine=CHART_EMBEDDING_MODEL)
        )
        chart_ids = pd_vss_lookup(self.embeddings, embedding_query, n=3).index.tolist()
        return self.specifications.specification.loc[chart_ids]


def get_bootstrap_questions() -> list[str]:
    with open('data/TPCDS_SF10TCL-queries.json', 'r') as f:
        return json.load(f)


def get_random_questions():
    return random.choices(get_bootstrap_questions(), k=10)


def extract_json(text: str) -> dict:
    raw = text[text.index('{'):text.rindex('}') + 1]
    return json.loads(raw)


class ChartGenerator:
    def __init__(self, index: ChartIndex):
        self.index = index

    def generate(self, question: str, query: str, result: DataFrame) -> dict:
        data_values = result.to_dict(orient='records')
        with pd.option_context(*LLM_PANDAS_DISPLAY_OPTIONS):
            prompt = VEGA_LITE_CHART_PROMPT_FORMAT.format(
                result=data_values[:CHART_DEFAULT_CONTEXT_ROW_LIMIT],
                question=question,
                query=query,
                examples=self.index.top_charts(question, result).tolist()
            )
        print(prompt)
        completion = generate_completion(prompt)
        print(completion)
        specification = extract_json(completion)
        # TODO: Validate using JSON Schema, feed errors back into the model.
        # specification['data'] = {'values': data_values}
        return specification
