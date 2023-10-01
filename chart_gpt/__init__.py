import glob
import json
from datetime import date
from datetime import datetime
from os import getenv

import numpy as np
import openai
import pandas as pd
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

LLM_PANDAS_DISPLAY_OPTIONS = (
    "display.max_columns", 100,
    "display.width", 1000,
    "display.max_colwidth", 16
)

SQL_GENERATION_MODEL = 'gpt-4'

DEFAULT_SAMPLE_LIMIT = 3

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
        schema=getenv('SF_SCHEMA'),
        # application='ChartGPT 0.0.0'
    )


def get_table_context(data: "SQLIndexData") -> Series:
    descriptions = pd.concat([data.descriptions, data.foreign_keys])
    ddl = descriptions.groupby(axis=1, level='table').apply(get_create_table)
    select_x = data.samples.groupby(axis=1, level='table').apply(get_select_x)
    context = ddl + '\n' + select_x
    return context


class DatabaseCrawler:
    def __init__(self, connection: SnowflakeConnection):
        self.connection = connection

    def show_tables(self) -> list[str]:
        cursor = self.connection.cursor(cursor_class=DictCursor)
        return [table['name'] for table in cursor.execute("show tables;").fetchall()]

    def describe_table(self, table: str) -> DataFrame:
        cursor = self.connection.cursor(cursor_class=DictCursor)
        columns = cursor.execute("describe table identifier(%(table_name)s);",
                                 {"table_name": table}).fetchall()
        df = pd.DataFrame(columns).set_index('name').dropna(axis="columns").transpose()
        return df

    def sample_table(self, table: str, limit: int = DEFAULT_SAMPLE_LIMIT) -> DataFrame:
        cursor = self.connection.cursor(cursor_class=DictCursor)
        columns = cursor.execute(
            "select * from identifier(%(table_name)s) sample (%(limit)s rows);",
            {"table_name": table, "limit": limit}).fetchall()
        df = pd.DataFrame(columns)
        df = df[[c for c in df.columns if df[c].notna().any()]]
        return df

    def get_table_descriptions(self) -> DataFrame:
        table_names = self.show_tables()
        return pd.concat(
            [self.describe_table(table) for table in table_names],
            keys=table_names,
            names=['table', 'column'],
            axis='columns'
        )

    def get_table_samples(self, n_rows: int = DEFAULT_SAMPLE_LIMIT) -> DataFrame:
        table_names = self.show_tables()
        return pd.concat(
            [self.sample_table(table, limit=n_rows) for table in table_names],
            keys=table_names,
            names=['table', 'column'],
            axis='columns'
        )

    def get_index_data(self) -> "SQLIndexData":
        samples = self.get_table_samples()
        descriptions = self.get_table_descriptions()
        foreign_keys = self.get_foreign_keys()
        return SQLIndexData(
            samples=samples,
            descriptions=descriptions,
            foreign_keys=foreign_keys,
        )

    def get_foreign_keys(self) -> DataFrame:
        database, schema = self.connection.cursor().execute(
            "select current_database(), current_schema();").fetchone()
        cursor = self.connection.cursor(cursor_class=DictCursor)
        return DataFrame(
            cursor.execute(f"show imported keys in schema {database}.{schema};").fetchall()
        ).drop_duplicates(['fk_table_name', 'fk_column_name']) \
            .set_index(['fk_table_name', 'fk_column_name']) \
            .rename_axis(['table', 'column']) \
            [['pk_table_name', 'pk_column_name']] \
            .transpose()

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


def postprocess_generated_sql(answer: str) -> str:
    start, stop = '```sql', '```'
    start_index = answer.index(start) + len(start)
    stop_index = answer.index(stop, start_index)
    return answer[start_index:stop_index].strip()


def chat_summarize_data(df, question, query):
    with pd.option_context(*LLM_PANDAS_DISPLAY_OPTIONS):
        return generate_completion(f"""
            User's question: {question}.
            Generated SQL query: {query}
            SQL query result set: {df}
            Answer to the user's question:
        """, model='gpt-3.5-turbo')


class SQLIndexData(BaseModel):
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
    # (table) -> dim_0, dim_1, ..., dim_n
    embeddings: DataFrame
    context: Series

    class Config:
        arbitrary_types_allowed = True

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


class SQLGenerator:
    def __init__(self, connection: SnowflakeConnection, index: SQLIndex):
        self.connection = connection
        self.index = index

    def generate_valid_query(self, question: str) -> str:
        errors = []
        n = 1
        while True:
            prompt = self.build_prompt(question, errors)
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
        ).rename(lambda i: f"dim_{i}", axis=1).drop(CHART_CONTEXT_EXCLUDE_EXAMPLES)
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
            },
            default=json_dumps_default
        )
        embedding_query = np.array(
            get_embedding(embedding_query_string, engine=CHART_EMBEDDING_MODEL)
        )
        chart_ids = pd_vss_lookup(self.embeddings, embedding_query, n=3).index.tolist()
        return self.specifications.specification.loc[chart_ids]


def extract_json(text: str, start: str = '{', stop: str = '}') -> dict:
    raw = text[text.index(start):text.rindex(stop) + 1]
    return json.loads(raw)


def json_dumps_default(o):
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    raise TypeError(f"Object {o} not JSON serializable")


class ChartGenerator:
    def __init__(self, index: ChartIndex):
        self.index = index

    def generate(self, question: str, query: str, result: DataFrame) -> dict:
        data_values = result.to_dict(orient='records')
        with pd.option_context(*LLM_PANDAS_DISPLAY_OPTIONS):
            prompt = VEGA_LITE_CHART_PROMPT_FORMAT.format(
                result=json.dumps(data_values[:CHART_DEFAULT_CONTEXT_ROW_LIMIT],
                                  default=json_dumps_default),
                question=question,
                query=query,
                examples=json.dumps(
                    self.index.top_charts(question, result).map(json.loads).tolist(),
                    default=json_dumps_default
                )
            )
        completion = generate_completion(prompt)
        specification = extract_json(completion)
        specification['data'] = {'values': data_values}
        return specification


def get_column_ddl(column: Series) -> str:
    # https://docs.snowflake.com/en/sql-reference/sql/create-table#syntax
    parts = [column.name[1], column['type']]
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
